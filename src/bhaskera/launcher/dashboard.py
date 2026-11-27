"""
bhaskera.launcher.dashboard
===========================
``bhaskera-dashboard`` / ``bhaskera-start`` — start the MLflow UI on the
login node and tell you the exact ``ssh -L`` tunnel command to run on your
laptop.  No Prometheus, no Grafana, no Docker, no root.

Design
------
* The training process (GPU node) pushes metrics to a file-backed MLflow
  store under ``$HOME/mlflow-runs``.  Login nodes share ``$HOME`` with
  compute nodes on most HPC clusters, so the UI reads live data without
  any network plumbing.
* Config persists to ``~/.bhaskera/mlflow-ui.json`` so subsequent runs
  need zero arguments.
* Subcommands: ``start`` (default), ``stop``, ``status``, ``tunnel``, and ``ray`` (the commands for
  the Ray Dashboard of a running job).

Typical workflow
----------------
::

    # On the login node — once:
    bhaskera-dashboard

    # From your laptop:
    ssh -L 5000:localhost:5000 ldls-iiitd@<login-node>
    open http://localhost:5000

    # Or use the exact command the tool prints.
"""
from __future__ import annotations

from bhaskera import paths

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from bhaskera.launcher import launch, ray_hub

logger = logging.getLogger("bhaskera.dashboard")


_DEFAULT_PORT       = 5000
_CONFIG_DIR         = paths.state_dir()
_CONFIG_PATH        = _CONFIG_DIR / "mlflow-ui.json"
_PID_FILE           = _CONFIG_DIR / "mlflow-ui.pid"
_LOG_FILE           = _CONFIG_DIR / "mlflow-ui.log"
_DEFAULT_STORE      = paths.mlflow_store()


@dataclass
class DashboardConfig:
    """Everything bhaskera-dashboard needs to start the MLflow UI.

    Saved to ``~/.bhaskera/mlflow-ui.json`` after the first run so
    subsequent invocations need zero arguments.
    """
    store:      str  = _DEFAULT_STORE
    port:       int  = _DEFAULT_PORT
    host:       str  = "127.0.0.1"   # MLflow has no login: keep it on this machine; the SSH tunnel connects locally
    login_node: str  = ""      # e.g. login08.cluster.example.edu
    user:       str  = ""      # remote SSH user, if different from $USER
    login_port: int  = 0       # SSH port of the login address, 0 for the default (22)


    @classmethod
    def load(cls) -> "DashboardConfig":
        if _CONFIG_PATH.exists():
            try:
                return cls(**json.loads(_CONFIG_PATH.read_text()))
            except Exception as e:
                logger.debug(f"Could not read saved config: {e}")
        return cls()

    def save(self) -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))


def _read_pid() -> Optional[int]:
    try:
        return int(_PID_FILE.read_text().strip())
    except Exception:
        return None


def _write_pid(pid: int) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(pid))


def _clear_pid() -> None:
    _PID_FILE.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_start(cfg: DashboardConfig, show_tunnel: bool = True) -> None:
    """Start the MLflow UI as a detached background process. ``show_tunnel`` False when the dashboard links MLflow itself."""
    pid = _read_pid()
    if pid and _pid_alive(pid):
        print(f"MLflow UI is already running (PID {pid}). Use 'bhaskera-dashboard status'.")
        if show_tunnel:
            _print_tunnel(cfg)
        return

    if cfg.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: MLflow has no login and will listen on {cfg.host}: anyone who can reach this machine can read "
              "and delete your runs. Use --host 127.0.0.1 and the SSH tunnel instead.")

    if not shutil.which("mlflow"):
        _die(
            "mlflow not found on PATH.\n"
            "Install it with:  pip install mlflow\n"
            "Then re-run:      bhaskera-dashboard"
        )

    store_path = Path(cfg.store)
    try:
        store_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _die(f"Cannot create MLflow store directory {cfg.store!r}: {e}")

    cmd = [
        "mlflow", "ui",
        "--backend-store-uri", f"file://{store_path.resolve()}",
        "--host", cfg.host,
        "--port", str(cfg.port),
    ]

    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fd = open(_LOG_FILE, "a")

    # MLflow >= 3 refuses a file:// store unless this is set (the training jobs set it in
    # prod_train.sbatch; the UI must too, or every page request fails).
    env = {**os.environ, "MLFLOW_ALLOW_FILE_STORE": os.environ.get("MLFLOW_ALLOW_FILE_STORE", "true")}
    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=log_fd,
        env=env,
        start_new_session=True, 
    )
    _write_pid(proc.pid)

    time.sleep(1.5)
    if not _pid_alive(proc.pid):
        _die(
            f"MLflow UI exited immediately. Check the log:\n  {_LOG_FILE}\n"
            "Common cause: port already in use. Try --port <other_port>."
        )

    print("─" * 60)
    print("  MLflow UI started")
    print(f"  PID   : {proc.pid}")
    print(f"  Store : {cfg.store}")
    print(f"  Port  : {cfg.port}  (on {socket.gethostname()})")
    print(f"  Log   : {_LOG_FILE}")
    print()
    if show_tunnel:
        _print_tunnel(cfg)
    print("─" * 60)
    print()
    if show_tunnel:
        print("Stop with:  bhaskera-dashboard stop")
        print()


def cmd_stop() -> None:
    """Kill the background MLflow UI process."""
    pid = _read_pid()
    if not pid:
        print("No MLflow UI PID recorded — nothing to stop.")
        return
    if not _pid_alive(pid):
        print(f"PID {pid} is no longer running. Clearing stale record.")
        _clear_pid()
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(10):
        time.sleep(0.3)
        if not _pid_alive(pid):
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _clear_pid()
    print(f"MLflow UI (PID {pid}) stopped.")


def cmd_status(cfg: DashboardConfig) -> None:
    """Print whether the MLflow UI is running and the store details."""
    pid = _read_pid()
    if pid and _pid_alive(pid):
        print(f"  Status : RUNNING  (PID {pid})")
    elif pid:
        print(f"  Status : STOPPED  (stale PID {pid})")
    else:
        print("  Status : NOT STARTED")
    print(f"  Store  : {cfg.store}")
    print(f"  Port   : {cfg.port}")
    print(f"  Log    : {_LOG_FILE}")
    _print_tunnel(cfg)


def cmd_tunnel(cfg: DashboardConfig) -> None:
    """Print the ssh -L tunnel command (and optionally open it)."""
    _print_tunnel(cfg, verbose=True)


def ssh_target(cfg: DashboardConfig) -> str:
    """`user@host` to log in to from a laptop: the saved config first, then $BHASKERA_LOGIN_HOST (set by the
    module) and $USER, and the name of the node we are on as a last resort."""
    user = cfg.user or os.environ.get("USER", "")
    host = cfg.login_node or os.environ.get("BHASKERA_LOGIN_HOST", "") or socket.gethostname()
    return f"{user}@{host}" if user else host


def ssh_options(cfg: DashboardConfig) -> str:
    """`-p <port> ` when logging in to the cluster needs a non-default SSH port, otherwise an empty string."""
    port = cfg.login_port
    if not port:
        try:
            port = int(os.environ.get("BHASKERA_LOGIN_PORT", "0") or 0)
        except ValueError:
            port = 0
    return f"-p {port} " if port else ""


def _hop_port(port: int) -> int:
    """The port used on the first login node when the tunnel goes on to a named node (any free port will do)."""
    return port + 10000 if port <= 55000 else port - 10000


def _this_node() -> str:
    return socket.gethostname().split(".")[0]


def tunnel_command(cfg: DashboardConfig, local_port: int, remote_port: int, node: str = "", remote: str = "") -> str:
    """The ssh command to run on a laptop so that ``localhost:<local_port>`` reaches ``<remote_port>`` on ``node``.

    The cluster's address may hand out a different login node each time, so a service that lives on one login node is reached in
    two hops: log in, then ssh on to that node. (``ssh -J`` is not used: the login nodes refuse to forward to other machines.)
    With no ``node``, or when the address already is that node, one hop is enough. ``remote`` is a command to run on the node."""
    target = ssh_target(cfg)
    opts = ssh_options(cfg)
    if not node or node == target.rpartition("@")[2].split(".")[0]:
        cmd = f"ssh {opts}-L {local_port}:localhost:{remote_port} {target}"
        return cmd + (f" \"{remote}\"" if remote else "")
    hop = _hop_port(local_port)
    inner = f"ssh -t -t -L {hop}:localhost:{remote_port} {node}" + (f" \\\"{remote}\\\"" if remote else "")
    return f"ssh {opts}-t -t -L {local_port}:localhost:{hop} {target} \"{inner}\""


def _print_tunnel(cfg: DashboardConfig, *, verbose: bool = False) -> None:
    local_url  = f"http://localhost:{cfg.port}"
    # the page runs on this login node, so the tunnel must end on this node (the address alone may land on another one)
    tunnel_cmd = tunnel_command(cfg, cfg.port, cfg.port, node=_this_node())

    if verbose:
        print()
    print(f"  From your laptop (leave it open):")
    print(f"    {tunnel_cmd}")
    print(f"    then open {local_url}")
    if verbose:
        print()


# Ray Dashboard of an ExaAI job
#
# The Ray head (and with it the dashboard) runs on the first node of a job, and nobody knows in advance when
# a job will start or on which nodes. `bhaskera-dashboard ray` therefore finds the job itself, waits for it to
# get nodes, and forwards the dashboard of its first node to a fixed port of the login node. The laptop only
# ever forwards two fixed ports to the login node (derived from the user id, so users who share a login node
# do not collide), which is why the command to type never changes.

_LAPTOP_PORT = 8265                 # the one address on the laptop: http://localhost:8265
_JOB_COMMENT = "bhaskera"           # submit.sh puts this in the Slurm job comment


def _slurm(cmd: list[str]) -> str:
    """Output of a Slurm command, or an empty string when it is unavailable or fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def hub_port() -> int:
    """Login-node port of the hub: fixed for a user, different between users that share a login node."""
    return 20000 + os.getuid() % 10000


def mlflow_port(cfg: DashboardConfig) -> int:
    """Login-node port of the MLflow UI: the saved or chosen port, else one of your own (like the Ray ports),
    so that users who share a login node do not collide."""
    if cfg.port and cfg.port != _DEFAULT_PORT:
        return cfg.port
    return 40000 + os.getuid() % 10000


def _active_job_ids() -> Optional[set[str]]:
    """Ids of this user's Slurm jobs that are not finished, or None when Slurm cannot be asked."""
    try:
        r = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return set(r.stdout.split()) if r.returncode == 0 else None


def _unfinished_runs(store: Path) -> list[tuple[str, str]]:
    """(run id, Slurm job id) of every run in the file store that is still marked RUNNING. Reads only the
    small status files, so it stays fast with hundreds of runs."""
    found = []
    for meta in store.glob("*/*/meta.yaml"):
        try:
            if "\nstatus: 1\n" not in meta.read_text():
                continue
            job = (meta.parent / "tags" / "slurm_job").read_text().strip()
        except OSError:
            continue
        if job:
            found.append((meta.parent.name, job))
    return found


def close_orphan_runs(cfg: DashboardConfig) -> int:
    """Mark runs that still say RUNNING although their Slurm job has ended (killed, time limit, node failure)
    as KILLED. Their curves were saved while the job ran; this only fixes the label. Returns the count."""
    store = Path(cfg.store)
    if not store.is_dir():
        return 0
    unfinished = _unfinished_runs(store)
    if not unfinished:
        return 0
    active = _active_job_ids()
    if active is None:
        return 0
    dead = [run for run, job in unfinished if job not in active]
    if not dead:
        return 0
    try:
        from mlflow.tracking import MlflowClient
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        client = MlflowClient(tracking_uri=f"file://{store.resolve()}")
        for run in dead:
            client.set_terminated(run, status="KILLED")
        return len(dead)
    except Exception as e:  # never block the dashboard over a label
        logging.getLogger(__name__).debug("close_orphan_runs: %s", e)
        return 0


def mlflow_page_state(port: int, fetch=None) -> Optional[bool]:
    """What answers on a port: None when nothing does, True for a working MLflow page, False for anything else (for example an
    old MLflow whose web files were removed, which answers with 'Unable to display MLflow UI')."""
    import urllib.error
    import urllib.request
    try:
        if fetch is not None:
            status, body = fetch(port)
        else:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=4) as r:
                status, body = r.status, r.read(4000)
    except urllib.error.HTTPError as e:
        return False if e.code >= 500 else True
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    return status == 200 and b"Unable to display MLflow UI" not in body


def ensure_mlflow(cfg: DashboardConfig) -> None:
    """Start the MLflow UI on this login node unless it already runs. A failure only costs the MLflow page."""
    cfg.port = mlflow_port(cfg)
    cfg.host = "127.0.0.1"          # never keep a saved non-loopback host
    closed = close_orphan_runs(cfg)
    if closed:
        print(f"{closed} MLflow run(s) of finished jobs were still marked running; marked as killed.")
    pid = _read_pid()
    if pid and _pid_alive(pid):
        print(f"MLflow is running on {socket.gethostname()}, port {cfg.port}.")
        return
    if mlflow_page_state(cfg.port) is False:
        # something answers on the port but is not a working MLflow (an old one whose files are gone, say): leave it alone and
        # start a fresh one on a free port
        old = cfg.port
        cfg.port = ray_hub.free_port()
        print(f"The MLflow page on port {old} of {socket.gethostname()} does not work (it is probably an old MLflow whose files were "
              f"removed); starting a new one on port {cfg.port}.")
    try:
        cmd_start(cfg, show_tunnel=False)
    except SystemExit:
        print("MLflow could not be started here; run 'bhaskera-dashboard start --port <free port>' to see why.")


def my_jobs() -> list[dict]:
    """The current user's running and pending jobs."""
    fmt = "%i|%j|%T|%N|%r|%k"
    rows = []
    for line in _slurm(["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "R,PD", "-o", fmt]).splitlines():
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 6:
            rows.append({"id": parts[0], "name": parts[1], "state": parts[2], "nodes": parts[3],
                         "reason": parts[4], "comment": parts[5].strip()})
    return rows


_SERVE_COMMENT = "bhaskera-serve"     # submit.sh eval/serve put these in the Slurm job comment
_KINDS = {_JOB_COMMENT: "train", _SERVE_COMMENT: "serve"}


def hub_jobs(jobs: list[dict], job_id: Optional[str] = None) -> list[dict]:
    """The jobs to link: every running ExaAI training job and chat server (those started with submit.sh carry a Slurm
    comment), or just the one asked for with --job, which may be any running job. Each gets a ``kind``."""
    running = [j for j in jobs if j["state"] == "RUNNING" and j["nodes"]]
    if job_id:
        return [{**j, "kind": _KINDS.get(j["comment"], "train")} for j in running if j["id"] == job_id]
    return [{**j, "kind": _KINDS[j["comment"]]} for j in running if j["comment"] in _KINDS]


def head_node(node_list: str) -> str:
    """First node of a job: the one that runs the Ray head and its dashboard."""
    hosts = _slurm(["scontrol", "show", "hostnames", node_list]).split()
    if hosts:
        return hosts[0]
    if "[" in node_list:                      # scontrol unavailable: read 'cbgpu[0241-0245]' ourselves
        prefix, inside = node_list.split("[", 1)
        first = inside.split(",")[0].split("-")[0].rstrip("]")
        return prefix + first
    return node_list.split(",")[0]


def _remote_command() -> str:
    """What the laptop runs on the login node: load the module, or call this tool by its full path when the
    copy in use was installed by the user and there is no module."""
    module = os.environ.get("BHASKERA_MODULE", "")
    if module:
        return f"bash -lc 'module load {module} && bhaskera-dashboard ray'"
    tool = shutil.which("bhaskera-dashboard") or "bhaskera-dashboard"
    return f"bash -lc '{tool} ray'"


def laptop_command(cfg: DashboardConfig, local_port: int = _LAPTOP_PORT, node: str = "") -> str:
    """The one command to run on a laptop: it logs in and runs this tool, which does the rest. With ``node`` (for example ``login01``),
    the page is started on that login node, whichever one the address would have handed out (see ``tunnel_command``)."""
    return tunnel_command(cfg, local_port, hub_port(), node=node, remote=_remote_command())


def ssh_config_block(cfg: DashboardConfig, local_port: int = _LAPTOP_PORT) -> str:
    """A `~/.ssh/config` entry for the laptop, so that `ssh bhaskera-ray` replaces the long command."""
    target = ssh_target(cfg)
    user, _, host = target.rpartition("@")
    lines = ["Host bhaskera-ray", f"    HostName {host}"]
    if user:
        lines.append(f"    User {user}")
    opts = ssh_options(cfg).split()
    if opts:
        lines.append(f"    Port {opts[1]}")
    lines += [f"    LocalForward {local_port} localhost:{hub_port()}",
              f"    RemoteCommand {_remote_command()}",
              "    RequestTTY yes", "    ServerAliveInterval 30"]
    return "\n".join(lines)


def cmd_ssh_config(cfg: DashboardConfig, local_port: int = _LAPTOP_PORT) -> None:
    print("Add this to ~/.ssh/config on your laptop (once):")
    print()
    print(ssh_config_block(cfg, local_port))
    print()
    print("From then on, this is all you type:")
    print()
    print("    ssh bhaskera-ray")
    print(f"    then open http://localhost:{local_port} in a browser; everything is linked from that page.")


def forward_command(head: str, ray_port: int, grafana_port: int) -> list[str]:
    """The ssh command, run on the login node, that links the login-node ports to the first node of the job."""
    return ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes",
            "-L", f"{ray_port}:localhost:{ray_hub.RAY_PORT}",
            "-L", f"{grafana_port}:localhost:{ray_hub.GRAFANA_PORT}", head]


def forward_command_serve(head: str, local_port: int, remote_port: int) -> list[str]:
    """The ssh command, run on the login node, that links a login-node port to a chat server on the job's first node."""
    return ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes",
            "-L", f"{local_port}:localhost:{remote_port}", head]


def cmd_ray(cfg: DashboardConfig, job: Optional[str], info: bool = False, local_port: int = _LAPTOP_PORT, node: str = "") -> None:
    print()
    print("From your laptop, this one command logs in and keeps everything open")
    print("(the command is always the same; it finds your jobs and their nodes by itself):")
    print()
    print(f"    {laptop_command(cfg, local_port, node)}")
    print()
    print(f"Then open http://localhost:{local_port} in a browser: your running jobs, with their Ray Dashboards,")
    print("charts and MLflow, are linked from that page.")
    print()
    if info:
        return

    ensure_mlflow(cfg)
    print()
    hub = _make_hub(cfg, job, local_port, mlflow_port(cfg))
    try:
        server = ray_hub.serve(hub, hub_port())
    except OSError as e:
        _die(f"cannot listen on the login-node port {hub_port()} ({e}); is another "
             "'bhaskera-dashboard ray' of yours running? Stop it first (Ctrl-C).")
    print(f"Open http://localhost:{local_port} in your browser. Keep this window open; it closes with Ctrl-C.")
    print("Jobs that start later are added by themselves, and MLflow stays available after they end.\n", flush=True)
    _watch(hub, server)


def _make_hub(cfg: DashboardConfig, job: Optional[str], local_port: int, mlflow: Optional[int]) -> "ray_hub.Hub":
    """The hub with its page and Start buttons wired to this user's runs."""
    hub = ray_hub.Hub(lambda: hub_jobs(my_jobs(), job), head_node, forward_command, laptop_port=local_port, serve_forward_cmd=forward_command_serve,
                      mlflow_port=mlflow)
    from . import hub_state
    try:
        from importlib.metadata import version as _version
        _bhaskera_version = _version("bhaskera")
    except Exception:
        _bhaskera_version = ""
    builder = hub_state.StateBuilder(
        hub, cfg.store, os.environ.get("USER", ""), version=_bhaskera_version,
        docs_url=os.environ.get("BHASKERA_DOCS_URL", "https://kishoryd.github.io/ParamRudra-20PF-UserManual/bhaskera/"))
    hub.state_provider, hub.log_provider = builder.build, builder.log
    hub.launcher = launch.Launcher(launch.find_kit(), runs=lambda: hub_state.recent_runs(cfg.store), store=cfg.store, jobs=my_jobs)
    hub.launcher.after_change = lambda: builder.forget("runs")
    return hub


def _start_portal_mlflow(cfg: DashboardConfig, prefix: str) -> Optional[tuple]:
    """MLflow's own pages for this session, served under <portal prefix>/mlflow so that they work inside the portal tab.
    Returns (process, port), or None when MLflow cannot start (the page then simply has no MLflow link)."""
    exe = shutil.which("mlflow")
    if not exe:
        print("MLflow is not installed here; the MLflow pages are not available in this session.", flush=True)
        return None
    store = Path(cfg.store)
    try:
        store.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"MLflow store {cfg.store} is not usable ({e}); the MLflow pages are not available.", flush=True)
        return None
    try:
        close_orphan_runs(cfg)
    except Exception:
        pass
    port = ray_hub.free_port()
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(_LOG_FILE.parent / "mlflow-portal.log", "a")
    env = {**os.environ, "MLFLOW_ALLOW_FILE_STORE": os.environ.get("MLFLOW_ALLOW_FILE_STORE", "true")}
    proc = subprocess.Popen([exe, "ui", "--backend-store-uri", f"file://{store.resolve()}", "--host", "127.0.0.1", "--port", str(port),
                             "--static-prefix", f"{prefix}/mlflow"], stdout=log, stderr=log, env=env)
    for _ in range(60):
        if proc.poll() is not None:
            print(f"MLflow exited at once (see {log.name}); the MLflow pages are not available.", flush=True)
            return None
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return proc, port
        except OSError:
            time.sleep(0.5)
    proc.terminate()
    return None


def cmd_portal(cfg: DashboardConfig, bind: str, port: int) -> None:
    """The dashboard for use behind a web portal (Open OnDemand): reached over the network, protected by a session token that only
    the portal's own link carries (BHASKERA_HUB_TOKEN). MLflow's pages are served inside the same tab; the Ray Dashboard, which cannot
    live under a path prefix, is not offered (the page shows the same job numbers itself)."""
    token = os.environ.get("BHASKERA_HUB_TOKEN", "")
    if len(token) < 16:
        _die("set BHASKERA_HUB_TOKEN to a random string of at least 16 characters (the portal app does this for you)")
    prefix = os.environ.get("BHASKERA_EXTERNAL_PREFIX", "").rstrip("/")
    hub = _make_hub(cfg, None, port, None)
    hub.proxy_token, hub.portal_prefix = token, prefix
    started = _start_portal_mlflow(cfg, prefix)
    if started:
        hub.mlflow_port = started[1]
    workdir = launch.find_kit()
    state = hub.state_provider

    def with_portal_links() -> dict:
        out = state()
        out["portal"] = {"mlflow": bool(hub.mlflow_port), "jobs": "/pun/sys/dashboard/activejobs", "logout": "/logout",
                         "files": f"/pun/sys/dashboard/files/fs{workdir}" if workdir else ""}
        return out

    hub.state_provider = with_portal_links
    try:
        server = ray_hub.serve(hub, port, bind)
    except OSError as e:
        if started:
            started[0].terminate()
        _die(f"cannot listen on {bind}:{port} ({e})")
    print(f"ExaAI is ready on {bind}:{port}. It closes with Ctrl-C or when its job ends.", flush=True)
    try:
        _watch(hub, server, quiet=True)
    finally:
        if started:
            started[0].terminate()


def _watch(hub: "ray_hub.Hub", server, quiet: bool = False) -> None:
    announced_idle = False
    try:
        while True:
            added, removed = hub.refresh()
            for link in added:
                print(f"job {link.job_id} ({link.name}) is running on {link.head}; added to the page.", flush=True)
            for job_id in removed:
                print(f"job {job_id} is no longer running; its Ray Dashboard and Grafana are closed. "
                      "Its metrics stay in MLflow.", flush=True)
            if not quiet and not hub.links() and not announced_idle:
                print("no ExaAI job is running yet; waiting (MLflow already lists finished runs) ...", flush=True)
            announced_idle = not hub.links()
            if hub.stopping.wait(10):                               # the user logged out on the page
                print("Logged out from the page: stopping. Your jobs keep running; run the same command to open the page again.", flush=True)
                break
    except KeyboardInterrupt:
        print()
    finally:
        hub.close()
        server.shutdown()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bhaskera-dashboard",
        description=(
            "Manage the MLflow UI for ExaAI training runs.\n\n"
            "First run: supply --store / --port / --login-node once;\n"
            "subsequent runs need no arguments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "action",
        nargs="?",
        default="start",
        choices=["start", "stop", "status", "tunnel", "ray", "ssh-config", "portal"],
        help="Subcommand (default: start). 'ray' prints the commands for the Ray Dashboard of a running job; 'ssh-config' prints a laptop SSH entry that shortens them.",
    )
    p.add_argument(
        "--login-port",
        type=int,
        default=None,
        metavar="PORT",
        help="SSH port used to log in to the cluster (saved by 'start'; env: BHASKERA_LOGIN_PORT).",
    )
    p.add_argument(
        "--job",
        default=None,
        metavar="ID",
        help="Slurm job id for the 'ray' subcommand (default: your newest ExaAI job, running or queued).",
    )
    p.add_argument(
        "--ray-port",
        type=int,
        default=_LAPTOP_PORT,
        help="Port on your laptop for the Ray Dashboard in the printed command (default: 8265).",
    )
    p.add_argument(
        "--info",
        action="store_true",
        help="With 'ray': only print the laptop command, do not wait for a job.",
    )
    p.add_argument(
        "--store",
        default=None,
        metavar="URI",
        help=f"MLflow file-store path (default: ~/mlflow-runs). "
             f"Env override: MLFLOW_TRACKING_URI.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port for the MLflow UI (default: {_DEFAULT_PORT}). "
             f"Env override: MLFLOW_PORT.",
    )
    p.add_argument(
        "--login-node",
        default=None,
        metavar="HOSTNAME",
        help="Login-node hostname to print in the ssh -L command "
             "(default: current hostname).",
    )
    p.add_argument(
        "--node",
        default="",
        metavar="LOGIN_NODE",
        help="With 'ray': the login node to run the page on, for example login01 (the one with internet). The printed command logs in "
             "and goes on to that node.",
    )
    p.add_argument(
        "--user",
        default=None,
        metavar="USER",
        help="SSH username for the tunnel command (default: omitted).",
    )
    p.add_argument("--bind", default="127.0.0.1", metavar="ADDRESS",
                   help="'portal' only: the address to listen on (the portal's proxy must reach it). Default 127.0.0.1.")
    p.add_argument("--hub-port", type=int, default=None, metavar="PORT", help="'portal' only: the port to listen on.")
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist config to ~/.bhaskera/mlflow-ui.json.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _merge_env(cfg: DashboardConfig) -> DashboardConfig:
    """Let env vars override saved config (env < CLI args < saved)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if uri.startswith("file://") and not cfg.store:
        cfg.store = uri[len("file://"):]
    env_port = os.environ.get("MLFLOW_PORT")
    if env_port and cfg.port == _DEFAULT_PORT:
        try:
            cfg.port = int(env_port)
        except ValueError:
            pass
    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    parser = _build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = DashboardConfig.load()
    _merge_env(cfg)

    if args.store        is not None: cfg.store       = args.store
    if args.port         is not None: cfg.port        = args.port
    if args.login_node   is not None: cfg.login_node  = args.login_node
    if args.user         is not None: cfg.user        = args.user
    if args.login_port   is not None: cfg.login_port  = args.login_port

    if not cfg.store:
        cfg.store = _DEFAULT_STORE
    if not cfg.port:
        cfg.port  = _DEFAULT_PORT

    if not args.no_save and args.action in ("start",):
        cfg.save()

    if args.action == "start":
        cmd_start(cfg)
    elif args.action == "stop":
        cmd_stop()
    elif args.action == "status":
        cmd_status(cfg)
    elif args.action == "tunnel":
        cmd_tunnel(cfg)
    elif args.action == "ssh-config":
        cmd_ssh_config(cfg, args.ray_port)
    elif args.action == "ray":
        cmd_ray(cfg, args.job, args.info, args.ray_port, args.node)
    elif args.action == "portal":
        cmd_portal(cfg, args.bind, args.hub_port or hub_port())


if __name__ == "__main__":
    main()
