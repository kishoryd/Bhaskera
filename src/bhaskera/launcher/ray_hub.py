"""One address for the Ray Dashboards of all your running Bhaskera jobs.

Every Slurm job starts its own Ray cluster, and a Ray Dashboard shows only its own cluster. To see several
jobs without knowing their nodes, `bhaskera-dashboard ray` runs this hub on the login node:

* for each running job it keeps an ssh link from the login node to the first node of that job
  (the Ray Dashboard and Grafana listen there only);
* one small proxy, on the single port the laptop forwards, picks the target from the address that was typed:
  ``http://job-<id>.localhost:8265`` is the Ray Dashboard of that job, ``http://grafana-<id>.localhost:8265``
  its Grafana charts and ``http://mlflow.localhost:8265`` MLflow. Browsers send every ``*.localhost`` name to
  the laptop itself, so no setup is needed;
* ``http://localhost:8265`` is a list of the running jobs with links to all of these.

Jobs that start later are picked up by themselves, jobs that end are dropped.
"""
from __future__ import annotations

import html
import json
import http.client
import http.server
import re
import socket
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import hub_guard, launch
from .serve import port_for_job

RAY_PORT = 8265        # on the first node of a job
GRAFANA_PORT = 3000    # on the first node of a job

_HOST = re.compile(r"^(job|grafana)-(\d+)\.localhost(?::\d+)?$|^(mlflow)\.localhost(?::\d+)?$", re.IGNORECASE)
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
        "transfer-encoding", "upgrade", "host", "origin"}


def target_from_host(host: str) -> Optional[tuple[str, str]]:
    """What an address like ``job-128944.localhost:8265`` asks for: ("job", "128944"), ("grafana", "128944") or
    ("mlflow", ""); None for any other address."""
    m = _HOST.match((host or "").strip())
    if not m:
        return None
    if m.group(3):
        return "mlflow", ""
    return m.group(1).lower(), m.group(2)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class JobLink:
    job_id: str
    name: str
    head: str
    nodes: str
    ray_port: int          # login-node port that reaches the job's Ray Dashboard
    grafana_port: int      # login-node port that reaches the job's Grafana
    proc: object = field(default=None, repr=False)
    kind: str = "train"    # "train" (Ray + Grafana) or "serve" (a chat server for a tuned model)
    serve_port: int = 0    # login-node port that reaches the job's chat server (kind "serve")
    serve_remote_port: int = 0


class Hub:
    """Keeps one ssh link per running job and answers which local port belongs to which job."""

    def __init__(self, list_jobs: Callable[[], list[dict]], head_of: Callable[[str], str],
                 forward_cmd: Callable[[str, int, int], list[str]], *, laptop_port: int = RAY_PORT,
                 mlflow_port: Optional[int] = None,
                 spawn: Callable[..., object] = subprocess.Popen, pick_port: Callable[[], int] = free_port,
                 peer_check: Callable[[tuple, tuple], bool] = hub_guard.same_user_peer,
                 serve_forward_cmd: Optional[Callable[[str, int, int], list[str]]] = None):
        self._list_jobs, self._head_of, self._forward_cmd = list_jobs, head_of, forward_cmd
        self.laptop_port, self.mlflow_port = laptop_port, mlflow_port
        self._spawn, self._pick_port = spawn, pick_port
        self.peer_check = peer_check
        self._serve_forward_cmd = serve_forward_cmd
        self._links: dict[str, JobLink] = {}
        self._lock = threading.Lock()
        # Set by the command: what the home page shows (see hub_state.py) and how it reads a job's log.
        self.state_provider: Optional[Callable[[], dict]] = None
        self.log_provider: Optional[Callable[[str, int], dict]] = None
        self.launcher = None   # launch.Launcher: the Start buttons

    def refresh(self) -> tuple[list[JobLink], list[str]]:
        """Bring the links in line with the running jobs. Returns (links added, ids of links removed)."""
        jobs = {j["id"]: j for j in self._list_jobs()}
        added, removed = [], []
        with self._lock:
            for job_id in list(self._links):
                link = self._links[job_id]
                if job_id not in jobs or (link.proc is not None and link.proc.poll() is not None):
                    self._stop(link)
                    del self._links[job_id]
                    removed.append(job_id)
            for job_id, job in jobs.items():
                if job_id in self._links:
                    continue
                head = self._head_of(job["nodes"])
                if job.get("kind") == "serve":
                    if self._serve_forward_cmd is None:
                        continue
                    remote = port_for_job(job_id)
                    local = self._pick_port()
                    proc = self._spawn(self._serve_forward_cmd(head, local, remote))
                    link = JobLink(job_id, job["name"], head, job["nodes"], 0, 0, proc, "serve", local, remote)
                    self._links[job_id] = link
                    added.append(link)
                    continue
                ray_port, grafana_port = self._pick_port(), self._pick_port()
                proc = self._spawn(self._forward_cmd(head, ray_port, grafana_port))
                link = JobLink(job_id, job["name"], head, job["nodes"], ray_port, grafana_port, proc)
                self._links[job_id] = link
                added.append(link)
        return added, removed

    def links(self) -> list[JobLink]:
        with self._lock:
            return sorted(self._links.values(), key=lambda l: int(l.job_id))

    def link(self, job_id: str) -> Optional[JobLink]:
        with self._lock:
            return self._links.get(job_id)

    def close(self) -> None:
        with self._lock:
            for link in self._links.values():
                self._stop(link)
            self._links.clear()

    @staticmethod
    def _stop(link: JobLink) -> None:
        proc = link.proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    def index_html(self) -> str:
        if self.state_provider is not None:
            from .hub_ui import INDEX_HTML
            return INDEX_HTML
        rows = []
        for l in self.links():
            rows.append(
                f"<tr><td>{html.escape(l.job_id)}</td><td>{html.escape(l.name)}</td>"
                f"<td>{html.escape(l.nodes)}</td>"
                f"<td><a href='http://job-{l.job_id}.localhost:{self.laptop_port}/'>Ray Dashboard</a></td>"
                f"<td><a href='http://grafana-{l.job_id}.localhost:{self.laptop_port}/'>Grafana charts</a></td></tr>")
        body = ("".join(rows) if rows else
                "<tr><td colspan=5>No Bhaskera job is running. This page updates when one starts.</td></tr>")
        return (
            "<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=15>"
            "<title>Bhaskera jobs</title>"
            "<style>body{font:15px system-ui,sans-serif;margin:2rem;color:#222}"
            "table{border-collapse:collapse}td,th{padding:.4rem .9rem;border-bottom:1px solid #ddd;text-align:left}"
            "a{color:#0b5cad}</style>"
            "<h2>Your running Bhaskera jobs</h2>"
            "<table><tr><th>Slurm job</th><th>Name</th><th>Nodes</th><th></th><th></th></tr>"
            f"{body}</table>"
            f"<p>Training curves of every run, finished and failed ones too: "
            f"<a href='http://mlflow.localhost:{self.laptop_port}/'>MLflow</a></p>")


def _handler(hub: Hub):
    """Request handler of the proxy: Ray Dashboard, Grafana or MLflow, by the address that was typed."""

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "bhaskera-hub"

        def log_message(self, *args):
            pass

        def _send(self, code: int, body: str, ctype: str = "text/html; charset=utf-8") -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in hub_guard.SECURITY_HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _json(self, payload: dict, code: int = 200) -> None:
            try:
                data = json.dumps(payload, default=str)
            except (TypeError, ValueError):
                return self._send(500, "{}", "application/json")
            self.send_response(code)
            body = data.encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            for k, v in hub_guard.SECURITY_HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _upstream(self, kind: str, job_id: str) -> tuple[Optional[int], int, str]:
            """(login-node port, port the service believes it has, what to call it in messages)."""
            if kind == "mlflow":
                return hub.mlflow_port, hub.mlflow_port or 0, "MLflow"
            link = hub.link(job_id)
            if link is None:
                return None, 0, f"Job {html.escape(job_id)}"
            if kind == "grafana":
                return link.grafana_port, GRAFANA_PORT, f"The charts of job {html.escape(job_id)}"
            return link.ray_port, RAY_PORT, f"The dashboard of job {html.escape(job_id)}"

        def _relay(self, port: int, service_port: int, path: str, body: Optional[bytes], what: str) -> None:
            """Send the request to a service behind an SSH link and pass its answer back as it arrives (streaming works)."""
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
            headers["Host"] = f"localhost:{service_port}"
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
                conn.request(self.command, path, body, headers)
                resp = conn.getresponse()
            except OSError:
                return self._send(502, f"{what} is not up yet; try again in a minute.")
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in _HOP:
                    self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                read = getattr(resp, "read1", resp.read)             # read1 returns what has arrived instead of waiting for a full block
                try:
                    while True:
                        chunk = read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (OSError, http.client.HTTPException):
                    pass
            conn.close()

        def _chat(self, path: str, query: str) -> None:
            """/api/chat?job=ID (POST) and /api/serve?job=ID (GET): the chat server of one of this user's serve jobs."""
            args = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            link = hub.link(args.get("job", ""))
            if link is None or link.kind != "serve" or not link.serve_port:
                return self._json({"error": "no chat server is running for this job"}, 404)
            if path == "/api/serve":
                return self._relay(link.serve_port, link.serve_remote_port, "/health", None, "The chat server")
            if self.command != "POST":
                return self._json({"error": "use POST"}, 405)
            origin = self.headers.get("Origin")
            if (origin and not hub_guard.origin_allowed(origin)) or self.headers.get("X-Requested-With") != "bhaskera":
                return self._json({"error": "this request does not come from the Bhaskera page"}, 403)
            length = int(self.headers.get("Content-Length") or 0)
            if length > hub_guard.MAX_CHAT_BODY:
                return self._json({"error": "request too large"}, 413)
            body = self.rfile.read(length) if length else b"{}"
            self._relay(link.serve_port, link.serve_remote_port, "/v1/chat/completions", body, "The chat server")

        def _launch(self) -> None:
            """/api/launch: GET gives what the form offers, POST starts (or checks) a job from a small form."""
            if hub.launcher is None:
                if self.command == "GET":
                    return self._json({"enabled": False, "reason": "Starting jobs is not available here."})
                return self._json({"error": "not available"}, 404)
            if self.command == "GET":
                return self._json(hub.launcher.options())
            if self.command != "POST":
                return self._json({"error": "use POST"}, 405)
            origin = self.headers.get("Origin")
            if (origin and not hub_guard.origin_allowed(origin)) or self.headers.get("X-Requested-With") != "bhaskera":
                return self._json({"error": "this request does not come from the Bhaskera page"}, 403)
            length = int(self.headers.get("Content-Length") or 0)
            if length > hub_guard.MAX_FORM_BODY:
                return self._json({"error": "request too large"}, 413)
            try:
                form = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(form, dict):
                    raise ValueError
            except ValueError:
                return self._json({"error": "the form could not be read"}, 400)
            try:
                return self._json(hub.launcher.launch(form))
            except launch.LaunchError as e:
                return self._json({"error": str(e)}, 400)

        def _handle(self) -> None:
            if not hub.peer_check(self.client_address, self.server.server_address):
                return self._send(403, "This page is private to the user who started it.")
            if not hub_guard.host_allowed(self.headers.get("Host", "")):
                return self._send(421, "Open the hub as http://localhost:<port>.")
            target = target_from_host(self.headers.get("Host", ""))
            if target is None:
                path, _, query = self.path.partition("?")
                if path in ("/", "/index.html"):
                    return self._send(200, hub.index_html())
                if path == "/api/state" and hub.state_provider is not None:
                    return self._json(hub.state_provider())
                if path == "/api/launch":
                    return self._launch()
                if path in ("/api/chat", "/api/serve"):
                    return self._chat(path, query)
                if path == "/api/log" and hub.log_provider is not None:
                    args = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
                    lines = int(args["lines"]) if args.get("lines", "").isdigit() else 300
                    return self._json(hub.log_provider(args.get("job", ""), lines))
                return self._send(404, "Open the list of jobs at http://localhost first.")
            kind, job_id = target
            if self.command not in hub_guard.SAFE_METHODS:
                origin = self.headers.get("Origin")
                if origin and not hub_guard.origin_allowed(origin):
                    return self._send(403, "Requests from other web sites are not allowed.")
                if not hub_guard.method_allowed(kind, self.command):
                    return self._send(405, "The Ray Dashboard is read-only here; use SSH to submit or stop jobs.")
            port, service_port, what = self._upstream(kind, job_id)
            if port is None:
                return self._send(404, f"{what} is not running.")
            if "upgrade" in self.headers.get("Connection", "").lower():
                return self._send(501, "Live connections are not available through the hub.")
            length = int(self.headers.get("Content-Length") or 0)
            if length > hub_guard.MAX_BODY:
                return self._send(413, "Request too large.")
            body = self.rfile.read(length) if length else None
            self._relay(port, service_port, self.path, body, what)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _handle

    return Handler


class _Server(http.server.ThreadingHTTPServer):
    """A threaded server that serves at most MAX_CONNECTIONS connections at once and drops the rest."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slots = threading.BoundedSemaphore(hub_guard.MAX_CONNECTIONS)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def serve(hub: Hub, port: int) -> http.server.ThreadingHTTPServer:
    """Start the proxy on 127.0.0.1 in its own thread (raises OSError when the port is taken)."""
    srv = _Server(("127.0.0.1", port), _handler(hub))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
