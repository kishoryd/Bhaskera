"""What the Bhaskera home page shows: your jobs with live training numbers, their nodes, recent MLflow runs,
partition limits and faulty nodes.

Everything is read from places that already exist (Slurm, the job log, the Ray Dashboard of each running job, the
MLflow file store, the list of faulty nodes), so the page costs the jobs nothing. The slow parts are cached for a few
seconds. All outside calls are passed in, which keeps this module testable without a cluster.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
RAY_PREFIX = re.compile(r"^\((?:RayTrainWorker|TrainController|raylet|PlacementGroupCleaner)[^)]*\)\s*")
STATUS_NAMES = {"1": "RUNNING", "2": "SCHEDULED", "3": "FINISHED", "4": "FAILED", "5": "KILLED"}
LOG_TAIL_BYTES = 96 * 1024
LOG_HEAD_BYTES = 48 * 1024


def clean(text: str) -> str:
    """The text without colour codes and without Ray's '(RayTrainWorker pid=…)' prefixes."""
    return "\n".join(RAY_PREFIX.sub("", line) for line in ANSI.sub("", text).splitlines())


def parse_duration(text: str) -> Optional[int]:
    """Slurm times ('1-00:00:00', '12:30:05', '7:44') in seconds; None for UNLIMITED or anything unreadable."""
    text = (text or "").strip()
    if not text or text.upper() in ("UNLIMITED", "INFINITE", "N/A", "NOT_SET"):
        return None
    try:
        days = 0
        if "-" in text:
            d, text = text.split("-", 1)
            days = int(d)
        parts = [int(p) for p in text.split(":")]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _num(line: str, name: str) -> Optional[float]:
    m = re.search(rf"\b{re.escape(name)}=([0-9.]+(?:[eE][+-]?\d+)?)", line)
    return float(m.group(1)) if m else None


def parse_training(text: str) -> dict:
    """The newest training numbers in a (cleaned) piece of job log: step, loss, speed, MFU and the last evaluation."""
    out: dict[str, Any] = {}
    for line in reversed(text.splitlines()):
        if "step" in line and "loss=" in line and "tok/s=" in line and "step" not in out:
            m = re.search(r"\[epoch (\d+)\]\[step (\d+)\]", line)
            if not m:
                continue
            out.update(epoch=int(m.group(1)), step=int(m.group(2)), loss=_num(line, "loss"), lr=_num(line, "lr"),
                       grad_norm=_num(line, "grad_norm"), tok_s=_num(line, "tok/s"), mfu=_num(line, "MFU"),
                       step_s=_num(line, "step_s"))
        if "eval_loss=" in line and "eval_loss" not in out:
            out["eval_loss"] = _num(line, "eval_loss")
            out["eval_acc"] = _num(line, "eval_acc")
        if "step" in out and "eval_loss" in out:
            break
    return out


def config_path_from_log(head: str) -> Optional[str]:
    """The recipe a job runs, from the line the launcher prints at the start of the log."""
    m = re.search(r"'--config',\s*'([^']+)'", head) or re.search(r"--config\s+(\S+\.ya?ml)", head)
    return m.group(1) if m else None


def max_steps_of(recipe: Path) -> Optional[int]:
    """training.max_steps of a recipe file (read without a YAML library: the key is on a line of its own)."""
    try:
        text = recipe.read_text()
    except OSError:
        return None
    block = re.search(r"^training:\s*\n((?:[ \t]+.*\n|\n)+)", text, re.M)
    m = re.search(r"^[ \t]+max_steps:\s*(\d+)", block.group(1) if block else text, re.M)
    return int(m.group(1)) if m else None


def progress_and_eta(step: Optional[int], max_steps: Optional[int], step_s: Optional[float]) -> tuple[Optional[float], Optional[int]]:
    """(fraction done, seconds left) when they can be told."""
    if not step or not max_steps or max_steps <= 0:
        return None, None
    frac = min(1.0, step / max_steps)
    eta = int(max(0, max_steps - step) * step_s) if step_s else None
    return frac, eta


def parse_squeue(text: str) -> list[dict]:
    """From `squeue -h -o '%i|%j|%T|%P|%D|%N|%l|%M|%r|%k|%V'`."""
    jobs = []
    for line in text.splitlines():
        p = line.rstrip("\n").split("|")
        if len(p) < 11:
            continue
        jobs.append({"id": p[0], "name": p[1], "state": p[2], "partition": p[3], "node_count": int(p[4] or 0),
                     "node_list": p[5], "limit_s": parse_duration(p[6]), "elapsed_s": parse_duration(p[7]) or 0,
                     "reason": "" if p[8] in ("None", "") else p[8], "bhaskera": p[9].strip() == "bhaskera",
                     "kind": {"bhaskera": "train", "bhaskera-serve": "serve", "bhaskera-eval": "eval"}.get(p[9].strip(), "other"),
                     "submitted": p[10]})
    return jobs


def _last_value(path: Path) -> Optional[float]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 400))
            lines = f.read().decode(errors="replace").strip().splitlines()
        return float(lines[-1].split()[1]) if lines else None
    except (OSError, ValueError, IndexError):
        return None


def _min_value(path: Path) -> Optional[float]:
    try:
        values = [float(l.split()[1]) for l in path.read_text().splitlines() if l.strip()]
        return min(values) if values else None
    except (OSError, ValueError, IndexError):
        return None


def _meta(path: Path) -> dict:
    out = {}
    try:
        for line in path.read_text().splitlines():
            if ":" in line and not line.startswith(" "):
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


def _tag(run: Path, name: str) -> str:
    try:
        return (run / "tags" / name).read_text().strip()
    except OSError:
        return ""


STALE_AFTER_S = 20 * 60


def _lm_eval_deltas(metrics: Path) -> dict:
    """{task_metric: change} of the benchmark comparison stored by bhaskera.launcher.evaluate (files lmeval_*_delta)."""
    out = {}
    try:
        for entry in os.scandir(metrics):
            if entry.name.startswith("lmeval_") and entry.name.endswith("_delta"):
                value = _last_value(Path(entry.path))
                if value is not None:
                    out[entry.name[len("lmeval_"):-len("_delta")]] = value
    except OSError:
        pass
    return out


def _lm_eval_scores(metrics: Path) -> dict:
    """{task_metric: {base, tuned}} scores stored by bhaskera.launcher.evaluate (files lmeval_*_base and lmeval_*_tuned)."""
    out: dict = {}
    try:
        for entry in os.scandir(metrics):
            if entry.name.startswith("lmeval_"):
                for side in ("base", "tuned"):
                    if entry.name.endswith("_" + side):
                        value = _last_value(Path(entry.path))
                        if value is not None:
                            out.setdefault(entry.name[len("lmeval_"):-len(side) - 1], {})[side] = value
    except OSError:
        pass
    return out


def _newest_mtime(metrics: Path, fallback: Path) -> float:
    """When a run last wrote a number (older versions never closed their runs, so 'running' can be a long time ago)."""
    newest = 0.0
    try:
        for entry in os.scandir(metrics):
            newest = max(newest, entry.stat().st_mtime)
    except OSError:
        pass
    if not newest:
        try:
            newest = fallback.stat().st_mtime
        except OSError:
            pass
    return newest


def recent_runs(store: str | Path, limit: int = 40, now: Optional[float] = None) -> list[dict]:
    """The newest runs in an MLflow file store, read from the small files of each run (fast with hundreds of runs)."""
    store = Path(store)
    now = time.time() if now is None else now
    metas = []
    for meta in store.glob("*/*/meta.yaml"):
        try:
            metas.append((meta.stat().st_mtime, meta))
        except OSError:
            continue
    metas.sort(reverse=True)
    runs = []
    for _mtime, meta in metas:
        if len(runs) >= limit:
            break
        info = _meta(meta)
        if info.get("lifecycle_stage") == "deleted" or not info.get("run_name"):
            continue
        run = meta.parent
        try:
            started = int(info.get("start_time", "0")) / 1000.0
            ended = int(info["end_time"]) / 1000.0 if info.get("end_time", "null") not in ("null", "") else None
        except ValueError:
            started, ended = 0.0, None
        status = STATUS_NAMES.get(info.get("status", ""), "UNKNOWN")
        metrics = run / "metrics"
        stale = status == "RUNNING" and now - _newest_mtime(metrics, meta) > STALE_AFTER_S
        adapter = (run / "artifacts" / "adapter" / "adapter_config.json").exists()
        runs.append({
            "has_adapter": adapter, "lm_eval": _lm_eval_deltas(metrics), "lm_scores": _lm_eval_scores(metrics),
            "run_id": run.name, "experiment_id": run.parent.name, "name": info["run_name"], "status": status,
            "started": started, "duration_s": int((ended or now) - started) if started else None, "stale": stale,
            "job": _tag(run, "slurm_job"), "group": _tag(run, "group"),
            "loss": _last_value(metrics / "loss"), "eval_loss": _min_value(metrics / "eval_loss"),
            "eval_acc": _last_value(metrics / "eval_accuracy"), "mfu": _last_value(metrics / "throughput_mfu_pct"),
        })
    return runs


def fetch_serve_health(port: int, timeout: float = 3.0) -> Optional[dict]:
    """/health of a chat server (reached through the hub's ssh link), or None when it does not answer."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", "/health", headers={"Host": "localhost"})
        resp = conn.getresponse()
        return json.loads(resp.read()) if resp.status == 200 else None
    except (OSError, ValueError, http.client.HTTPException):
        return None


def fetch_ray_nodes(port: int, timeout: float = 3.0) -> list[dict]:
    """The nodes of a job with their GPU load, from the job's Ray Dashboard (reached through the hub's ssh link)."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", "/nodes?view=summary", headers={"Host": "localhost:8265"})
        resp = conn.getresponse()
        if resp.status != 200:
            return []
        rows = json.loads(resp.read())["data"]["summary"]
    except (OSError, ValueError, KeyError, http.client.HTTPException):
        return []
    nodes = []
    for n in rows:
        gpus = [{"util": g.get("utilizationGpu"), "mem_used": g.get("memoryUsed"), "mem_total": g.get("memoryTotal")}
                for g in n.get("gpus") or []]
        nodes.append({"host": n.get("hostname", "?"), "alive": (n.get("raylet") or {}).get("state", "ALIVE") == "ALIVE",
                      "cpu": n.get("cpu"), "gpus": gpus})
    return sorted(nodes, key=lambda n: n["host"])


def read_head_tail(path: Path) -> tuple[str, str]:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(LOG_HEAD_BYTES).decode(errors="replace")
            f.seek(max(0, size - LOG_TAIL_BYTES))
            tail = f.read().decode(errors="replace")
        return head, tail
    except OSError:
        return "", ""


def scontrol_job(job_id: str) -> dict:
    """User, log file and working folder of a job, from `scontrol show job`."""
    try:
        out = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    fields = dict(re.findall(r"(\w+)=(\S+)", out))
    return {"user": fields.get("UserId", "").split("(")[0], "stdout": fields.get("StdOut", ""),
            "workdir": fields.get("WorkDir", "")}


class StateBuilder:
    """Collects everything the page shows into one dict. Slow lookups are cached for ``ttl`` seconds."""

    def __init__(self, hub, store: str, user: str, *, list_jobs: Optional[Callable[[], list[dict]]] = None,
                 job_files: Callable[[str], dict] = scontrol_job, ray_nodes: Callable[[int], list[dict]] = fetch_ray_nodes,
                 serve_health: Callable[[int], Optional[dict]] = fetch_serve_health,
                 bad_nodes: Optional[Callable[[], dict]] = None, partitions: Optional[Callable[[list[str]], list[dict]]] = None,
                 version: str = "", docs_url: str = "", ttl: float = 8.0, clock: Callable[[], float] = time.time):
        self.hub, self.store, self.user = hub, store, user
        self._list_jobs, self._job_files, self._ray_nodes = list_jobs or squeue_jobs, job_files, ray_nodes
        self._serve_health = serve_health
        self._bad_nodes = bad_nodes or _load_bad_nodes
        self._partitions = partitions or partition_status
        self.version, self.docs_url, self.ttl, self._clock = version, docs_url, ttl, clock
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _cached(self, key: str, ttl: float, make: Callable[[], Any]) -> Any:
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < ttl:
                return hit[1]
        value = make()
        with self._lock:
            self._cache[key] = (now, value)
        return value

    def _job_detail(self, job: dict, link) -> dict:
        """Live numbers of one running job: training line from the log, progress, and the load of its nodes."""
        detail: dict[str, Any] = {}
        files = self._cached(f"files:{job['id']}", 60.0, lambda: self._job_files(job["id"]))
        if files.get("stdout"):
            head, tail = self._cached(f"log:{job['id']}", self.ttl, lambda: read_head_tail(Path(files["stdout"])))
            detail["train"] = parse_training(clean(tail))
            config = config_path_from_log(head)
            if config:
                recipe = Path(config) if os.path.isabs(config) else Path(files.get("workdir", ".")) / config
                max_steps = self._cached(f"steps:{job['id']}", 300.0, lambda: max_steps_of(recipe))
                t = detail["train"]
                t["max_steps"] = max_steps
                t["progress"], t["eta_s"] = progress_and_eta(t.get("step"), max_steps, t.get("step_s"))
        if link is not None:
            detail["nodes"] = self._cached(f"nodes:{job['id']}", self.ttl, lambda: self._ray_nodes(link.ray_port))
        return detail

    def build(self) -> dict:
        jobs = self._cached("squeue", 4.0, self._list_jobs)
        links = {l.job_id: l for l in self.hub.links()}
        runs = self._cached("runs", 15.0, lambda: recent_runs(self.store))
        active = {j["id"] for j in jobs}
        for r in runs:                                   # a run of a job that is still queued or running is not stalled
            if r["stale"] and r["job"] in active:
                r["stale"] = False
        by_job = {r["job"]: r for r in runs if r["job"]}
        out_jobs = []
        for job in jobs:
            item = dict(job)
            link = links.get(job["id"])
            item["linked"] = link is not None
            item["run"] = by_job.get(job["id"])
            if job["state"] in ("RUNNING", "COMPLETING") and job.get("kind") == "serve":
                health = self._cached(f"serve:{job['id']}", self.ttl, lambda: self._serve_health(link.serve_port)) if link and link.serve_port else None
                item["serve"] = {"ready": bool(health), **(health or {})}
            elif job["state"] in ("RUNNING", "COMPLETING"):
                item.update(self._job_detail(job, link))
            out_jobs.append(item)
        nodes = [n for j in out_jobs for n in j.get("nodes", [])]
        utils = [g["util"] for n in nodes for g in n["gpus"] if g.get("util") is not None]
        running = [j for j in out_jobs if j["state"] == "RUNNING"]
        bad = self._bad_nodes()
        from . import cluster_profile
        profile = cluster_profile.current()
        known = sorted({j["partition"] for j in jobs} | {r.name for r in profile.partitions})
        seen_gpus = [len(n["gpus"]) for n in nodes if n.get("gpus")]
        per_node = profile.gpus_per_node or (max(seen_gpus) if seen_gpus else None)
        return {
            "now": self._clock(), "user": self.user, "version": self.version,
            "docs_url": self.docs_url or profile.docs_url, "cluster": profile.name, "gpus_per_node": per_node,
            "jobs": out_jobs,
            "runs": runs,
            "bad_nodes": [{"host": h, "since": t, "reason": why} for h, (t, why) in sorted(bad.items())],
            "partitions": self._cached("partitions", 30.0, lambda: self._partitions(known)),
            "totals": {"running": len(running), "queued": sum(1 for j in out_jobs if j["state"] == "PENDING"),
                       "nodes": sum(j["node_count"] for j in running),
                       "gpus": per_node * sum(j["node_count"] for j in running) if per_node else None,
                       "gpu_util": round(sum(utils) / len(utils)) if utils else None},
        }

    def log(self, job_id: str, lines: int = 300) -> dict:
        """The end of the log of one of this user's jobs (the log of a finished job is not available here)."""
        if not re.fullmatch(r"\d+", job_id or ""):
            return {"error": "bad job id"}
        files = self._job_files(job_id)
        if not files.get("stdout") or files.get("user") != self.user:
            return {"error": "no log for this job"}
        _head, tail = read_head_tail(Path(files["stdout"]))
        rows = clean(tail).splitlines()
        rows = [r for r in rows if r.strip()][-max(10, min(lines, 1000)):]
        return {"job": job_id, "lines": rows}


def squeue_jobs() -> list[dict]:
    fmt = "%i|%j|%T|%P|%D|%N|%l|%M|%r|%k|%V"
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "R,PD,CG", "-o", fmt],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_squeue(out)


def _load_bad_nodes() -> dict:
    from . import bad_nodes
    return bad_nodes.load()


def partition_status(partitions: list[str]) -> list[dict]:
    """Per-partition limits and this user's use of them (see qos.py)."""
    from . import qos
    out = []
    for name in partitions:
        try:
            limits = qos.limits_for(name)
            running, submitted = qos.user_jobs(name)
        except Exception:
            continue
        out.append({"partition": name, "running": running, "submitted": submitted, "max_running": limits.max_running,
                    "max_submitted": limits.max_submitted, "max_hours": limits.max_hours})
    return out
