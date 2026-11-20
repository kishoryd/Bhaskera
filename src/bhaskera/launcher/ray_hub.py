"""One address for the Ray Dashboards of all your running Bhaskera jobs.

Every Slurm job starts its own Ray cluster, and a Ray Dashboard shows only its own cluster. To see several
jobs without knowing their nodes, `bhaskera-dashboard ray` runs this hub on the login node:

* for each running job it keeps an ssh link from the login node to the first node of that job
  (the Ray Dashboard and Grafana listen there only);
* two small proxies, on the two ports the laptop forwards, pick the job from the address that was typed:
  ``http://job-<id>.localhost:8265`` is the Ray Dashboard of that job and ``http://job-<id>.localhost:3000``
  its Grafana charts. Browsers send every ``*.localhost`` name to the laptop itself, so no setup is needed;
* ``http://localhost:8265`` is a list of the running jobs with links.

Jobs that start later are picked up by themselves, jobs that end are dropped.
"""
from __future__ import annotations

import html
import http.client
import http.server
import re
import socket
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

RAY_PORT = 8265        # on the first node of a job
GRAFANA_PORT = 3000    # on the first node of a job

_HOST = re.compile(r"^job-(\d+)\.localhost(?::\d+)?$", re.IGNORECASE)
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
        "transfer-encoding", "upgrade", "host", "origin"}


def job_from_host(host: str) -> Optional[str]:
    """The job id in an address like ``job-128944.localhost:8265``, else None."""
    m = _HOST.match((host or "").strip())
    return m.group(1) if m else None


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


class Hub:
    """Keeps one ssh link per running job and answers which local port belongs to which job."""

    def __init__(self, list_jobs: Callable[[], list[dict]], head_of: Callable[[str], str],
                 forward_cmd: Callable[[str, int, int], list[str]], *, laptop_ray: int = RAY_PORT,
                 laptop_grafana: int = GRAFANA_PORT, laptop_mlflow: int = 5055,
                 spawn: Callable[..., object] = subprocess.Popen, pick_port: Callable[[], int] = free_port):
        self._list_jobs, self._head_of, self._forward_cmd = list_jobs, head_of, forward_cmd
        self.laptop_ray, self.laptop_grafana, self.laptop_mlflow = laptop_ray, laptop_grafana, laptop_mlflow
        self._spawn, self._pick_port = spawn, pick_port
        self._links: dict[str, JobLink] = {}
        self._lock = threading.Lock()

    # -- the job list ------------------------------------------------------------------------------
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

    # -- the page ----------------------------------------------------------------------------------
    def index_html(self) -> str:
        rows = []
        for l in self.links():
            rows.append(
                f"<tr><td>{html.escape(l.job_id)}</td><td>{html.escape(l.name)}</td>"
                f"<td>{html.escape(l.nodes)}</td>"
                f"<td><a href='http://job-{l.job_id}.localhost:{self.laptop_ray}/'>Ray Dashboard</a></td>"
                f"<td><a href='http://job-{l.job_id}.localhost:{self.laptop_grafana}/'>Grafana charts</a></td></tr>")
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
            f"<a href='http://localhost:{self.laptop_mlflow}/'>MLflow</a></p>")


def _handler(hub: Hub, kind: str):
    """Request handler for the Ray Dashboard proxy (kind 'ray') or the Grafana proxy (kind 'grafana')."""
    upstream_service = RAY_PORT if kind == "ray" else GRAFANA_PORT

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "bhaskera-hub"

        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: str, ctype: str = "text/html; charset=utf-8") -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _handle(self) -> None:
            job_id = job_from_host(self.headers.get("Host", ""))
            if job_id is None:
                if self.path.split("?")[0] in ("/", "/index.html"):
                    return self._send(200, hub.index_html())
                return self._send(404, "Open the list of jobs at http://localhost first.")
            link = hub.link(job_id)
            if link is None:
                return self._send(404, f"Job {html.escape(job_id)} is not running.")
            if "upgrade" in self.headers.get("Connection", "").lower():
                return self._send(501, "Live connections are not available through the hub.")
            port = link.ray_port if kind == "ray" else link.grafana_port
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
            headers["Host"] = f"localhost:{upstream_service}"
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
                conn.request(self.command, self.path, body, headers)
                resp = conn.getresponse()
            except OSError:
                return self._send(502, f"The dashboard of job {html.escape(job_id)} is not up yet; try again in a minute.")
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in _HOP:
                    self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                try:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (OSError, http.client.HTTPException):
                    pass
            conn.close()

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _handle

    return Handler


def serve(hub: Hub, ray_listen: int, grafana_listen: int) -> list[http.server.ThreadingHTTPServer]:
    """Start the two proxies on 127.0.0.1 (raises OSError when a port is taken). Each runs in its own thread."""
    servers = []
    try:
        for port, kind in ((ray_listen, "ray"), (grafana_listen, "grafana")):
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _handler(hub, kind))
            srv.daemon_threads = True
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            servers.append(srv)
    except OSError:
        for srv in servers:
            srv.shutdown()
        raise
    return servers
