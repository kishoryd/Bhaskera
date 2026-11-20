"""The hub that serves the Ray Dashboards of several jobs from one address. Runs on local ports only."""
from __future__ import annotations

import http.client
import http.server
import threading

import pytest

from bhaskera.launcher import ray_hub


class _Proc:
    def __init__(self):
        self.alive, self.terminated = True, False

    def poll(self):
        return None if self.alive else 1

    def terminate(self):
        self.terminated, self.alive = True, False

    def wait(self, timeout=None):
        return 0


def _upstream(label: str):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = f"{label} host={self.headers.get('Host')} path={self.path}".encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _get(port, path="/", host="localhost"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", path, headers={"Host": host})
    r = c.getresponse()
    return r.status, r.read().decode()


def test_job_from_host():
    assert ray_hub.job_from_host("job-128944.localhost:8265") == "128944"
    assert ray_hub.job_from_host("JOB-7.localhost") == "7"
    assert ray_hub.job_from_host("localhost:8265") is None
    assert ray_hub.job_from_host("job-7.example.com") is None


def test_refresh_adds_and_drops_links_as_jobs_come_and_go():
    jobs = [{"id": "11", "name": "a", "nodes": "n[1-2]"}]
    procs, cmds = [], []
    ports = iter(range(50000, 50100))
    hub = ray_hub.Hub(lambda: jobs, lambda nodes: "head-" + nodes, lambda h, r, g: cmds.append((h, r, g)) or [h],
                      spawn=lambda cmd: procs.append(_Proc()) or procs[-1], pick_port=lambda: next(ports))
    added, removed = hub.refresh()
    assert [l.job_id for l in added] == ["11"] and removed == []
    assert cmds == [("head-n[1-2]", 50000, 50001)]
    jobs.append({"id": "12", "name": "b", "nodes": "m[1-2]"})
    added, _ = hub.refresh()
    assert [l.job_id for l in added] == ["12"] and len(procs) == 2          # job 11 keeps its link
    jobs.pop(0)
    _, removed = hub.refresh()
    assert removed == ["11"] and procs[0].terminated and [l.job_id for l in hub.links()] == ["12"]
    procs[1].alive = False                                                   # the ssh link died: set it up again
    hub.refresh()
    assert len(procs) == 3
    hub.close()
    assert procs[2].terminated


def test_proxy_routes_by_job_name_in_the_address():
    up_a, up_b = _upstream("job-A"), _upstream("job-B")
    ports = {"1": (up_a.server_port, up_a.server_port), "2": (up_b.server_port, up_b.server_port)}
    jobs = [{"id": "1", "name": "first", "nodes": "x"}, {"id": "2", "name": "second", "nodes": "y"}]
    it = iter(sum(([ports[j["id"]][0], ports[j["id"]][1]] for j in jobs), []))
    hub = ray_hub.Hub(lambda: jobs, lambda n: n, lambda h, r, g: [h], spawn=lambda cmd: _Proc(),
                      pick_port=lambda: next(it))
    hub.refresh()
    servers = ray_hub.serve(hub, ray_hub.free_port(), ray_hub.free_port())
    try:
        ray_port = servers[0].server_port
        status, body = _get(ray_port, "/api/x?y=1", host="job-1.localhost:8265")
        assert status == 200 and body.startswith("job-A") and "path=/api/x?y=1" in body
        assert "host=localhost:8265" in body                                 # the dashboard sees its own address
        assert _get(ray_port, "/", host="job-2.localhost:8265")[1].startswith("job-B")
        status, page = _get(ray_port, "/", host="localhost:8265")
        assert status == 200 and "job-1.localhost:8265" in page and "job-2.localhost:8265" in page
        assert _get(ray_port, "/", host="job-99.localhost:8265")[0] == 404
        assert _get(servers[1].server_port, "/", host="job-1.localhost:3000")[1].startswith("job-A")
    finally:
        for s in servers + [up_a, up_b]:
            s.shutdown()


def test_proxy_reports_a_dashboard_that_is_not_up_yet():
    jobs = [{"id": "1", "name": "a", "nodes": "x"}]
    hub = ray_hub.Hub(lambda: jobs, lambda n: n, lambda h, r, g: [h], spawn=lambda cmd: _Proc(),
                      pick_port=ray_hub.free_port)
    hub.refresh()
    servers = ray_hub.serve(hub, ray_hub.free_port(), ray_hub.free_port())
    try:
        assert _get(servers[0].server_port, "/", host="job-1.localhost:8265")[0] == 502
    finally:
        for s in servers:
            s.shutdown()


def test_serve_fails_cleanly_when_a_port_is_taken():
    hub = ray_hub.Hub(lambda: [], lambda n: n, lambda h, r, g: [h])
    busy = ray_hub.serve(hub, ray_hub.free_port(), ray_hub.free_port())
    try:
        with pytest.raises(OSError):
            ray_hub.serve(hub, busy[0].server_port, ray_hub.free_port())
    finally:
        for s in busy:
            s.shutdown()
