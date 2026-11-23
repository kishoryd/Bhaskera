"""POST /api/launch on the hub: the same guards as the chat route, and the answers the page relies on."""
from __future__ import annotations

import http.client
import json
import subprocess

import pytest

from bhaskera.launcher import launch, ray_hub

RUN = "c" * 32
HEAD = {"Origin": "http://localhost:8265", "X-Requested-With": "bhaskera", "Content-Type": "application/json", "Host": "localhost:8265"}


class _Proc:
    def poll(self):
        return None


@pytest.fixture
def server(tmp_path):
    (tmp_path / "recipes").mkdir()
    (tmp_path / "submit.sh").write_text("")
    (tmp_path / "recipes" / "ok.yaml").write_text("model:\n  name: /m/X\n")
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "submitted job 7: x", "")

    hub = ray_hub.Hub(lambda: [], lambda n: n, lambda h, r, g: [h], spawn=lambda c: _Proc(), peer_check=lambda c, s: True)
    hub.launcher = launch.Launcher(tmp_path, runs=lambda: [{"run_id": RUN, "name": "r", "status": "FINISHED", "has_adapter": True}],
                                   env={"BHASKERA_DIR": "/x", "SLURM_ACCOUNT": "a"}, run=run)
    srv = ray_hub.serve(hub, ray_hub.free_port())
    yield srv, hub, calls
    srv.shutdown()


def _post(srv, body, headers=HEAD, path="/api/launch"):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    c.request("POST", path, data, {**headers, "Content-Length": str(len(data))})
    r = c.getresponse()
    raw = r.read() or b"{}"
    try:
        return r.status, json.loads(raw)
    except ValueError:                      # the host and peer guards answer in plain text
        return r.status, {"text": raw.decode()}


def test_options_lists_recipes_and_runs(server):
    srv, _hub, _ = server
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    c.request("GET", "/api/launch", headers={"Host": "localhost:8265"})
    r = c.getresponse()
    o = json.loads(r.read())
    assert r.status == 200 and o["enabled"] and o["recipes"][0]["file"] == "ok.yaml" and o["runs"][0]["run_id"] == RUN


def test_start_runs_the_script(server):
    srv, _hub, calls = server
    code, out = _post(srv, {"action": "start", "kind": "train", "recipe": "ok.yaml", "nodes": 1})
    assert code == 200 and out["ok"] and out["job"] == "7" and calls and calls[0][0] == "bash"


@pytest.mark.parametrize("headers", [
    {**HEAD, "Origin": "http://evil.example"},
    {k: v for k, v in HEAD.items() if k != "X-Requested-With"},
])
def test_foreign_requests_are_refused(server, headers):
    srv, _hub, calls = server
    code, out = _post(srv, {"action": "start", "kind": "train", "recipe": "ok.yaml", "nodes": 1}, headers)
    assert code == 403 and calls == []


def test_bad_inputs(server):
    srv, _hub, calls = server
    assert _post(srv, b"not json")[0] == 400
    assert _post(srv, b"[1,2]")[0] == 400
    assert _post(srv, b"x" * 300000)[0] == 413
    code, out = _post(srv, {"action": "start", "kind": "train", "recipe": "ok.yaml", "nodes": "1; ls"})
    assert code == 400 and "whole number" in out["error"] and calls == []


def test_rebinding_host_is_refused(server):
    srv, _hub, calls = server
    code, _ = _post(srv, {"action": "start", "kind": "train", "recipe": "ok.yaml", "nodes": 1}, {**HEAD, "Host": "evil.example:8265", "Origin": "http://localhost:8265"})
    assert code == 421 and calls == []


def test_peer_of_another_user_is_refused(server):
    srv, hub, calls = server
    hub.peer_check = lambda c, s: False
    code, _ = _post(srv, {"action": "start", "kind": "train", "recipe": "ok.yaml", "nodes": 1})
    assert code == 403 and calls == []


def test_no_launcher_means_off(server):
    srv, hub, _ = server
    hub.launcher = None
    assert _post(srv, {"kind": "train"})[0] == 404


def test_resume_route(server):
    srv, hub, calls = server
    hub.launcher._runs = lambda: [{"run_id": "e" * 32, "name": "crashed", "status": "FAILED", "job": "55"}]
    hub.launcher._describe = lambda store, run: {"run_id": run, "name": "crashed", "status": "FAILED", "recipe": "r.yaml", "nodes": 2, "hours": 1,
                                                  "steps": [50], "latest": 50, "max_steps": "100", "resumable": True, "why": ""}
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    c.request("GET", "/api/resume?run=" + "e" * 32, headers={"Host": "localhost:8265"})
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read())["steps"] == [50]
    c.request("GET", "/api/resume?run=nope", headers={"Host": "localhost:8265"})
    assert c.getresponse().status == 400
    code, out = _post(srv, {"action": "start", "kind": "resume", "run": "e" * 32, "step": 50})
    assert code == 200 and out["ok"] and calls[-1][2:] == ["resume", "--run", "e" * 32, "--step", "50"]
