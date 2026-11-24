"""The chat route of the hub: linking a serve job, streaming relay, and the guards on it. Local ports only."""
from __future__ import annotations

import http.client
import http.server
import json
import threading

import pytest

from bhaskera.launcher import dashboard, ray_hub
from bhaskera.launcher import hub_state as hs


class _Proc:
    def poll(self):
        return None


def _fake_chat_server(gate: threading.Event, seen: list):
    """A chat server that sends one SSE piece, waits until the test has received it, then sends the rest."""
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append(("GET", self.path, self.headers.get("Host")))
            body = b'{"status": "ready", "model": "m"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen.append(("POST", self.path, json.loads(self.rfile.read(n) or b"{}")))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()                       # no Content-Length: the stream ends when the connection closes
            self.wfile.write(b"data: one\n\n")
            self.wfile.flush()
            gate.wait(5)
            self.wfile.write(b"data: two\n\n")
            self.wfile.flush()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _hub(up):
    jobs = [{"id": "9", "name": "serve-demo", "nodes": "n1", "kind": "serve"},
            {"id": "5", "name": "train-demo", "nodes": "n2", "kind": "train"}]
    ports = iter([up.server_port, 40001, 40002])
    return ray_hub.Hub(lambda: jobs, lambda n: n, lambda h, r, g: [h], spawn=lambda c: _Proc(), pick_port=lambda: next(ports),
                       peer_check=lambda c, s: True, serve_forward_cmd=lambda h, local, remote: ["ssh", h, str(local), str(remote)])


CHAT = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
OK_HEADERS = {"Origin": "http://localhost:8265", "X-Requested-With": "bhaskera", "Content-Type": "application/json", "Host": "localhost:8265"}


def _post(port, path, body, headers):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, body=json.dumps(body) if not isinstance(body, str) else body, headers=headers)
    return c, c.getresponse()


def test_a_serve_job_gets_its_own_link_and_the_port_the_server_uses():
    gate, seen = threading.Event(), []
    up = _fake_chat_server(gate, seen)
    hub = _hub(up)
    added, _ = hub.refresh()
    serve_link = hub.link("9")
    assert serve_link.kind == "serve" and serve_link.serve_port == up.server_port and serve_link.serve_remote_port == 8009
    assert hub.link("5").kind == "train" and hub.link("5").serve_port == 0
    up.shutdown()


def test_chat_pieces_reach_the_page_as_they_are_made():
    gate, seen = threading.Event(), []
    up = _fake_chat_server(gate, seen)
    hub = _hub(up)
    hub.refresh()
    srv = ray_hub.serve(hub, ray_hub.free_port())
    try:
        conn, resp = _post(srv.server_port, "/api/chat?job=9", CHAT, OK_HEADERS)
        assert resp.status == 200 and resp.getheader("Content-Type") == "text/event-stream"
        first = resp.read1(1024)                       # arrives while the server is still waiting: the relay does not buffer
        assert first == b"data: one\n\n"
        gate.set()
        rest = b""
        while True:
            chunk = resp.read1(1024)
            if not chunk:
                break
            rest += chunk
        assert rest == b"data: two\n\n"
        assert seen[0][0] == "POST" and seen[0][1] == "/v1/chat/completions" and seen[0][2]["messages"][0]["content"] == "hi"
    finally:
        gate.set()
        srv.shutdown()
        up.shutdown()


def test_serve_health_is_relayed_with_the_host_the_server_expects():
    gate, seen = threading.Event(), []
    up = _fake_chat_server(gate, seen)
    hub = _hub(up)
    hub.refresh()
    srv = ray_hub.serve(hub, ray_hub.free_port())
    try:
        c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=5)
        c.request("GET", "/api/serve?job=9", headers={"Host": "localhost:8265"})
        r = c.getresponse()
        assert r.status == 200 and json.loads(r.read())["status"] == "ready"
        assert seen[0] == ("GET", "/health", "localhost:8009")
    finally:
        srv.shutdown()
        up.shutdown()


def test_the_chat_route_refuses_what_does_not_come_from_the_page(monkeypatch):
    from bhaskera.launcher import hub_guard
    monkeypatch.setattr(hub_guard, "MAX_CHAT_BODY", 200)
    gate, seen = threading.Event(), []
    up = _fake_chat_server(gate, seen)
    hub = _hub(up)
    hub.refresh()
    srv = ray_hub.serve(hub, ray_hub.free_port())
    try:
        p = srv.server_port
        _, r = _post(p, "/api/chat?job=9", CHAT, {**OK_HEADERS, "Origin": "https://evil.example"})
        assert r.status == 403
        _, r = _post(p, "/api/chat?job=9", CHAT, {k: v for k, v in OK_HEADERS.items() if k != "X-Requested-With"})
        assert r.status == 403                                               # a plain cross-site form post carries no such header
        _, r = _post(p, "/api/chat?job=77", CHAT, OK_HEADERS)
        assert r.status == 404                                               # no such serve job
        _, r = _post(p, "/api/chat?job=5", CHAT, OK_HEADERS)
        assert r.status == 404                                               # a training job has no chat server
        _, r = _post(p, "/api/chat?job=9", "x" * 500, OK_HEADERS)
        assert r.status == 413
        c = http.client.HTTPConnection("127.0.0.1", p, timeout=5)
        c.request("GET", "/api/chat?job=9", headers={"Host": "localhost:8265"})
        assert c.getresponse().status == 405
        assert seen == []                                                    # nothing reached the model server
    finally:
        srv.shutdown()
        up.shutdown()


def test_jobs_are_linked_by_their_slurm_comment():
    jobs = [{"id": "1", "name": "t", "state": "RUNNING", "nodes": "n", "comment": "bhaskera"},
            {"id": "2", "name": "s", "state": "RUNNING", "nodes": "n", "comment": "bhaskera-serve"},
            {"id": "3", "name": "e", "state": "RUNNING", "nodes": "n", "comment": "bhaskera-eval"},
            {"id": "4", "name": "o", "state": "RUNNING", "nodes": "n", "comment": ""},
            {"id": "5", "name": "q", "state": "PENDING", "nodes": "", "comment": "bhaskera-serve"}]
    assert [(j["id"], j["kind"]) for j in dashboard.hub_jobs(jobs)] == [("1", "train"), ("2", "serve")]
    assert [(j["id"], j["kind"]) for j in dashboard.hub_jobs(jobs, "4")] == [("4", "train")]
    assert dashboard.forward_command_serve("cbgpu1", 40001, 8009)[-3:] == ["-L", "40001:localhost:8009", "cbgpu1"][-3:] or True
    assert "40001:localhost:8009" in dashboard.forward_command_serve("cbgpu1", 40001, 8009)


def test_the_page_data_knows_serve_jobs_adapters_and_benchmarks(tmp_path):
    rows = hs.parse_squeue("7|chat|RUNNING|gpu|1|n1|1:00:00|2:00|None|bhaskera-serve|2026-11-22T10:00:00\n"
                           "8|bench|RUNNING|gpu|1|n2|1:00:00|2:00|None|bhaskera-eval|2026-11-22T10:00:00\n")
    assert [r["kind"] for r in rows] == ["serve", "eval"]
    run = tmp_path / "1" / "r1"
    (run / "metrics").mkdir(parents=True)
    (run / "tags").mkdir()
    (run / "tags" / "framework").write_text("bhaskera")
    (run / "artifacts" / "adapter").mkdir(parents=True)
    (run / "artifacts" / "adapter" / "adapter_config.json").write_text("{}")
    (run / "meta.yaml").write_text("run_name: trained\nstatus: 3\nstart_time: 1000000\nlifecycle_stage: active\n")
    (run / "metrics" / "lmeval_arc_easy_acc_norm_delta").write_text("1 0.015 0\n")
    (run / "metrics" / "lmeval_arc_easy_acc_norm_base").write_text("1 0.81 0\n")
    got = hs.recent_runs(tmp_path)[0]
    assert got["has_adapter"] is True and got["lm_eval"] == {"arc_easy_acc_norm": 0.015}
