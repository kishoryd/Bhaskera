"""The hub's guards: same user only, known addresses only, read-only Ray, no cross-site requests. Local ports only."""
from __future__ import annotations

import http.client
import http.server
import threading

import pytest

from bhaskera.launcher import hub_guard as g
from bhaskera.launcher import ray_hub


def test_only_the_hubs_own_addresses_are_accepted():
    for ok in ("localhost", "localhost:8265", "127.0.0.1:8265", "job-12.localhost:8265", "grafana-12.localhost", "mlflow.localhost:8265", "[::1]:80"):
        assert g.host_allowed(ok), ok
    for bad in ("", "evil.com", "evil.com:8265", "localhost.evil.com", "job-12.localhost.evil.com", "job-x.localhost", "10.0.0.5:8265",
                "evil.com#localhost", "localhost@evil.com"):
        assert not g.host_allowed(bad), bad


def test_origin_must_name_a_hub_address():
    assert g.origin_allowed("http://job-1.localhost:8265") and g.origin_allowed("http://localhost:8265")
    assert not g.origin_allowed("https://evil.example") and not g.origin_allowed("null") and not g.origin_allowed("http://localhost.evil.com")


def test_ray_is_read_only_the_others_may_post():
    assert g.method_allowed("job", "GET") and g.method_allowed("job", "HEAD")
    assert not g.method_allowed("job", "POST") and not g.method_allowed("job", "DELETE") and not g.method_allowed("job", "PUT")
    assert g.method_allowed("mlflow", "POST") and g.method_allowed("grafana", "POST")


def test_only_web_links_are_shown():
    assert g.safe_link("https://example.org/x") == "https://example.org/x"
    assert g.safe_link("javascript:alert(1)") == "" and g.safe_link("data:text/html,x") == "" and g.safe_link("") == ""


PROC = """  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:4E21 0100007F:D431 01 00000000:00000000 00:00000000 00000000  1000        0 11111 1
   1: 0100007F:D431 0100007F:4E21 01 00000000:00000000 00:00000000 00000000  1000        0 22222 1
   2: 0100007F:D432 0100007F:4E21 01 00000000:00000000 00:00000000 00000000  2222        0 33333 1
"""


def test_owner_of_a_loopback_connection_is_read_from_proc():
    server = ("127.0.0.1", 0x4E21)
    assert g.uid_of_connection(PROC, ("127.0.0.1", 0xD431), server) == 1000
    assert g.uid_of_connection(PROC, ("127.0.0.1", 0xD432), server) == 2222
    assert g.uid_of_connection(PROC, ("127.0.0.1", 0xD999), server) is None


def test_same_user_peer_decisions(tmp_path):
    f = tmp_path / "tcp"
    f.write_text(PROC)
    server = ("127.0.0.1", 0x4E21)
    assert g.same_user_peer(("127.0.0.1", 0xD431), server, uid=1000, proc=str(f)) is True
    assert g.same_user_peer(("127.0.0.1", 0xD432), server, uid=1000, proc=str(f)) is False      # another user's process
    assert g.same_user_peer(("127.0.0.1", 0xD999), server, uid=1000, proc=str(f)) is False      # not listed: refused
    assert g.same_user_peer(("127.0.0.1", 0xD431), server, uid=1000, proc=str(tmp_path / "none")) is True   # no /proc: nothing to check


def _upstream():
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            body = b"upstream-" + self.command.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_DELETE = _reply

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class _Proc:
    def poll(self):
        return None


def _hub(up, peer_ok=True):
    ports = iter([up.server_port, up.server_port])
    hub = ray_hub.Hub(lambda: [{"id": "1", "name": "a", "nodes": "x"}], lambda n: n, lambda h, r, gp: [h], spawn=lambda c: _Proc(),
                      pick_port=lambda: next(ports), mlflow_port=up.server_port, peer_check=lambda c, s: peer_ok)
    hub.refresh()
    hub.state_provider = lambda: {"jobs": []}
    return hub


def _req(port, method, path, host, headers=None, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request(method, path, body=body, headers={"Host": host, **(headers or {})})
    r = c.getresponse()
    return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read().decode()


def test_a_connection_from_another_user_is_refused_everywhere():
    up = _upstream()
    srv = ray_hub.serve(_hub(up, peer_ok=False), ray_hub.free_port())
    try:
        for host, path in (("localhost:8265", "/"), ("localhost:8265", "/api/state"), ("job-1.localhost:8265", "/"), ("mlflow.localhost:8265", "/")):
            assert _req(srv.server_port, "GET", path, host)[0] == 403
    finally:
        srv.shutdown()
        up.shutdown()


def test_unknown_host_names_are_refused():
    up = _upstream()
    srv = ray_hub.serve(_hub(up), ray_hub.free_port())
    try:
        assert _req(srv.server_port, "GET", "/api/state", "evil.example:8265")[0] == 421
        assert _req(srv.server_port, "GET", "/", "attacker.com")[0] == 421
        assert _req(srv.server_port, "GET", "/api/state", "localhost:8265")[0] == 200
    finally:
        srv.shutdown()
        up.shutdown()


def test_the_ray_dashboard_cannot_be_changed_through_the_hub():
    up = _upstream()
    srv = ray_hub.serve(_hub(up), ray_hub.free_port())
    try:
        assert _req(srv.server_port, "GET", "/api/jobs/", "job-1.localhost:8265")[2] == "upstream-GET"
        for method in ("POST", "DELETE"):
            status, _, body = _req(srv.server_port, method, "/api/jobs/", "job-1.localhost:8265", body="{}")
            assert status == 405 and "read-only" in body
    finally:
        srv.shutdown()
        up.shutdown()


def test_posts_from_other_web_sites_are_refused_but_the_apps_own_work():
    up = _upstream()
    srv = ray_hub.serve(_hub(up), ray_hub.free_port())
    try:
        host = "mlflow.localhost:8265"
        assert _req(srv.server_port, "POST", "/search", host, {"Origin": "https://evil.example"}, "{}")[0] == 403
        assert _req(srv.server_port, "POST", "/search", host, {"Origin": "null"}, "{}")[0] == 403
        status, _, body = _req(srv.server_port, "POST", "/search", host, {"Origin": "http://mlflow.localhost:8265"}, "{}")
        assert status == 200 and body == "upstream-POST"
        assert _req(srv.server_port, "POST", "/search", host, None, "{}")[2] == "upstream-POST"        # no Origin: not a browser
    finally:
        srv.shutdown()
        up.shutdown()


def test_large_bodies_are_refused(monkeypatch):
    up = _upstream()
    monkeypatch.setattr(g, "MAX_BODY", 100)
    srv = ray_hub.serve(_hub(up), ray_hub.free_port())
    try:
        assert _req(srv.server_port, "POST", "/x", "mlflow.localhost:8265", None, "a" * 500)[0] == 413
    finally:
        srv.shutdown()
        up.shutdown()


def test_the_hubs_own_pages_carry_security_headers():
    up = _upstream()
    srv = ray_hub.serve(_hub(up), ray_hub.free_port())
    try:
        for path in ("/", "/api/state"):
            _, headers, _ = _req(srv.server_port, "GET", path, "localhost:8265")
            assert headers["x-content-type-options"] == "nosniff" and headers["x-frame-options"] == "DENY"
            assert "frame-ancestors 'none'" in headers["content-security-policy"] and headers["referrer-policy"] == "no-referrer"
    finally:
        srv.shutdown()
        up.shutdown()


def test_mlflow_ui_and_ray_dashboard_do_not_listen_on_all_interfaces_by_default():
    from bhaskera.config import Config
    from bhaskera.launcher.dashboard import DashboardConfig
    assert DashboardConfig().host == "127.0.0.1"
    assert Config.from_dict({}).monitoring.dashboard_host == "127.0.0.1"
