"""The hub adds a link back to the dashboard to the MLflow and Ray Dashboard pages it relays, and nothing else."""
from __future__ import annotations

import http.client
import http.server
import json
import threading

import pytest

from bhaskera.launcher import hub_guard, ray_hub

HTML = b"<html><head><title>x</title></head><body><div id=root></div></body></html>"


class _Proc:
    def poll(self):
        return None


@pytest.fixture
def upstream():
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _answer(self, ctype, body, extra=()):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen.append((self.path, self.headers.get("Accept-Encoding")))
            if self.path == "/api/x":
                return self._answer("application/json", b'{"ok": true}')
            if self.path == "/secure":
                return self._answer("text/html", HTML, [("Content-Security-Policy", "default-src 'self'")])
            return self._answer("text/html; charset=utf-8", HTML)

        def do_POST(self):
            self._answer("text/html", HTML)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, seen
    srv.shutdown()


def _hub(up):
    jobs = [{"id": "5", "name": "t", "nodes": "n1", "kind": "train"}]
    ports = iter([up.server_port, up.server_port, up.server_port])
    hub = ray_hub.Hub(lambda: jobs, lambda n: n, lambda h, r, g: [h], spawn=lambda c: _Proc(), pick_port=lambda: next(ports),
                      peer_check=lambda c, s: True, mlflow_port=up.server_port)
    hub.refresh()
    return hub, ray_hub.serve(hub, ray_hub.free_port())


def _get(srv, host, path="/", method="GET"):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    c.request(method, path, headers={"Host": host})
    r = c.getresponse()
    return r.status, r.getheader("Content-Length"), r.read()


@pytest.mark.parametrize("host", ["mlflow.localhost:8265", "job-5.localhost:8265"])
def test_pages_get_a_link_back(upstream, host):
    up, seen = upstream
    _hub_, srv = _hub(up)
    code, length, body = _get(srv, host)
    srv.shutdown()
    assert code == 200 and b'id="bhaskera-back"' in body and b'href="http://localhost:8265/"' in body
    assert body.index(b"bhaskera-back") < body.index(b"</body>") and int(length) == len(body)
    assert seen[-1][1] == "identity"


def test_other_answers_are_untouched(upstream):
    up, _seen = upstream
    _hub_, srv = _hub(up)
    code, _l, body = _get(srv, "mlflow.localhost:8265", "/api/x")
    assert json.loads(body) == {"ok": True}
    assert b"bhaskera-back" not in _get(srv, "mlflow.localhost:8265", "/secure")[2], "a page with its own security policy is left alone"
    assert b"bhaskera-back" not in _get(srv, "grafana-5.localhost:8265")[2], "chart pages are shown inside frames and stay as they are"
    srv.shutdown()


def test_link_text_helpers():
    assert hub_guard.with_back_link(b"<p>hi</p>", "mlflow.localhost:9000").endswith(b"</a>")
    out = hub_guard.with_back_link(b"<body>x</body>", "job-1.localhost:9000")
    assert b'href="http://localhost:9000/"' in out and out.endswith(b"</body>")
    assert b'href="http://localhost/"' in hub_guard.with_back_link(b"<body></body>", "weird")
