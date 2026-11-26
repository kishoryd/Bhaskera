"""The dashboard behind a web portal: reached over the network, protected by a session token, working under any path prefix."""
from __future__ import annotations

import base64
import http.client
import json

import pytest

from bhaskera.launcher import hub_ui, launch, ray_hub

TOKEN = "t" * 24


class _Proc:
    def poll(self):
        return None


@pytest.fixture
def portal():
    hub = ray_hub.Hub(lambda: [], lambda n: n, lambda h, r, g: [h], spawn=lambda c: _Proc(), peer_check=lambda c, s: False)   # the same-user check would refuse everything
    hub.state_provider = lambda: {"jobs": [], "runs": []}
    hub.proxy_token = TOKEN
    hub.launcher = launch.Launcher(None, runs=lambda: [], env={})
    srv = ray_hub.serve(hub, ray_hub.free_port())
    yield srv
    srv.shutdown()


def call(srv, method="GET", path="/", headers=None, body=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    c.request(method, path, body, {"Host": "ood.example.org", **(headers or {})})
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read().decode()


def test_nothing_is_served_without_the_token(portal):
    for path in ("/", "/api/state", "/api/launch", "/?password=wrong"):
        assert call(portal, path=path)[0] == 403
    assert call(portal, "POST", "/", {"Content-Type": "application/x-www-form-urlencoded"}, "password=wrong")[0] == 403


def test_the_link_from_the_portal_sets_a_cookie_and_opens_the_page(portal):
    status, head, _ = call(portal, path=f"/?password={TOKEN}")
    assert status == 303 and head["Location"] == "./" and "bhaskera_session=" + TOKEN in head["Set-Cookie"] and "HttpOnly" in head["Set-Cookie"]
    status, _, page = call(portal, path="/", headers={"Cookie": f"bhaskera_session={TOKEN}"})
    assert status == 200 and "var PROXY=true;" in page
    status, _, body = call(portal, path="/api/state", headers={"Cookie": f"bhaskera_session={TOKEN}"})
    assert status == 200 and json.loads(body) == {"jobs": [], "runs": []}


def test_the_portal_can_also_send_the_token_as_a_form_or_basic_password(portal):
    status, head, _ = call(portal, "POST", "/", {"Content-Type": "application/x-www-form-urlencoded"}, f"password={TOKEN}")
    assert status == 303 and "Set-Cookie" in head
    basic = base64.b64encode(f"alice:{TOKEN}".encode()).decode()
    assert call(portal, path="/api/state", headers={"Authorization": f"Basic {basic}"})[0] == 200
    assert call(portal, path="/api/state", headers={"Authorization": "Basic " + base64.b64encode(b"alice:nope").decode()})[0] == 403


def test_changes_need_the_page_s_own_header_even_with_the_cookie(portal):
    cookie = {"Cookie": f"bhaskera_session={TOKEN}", "Origin": "https://ood.example.org", "Content-Type": "application/json"}
    assert call(portal, "POST", "/api/launch", cookie, "{}")[0] == 403
    assert call(portal, "POST", "/api/launch", {**cookie, "X-Requested-With": "bhaskera"}, "{}")[0] == 400      # past the checks: the launcher itself says it has no submit.sh


def test_the_page_works_under_any_prefix_and_without_localhost_links():
    page = hub_ui.INDEX_HTML
    assert 'fetch("/' not in page and 'api("/' not in page and '"/api/' not in page          # every request is relative to where the page is
    assert "var PROXY=false;" in page                                                      # the hub switches it on behind a portal


# ---- MLflow's own pages inside the portal tab

import http.server
import threading


@pytest.fixture
def portal_with_mlflow():
    seen = []

    class FakeMlflow(http.server.BaseHTTPRequestHandler):
        log_message = lambda *a: None

        def do_GET(self):
            seen.append((self.path, dict(self.headers)))
            body = b"<html>mlflow</html>"
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        do_POST = do_GET

    up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeMlflow)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    hub = ray_hub.Hub(lambda: [], lambda n: n, lambda h, r, g: [h], spawn=lambda c: _Proc(), peer_check=lambda c, s: False)
    hub.state_provider = lambda: {"jobs": [], "runs": []}
    hub.proxy_token = TOKEN
    hub.portal_prefix = "/rnode/node17/20001"
    hub.mlflow_port = up.server_address[1]
    srv = ray_hub.serve(hub, ray_hub.free_port())
    yield srv, seen
    srv.shutdown(); up.shutdown()


def test_mlflow_pages_get_the_prefix_back_and_never_see_the_session_token(portal_with_mlflow):
    srv, seen = portal_with_mlflow
    cookie = {"Cookie": f"bhaskera_session={TOKEN}", "Authorization": "Basic eDp5"}
    status, _, body = call(srv, path="/mlflow/static-files/app.js?x=1", headers=cookie)
    assert status == 200 and "mlflow" in body
    path, headers = seen[-1]
    assert path == "/rnode/node17/20001/mlflow/static-files/app.js?x=1"          # the portal stripped the prefix; MLflow was started to serve under it
    assert "Cookie" not in headers and "Authorization" not in headers and headers["Host"].startswith("localhost:")
    assert call(srv, path="/mlflow", headers=cookie)[:2][0] == 302 and call(srv, path="/mlflow", headers=cookie)[1]["Location"] == "mlflow/"
    status, _, _ = call(srv, "POST", "/mlflow/ajax-api/2.0/mlflow/runs/search", {**cookie, "Content-Type": "application/json"}, "{}")
    assert status == 200 and seen[-1][0] == "/rnode/node17/20001/mlflow/ajax-api/2.0/mlflow/runs/search"


def test_mlflow_pages_need_the_token_and_a_portal_that_keeps_the_prefix_also_works(portal_with_mlflow):
    srv, seen = portal_with_mlflow
    n = len(seen)
    assert call(srv, path="/mlflow/")[0] == 403 and len(seen) == n                  # nothing reaches MLflow without the token
    status, _, _ = call(srv, path="/rnode/node17/20001/mlflow/x", headers={"Cookie": f"bhaskera_session={TOKEN}"})   # a portal that does not strip
    assert status == 200 and seen[-1][0] == "/rnode/node17/20001/mlflow/x"
    status, _, page = call(srv, path="/rnode/node17/20001/", headers={"Cookie": f"bhaskera_session={TOKEN}"})
    assert status == 200 and "var PROXY=true;" in page


# ---- logging out

def test_logout_on_the_stand_alone_page_stops_it_and_needs_the_pages_own_header():
    hub = ray_hub.Hub(lambda: [], lambda n: n, lambda h, r, g: [h], spawn=lambda c: _Proc(), peer_check=lambda c, s: True)
    srv = ray_hub.serve(hub, ray_hub.free_port())
    try:
        def post(headers):
            c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
            c.request("POST", "/api/logout", b"{}", {"Host": "localhost:8265", **headers})
            r = c.getresponse()
            return r.status, json.loads(r.read() or b"{}")
        assert post({})[0] == 403 and not hub.stopping.is_set()                                                    # a request from anywhere else
        assert post({"X-Requested-With": "bhaskera", "Origin": "https://evil.example.org"})[0] == 403 and not hub.stopping.is_set()
        assert post({"X-Requested-With": "bhaskera"}) == (200, {"ok": True, "mode": "local"}) and hub.stopping.is_set()
    finally:
        srv.shutdown()


def test_logout_behind_a_portal_clears_the_pages_cookie_and_leaves_the_session_alone(portal):
    cookie = {"Cookie": f"bhaskera_session={TOKEN}"}
    assert call(portal, "POST", "/api/logout", cookie, "{}")[0] == 403
    status, head, body = call(portal, "POST", "/api/logout", {**cookie, "X-Requested-With": "bhaskera"}, "{}")
    assert status == 200 and json.loads(body)["mode"] == "portal" and "bhaskera_session=;" in head["Set-Cookie"] and "Max-Age=0" in head["Set-Cookie"]
