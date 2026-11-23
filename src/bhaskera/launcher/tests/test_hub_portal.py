"""The home page behind a web portal: reached over the network, protected by a session token, working under any path prefix."""
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
