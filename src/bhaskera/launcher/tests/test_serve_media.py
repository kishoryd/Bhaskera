"""Chat with images, video and audio: what the server accepts, refuses, and tells the page. No model needed."""
from __future__ import annotations

import base64
import http.client
import json
import threading

import pytest

from bhaskera.launcher import serve

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 32
JPG = b"\xff\xd8\xff\xe0" + b"\0" * 32
WAV = b"RIFF\0\0\0\0WAVEfmt " + b"\0" * 32
MP4 = b"\0\0\0\x20ftypisom" + b"\0" * 32


def b64(raw):
    return base64.b64encode(raw).decode()


def msg(*parts, role="user"):
    return {"messages": [{"role": role, "content": list(parts)}]}


def test_text_requests_are_unchanged():
    req = serve.parse_request({"messages": [{"role": "user", "content": "hi"}]})
    assert req["messages"] == [{"role": "user", "content": "hi"}] and req["media"] == []


def test_media_parts_are_decoded_and_typed_by_their_bytes():
    req = serve.parse_request(msg({"type": "text", "text": "what is this?"}, {"type": "image", "data": "data:image/png;base64," + b64(PNG)},
                                  {"type": "audio", "data": b64(WAV)}, {"type": "video", "data": b64(MP4)}))
    parts = req["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert (parts[1]["type"], parts[1]["ext"], parts[1]["bytes"]) == ("image", ".png", PNG)
    assert (parts[2]["ext"], parts[3]["ext"]) == (".wav", ".mp4") and req["media"] == ["audio", "image", "video"]
    assert serve.parse_request(msg({"type": "image", "data": b64(JPG)}))["messages"][0]["content"][0]["ext"] == ".jpg"


@pytest.mark.parametrize("body,word", [
    (msg({"type": "image", "data": b64(b"not an image at all, just text")}), "does not look like"),
    (msg({"type": "audio", "data": b64(PNG)}), "does not look like"),
    (msg({"type": "image", "data": "!!!not base64!!!"}), "base64"),
    (msg({"type": "image"}), "base64"),
    (msg({"type": "image", "data": ""}), "base64"),
    (msg({"type": "document", "data": b64(PNG)}), "text, image, video or audio"),
    (msg({"type": "image", "data": b64(PNG)}, role="assistant"), "only the user"),
    ({"messages": [{"role": "user", "content": []}]}, "content"),
    ({"messages": [{"role": "user", "content": 5}]}, "content"),
])
def test_bad_media_requests_are_explained(body, word):
    with pytest.raises(serve.RequestError, match=word):
        serve.parse_request(body)


def test_limits(monkeypatch):
    monkeypatch.setattr(serve, "MAX_MEDIA_BYTES", {"image": 100, "audio": 100, "video": 100})
    with pytest.raises(serve.RequestError, match="larger than"):
        serve.parse_request(msg({"type": "image", "data": b64(PNG + b"\0" * 400)}))
    monkeypatch.setattr(serve, "MAX_MEDIA_TOTAL", 150)
    with pytest.raises(serve.RequestError, match="add up"):
        serve.parse_request(msg(*[{"type": "image", "data": b64(PNG + b"\0" * 40)}] * 3))
    monkeypatch.setattr(serve, "MAX_MEDIA_TOTAL", 10**9)
    monkeypatch.setattr(serve, "MAX_MEDIA_ITEMS", 2)
    with pytest.raises(serve.RequestError, match="at most 2 files"):
        serve.parse_request(msg(*[{"type": "image", "data": b64(PNG)}] * 3))


class _Host:
    def __init__(self):
        self.seen = []

    def generate(self, req):
        self.seen.append(req)
        n = sum(1 for m in req["messages"] if isinstance(m["content"], list) for p in m["content"] if p["type"] != "text")
        yield f"saw {n} files"


def _server(modalities):
    host = _Host()
    srv = serve.serve(host, {"model": "m", "modalities": modalities}, 0, peer_check=lambda c, s: True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, host


def _post(srv, body):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    data = json.dumps(body).encode()
    c.request("POST", "/v1/chat/completions", data, {"Content-Type": "application/json", "Host": "localhost", "Content-Length": str(len(data))})
    r = c.getresponse()
    return r.status, json.loads(r.read())


def test_a_model_that_reads_images_answers_and_says_so_in_health():
    srv, host = _server(["text", "image", "video"])
    try:
        c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
        c.request("GET", "/health", headers={"Host": "localhost"})
        assert json.loads(c.getresponse().read())["modalities"] == ["text", "image", "video"]
        code, out = _post(srv, msg({"type": "text", "text": "hi"}, {"type": "image", "data": b64(PNG)}))
        assert code == 200 and out["choices"][0]["message"]["content"] == "saw 1 files"
    finally:
        srv.shutdown()


def test_a_model_is_not_sent_files_it_cannot_read():
    srv, host = _server(["text"])
    try:
        code, out = _post(srv, msg({"type": "image", "data": b64(PNG)}))
        assert code == 400 and "cannot read image" in out["error"] and "text only" in out["error"] and host.seen == []
    finally:
        srv.shutdown()
    srv, host = _server(["text", "image", "video"])
    try:
        code, out = _post(srv, msg({"type": "audio", "data": b64(WAV)}))
        assert code == 400 and "cannot read audio" in out["error"] and "image, video" in out["error"]
    finally:
        srv.shutdown()
