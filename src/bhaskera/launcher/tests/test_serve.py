"""The chat server for a tuned model, with a tiny random model and a real PEFT adapter (CPU, no downloads)."""
from __future__ import annotations

import http.client
import json

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from peft import LoraConfig, get_peft_model  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from bhaskera.launcher import serve  # noqa: E402
from bhaskera.utils.adapter_export import export_adapter  # noqa: E402

WORDS = "system user assistant : hello world the a is cat dog says what who are you i am fine thanks yes no".split()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    fast.chat_template = "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}assistant :"
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=128, eos_token_id=1, pad_token_id=0)
    base = LlamaForCausalLM(cfg)
    base.save_pretrained(d / "base")
    fast.save_pretrained(d / "base")
    peft = get_peft_model(LlamaForCausalLM.from_pretrained(d / "base"), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"],
                                                                                  task_type="CAUSAL_LM"))
    for n, p in peft.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=3.0)               # a strong "fine-tune", so the two models clearly differ
    export_adapter(peft, d / "adapter", rank=0, base_model=str(d / "base"))
    return d


@pytest.fixture(scope="module")
def host(tiny):
    return serve.ModelHost(str(tiny / "base"), str(tiny / "adapter"), dtype="float32")


def _run(host, peer_ok=True):
    srv = serve.serve(host, {"model": "tiny", "adapter": "tiny-adapter"}, 0, cap=64, peer_check=lambda c, s: peer_ok)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _call(srv, method, path, body=None, host_header="localhost"):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=60)
    data = json.dumps(body) if isinstance(body, dict) else body
    c.request(method, path, body=data, headers={"Host": host_header, "Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, r.read().decode()


GREEDY = {"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 8, "temperature": 0}


def test_request_checks_and_limits():
    ok = serve.parse_request({"messages": [{"role": "user", "content": "hi"}], "temperature": 9, "max_tokens": 99999, "top_p": 0}, cap=64)
    assert ok["temperature"] == 2.0 and ok["max_tokens"] == 64 and ok["top_p"] == 0.05 and ok["use_adapter"] is True and ok["stream"] is False
    for bad in ({}, {"messages": []}, {"messages": [{"role": "boss", "content": "x"}]}, {"messages": [{"role": "user", "content": 5}]},
                {"messages": [{"role": "user", "content": "x" * 30000}]}, {"messages": [{"role": "user", "content": "x"}] * 60},
                {"messages": [{"role": "user", "content": "x"}], "temperature": "hot"}):
        with pytest.raises(serve.RequestError):
            serve.parse_request(bad)


def test_port_and_labels():
    assert serve.port_for_job(130577) == 8577 and serve.port_for_job("x") == 8000
    assert serve.adapter_label("/runs/abc123/artifacts/adapter") == "abc123"
    assert serve.adapter_label("/runs/a161c92f49d14d1099284d6575756ef9/artifacts/adapter") == "a161c92f" and serve.adapter_label("/x/my-lora") == "my-lora" and serve.adapter_label(None) is None


def test_health_and_models(host):
    srv = _run(host)
    try:
        status, body = _call(srv, "GET", "/health")
        assert status == 200 and json.loads(body) == {"status": "ready", "model": "tiny", "adapter": "tiny-adapter"}
        assert json.loads(_call(srv, "GET", "/v1/models")[1])["data"][0]["id"] == "tiny"
        assert _call(srv, "GET", "/nothing")[0] == 404
    finally:
        srv.shutdown()


def test_the_tuned_and_the_base_model_answer_differently_from_the_same_weights(host):
    srv = _run(host)
    try:
        s1, tuned = _call(srv, "POST", "/v1/chat/completions", {**GREEDY, "use_adapter": True})
        s2, base = _call(srv, "POST", "/v1/chat/completions", {**GREEDY, "use_adapter": False})
        s3, tuned_again = _call(srv, "POST", "/v1/chat/completions", {**GREEDY, "use_adapter": True})
        assert (s1, s2, s3) == (200, 200, 200)
        t, b, t2 = (json.loads(x) for x in (tuned, base, tuned_again))
        assert t["choices"][0]["message"]["content"] != b["choices"][0]["message"]["content"]
        assert t["choices"][0]["message"]["content"] == t2["choices"][0]["message"]["content"]      # the adapter is back on afterwards
        assert t["bhaskera"]["adapter"] is True and b["bhaskera"]["adapter"] is False
    finally:
        srv.shutdown()


def test_streaming_and_whole_answers_are_the_same_text(host):
    srv = _run(host)
    try:
        whole = json.loads(_call(srv, "POST", "/v1/chat/completions", GREEDY)[1])["choices"][0]["message"]["content"]
        status, raw = _call(srv, "POST", "/v1/chat/completions", {**GREEDY, "stream": True})
        events = [l[len("data: "):] for l in raw.splitlines() if l.startswith("data: ")]
        assert status == 200 and events[-1] == "[DONE]"
        assert "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1]) == whole
    finally:
        srv.shutdown()


class _Words:
    """A stand-in for the model that yields known pieces (the toy model may answer with nothing but special tokens)."""

    def __init__(self, pieces, fail_after=None):
        self.pieces, self.fail_after = pieces, fail_after

    def generate(self, req):
        for i, piece in enumerate(self.pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("out of memory")
            yield piece


def test_streaming_format_pieces_finish_and_errors():
    srv = serve.serve(_Words(["Hel", "lo", " world"]), {"model": "m"}, 0, peer_check=lambda c, s: True)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        status, raw = _call(srv, "POST", "/v1/chat/completions", {**GREEDY, "stream": True})
        events = [l[len("data: "):] for l in raw.splitlines() if l.startswith("data: ")]
        pieces = [json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1]]
        assert status == 200 and pieces == ["Hel", "lo", " world", ""] and events[-1] == "[DONE]"
        assert json.loads(events[-2])["choices"][0]["finish_reason"] == "stop"
        status, body = _call(srv, "POST", "/v1/chat/completions", GREEDY)
        assert json.loads(body)["choices"][0]["message"]["content"] == "Hello world"
    finally:
        srv.shutdown()
    failing = serve.serve(_Words(["a", "b"], fail_after=1), {"model": "m"}, 0, peer_check=lambda c, s: True)
    threading.Thread(target=failing.serve_forever, daemon=True).start()
    try:
        _, raw = _call(failing, "POST", "/v1/chat/completions", {**GREEDY, "stream": True})
        assert '"error": "out of memory"' in raw and "[DONE]" not in raw                     # the error reaches the page, not silence
        status, body = _call(failing, "POST", "/v1/chat/completions", GREEDY)
        assert status == 500 and json.loads(body)["error"] == "out of memory"
    finally:
        failing.shutdown()


def test_bad_requests_and_strangers_are_refused(host):
    srv = _run(host)
    try:
        assert _call(srv, "POST", "/v1/chat/completions", {"messages": []})[0] == 400
        assert _call(srv, "POST", "/v1/chat/completions", "{not json")[0] == 400
        assert _call(srv, "POST", "/other", GREEDY)[0] == 404
        assert _call(srv, "GET", "/health", host_header="evil.example")[0] == 421
    finally:
        srv.shutdown()
    refused = _run(host, peer_ok=False)
    try:
        assert _call(refused, "GET", "/health")[0] == 403 and _call(refused, "POST", "/v1/chat/completions", GREEDY)[0] == 403
    finally:
        refused.shutdown()
