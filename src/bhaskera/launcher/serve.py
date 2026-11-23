"""Try a fine-tuned model: a small chat server for one model and its adapter, run on a GPU node.

    ./submit.sh serve --run <mlflow run id>                     # on the cluster: the base model and adapter come from the run
    ./submit.sh serve --model <base folder> --adapter <adapter folder>
    python -m bhaskera.launcher.serve --model <base> --adapter <adapter> --port 8000      # directly, on a node with a GPU

The home page (``bhaskera-dashboard ray``) links the running server and shows a chat window for it. The same question can
be sent to the tuned model, to the base model, or to both side by side: the base answer comes from the same weights with the
adapter switched off, so no second copy is loaded.

The server speaks a small part of the OpenAI chat API:

    GET  /health                  {"status": "ready", "model": ..., "adapter": ...}
    GET  /v1/models
    POST /v1/chat/completions     {"messages": [...], "max_tokens": 256, "temperature": 0.7, "top_p": 0.9,
                                   "stream": true, "use_adapter": true}

It listens on 127.0.0.1 only and serves only connections from the user who started it (see hub_guard.py); the home page
reaches it through an SSH link. One answer is generated at a time; others wait their turn.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from . import hub_guard

MAX_BODY = 1024 * 1024
MAX_MESSAGES = 50
MAX_CONTENT_CHARS = 20000
MAX_NEW_TOKENS_CAP = 1024


def adapter_label(path: Optional[str]) -> Optional[str]:
    """A short name for the adapter: its folder name, or the training run's folder when it is the stored 'adapter'."""
    if not path:
        return None
    path = path.rstrip("/")
    name = os.path.basename(path)
    if name != "adapter":
        return name
    run = os.path.basename(os.path.dirname(os.path.dirname(path))) or name
    return run[:8] if len(run) == 32 and all(c in "0123456789abcdef" for c in run) else run          # an MLflow run id: its first 8 characters


def port_for_job(job_id: str | int, base: int = 8000) -> int:
    """The port a serve job listens on, so the home page can find it from the job id alone."""
    try:
        return base + int(job_id) % 1000
    except (TypeError, ValueError):
        return base


class RequestError(Exception):
    """A bad request: the message is returned to the caller with status 400."""


def parse_request(body: dict, cap: int = MAX_NEW_TOKENS_CAP) -> dict:
    """Check a chat request and fill in defaults."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    if len(messages) > MAX_MESSAGES:
        raise RequestError(f"at most {MAX_MESSAGES} messages")
    clean = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
            raise RequestError("each message needs a role (system, user or assistant) and a text content")
        if len(m["content"]) > MAX_CONTENT_CHARS:
            raise RequestError(f"a message is longer than {MAX_CONTENT_CHARS} characters")
        clean.append({"role": m["role"], "content": m["content"]})

    def number(name, default, lo, hi):
        v = body.get(name, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise RequestError(f"{name} must be a number")
        return min(max(v, lo), hi)

    return {"messages": clean, "max_tokens": int(number("max_tokens", 256, 1, cap)),
            "temperature": float(number("temperature", 0.7, 0.0, 2.0)), "top_p": float(number("top_p", 0.9, 0.05, 1.0)),
            "stream": bool(body.get("stream", False)), "use_adapter": bool(body.get("use_adapter", True)),
            "thinking": bool(body.get("thinking", False))}


class ModelHost:
    """The base model with its adapter, loaded once, and the generation lock."""

    def __init__(self, model: str, adapter: Optional[str] = None, dtype: str = "bfloat16", four_bit: bool = False,
                 trust_remote_code: bool = False, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_path, self.adapter_path = model, adapter
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        kwargs: dict = {"torch_dtype": getattr(torch, dtype), "trust_remote_code": trust_remote_code}
        if torch.cuda.is_available():
            kwargs["device_map"] = device_map
        if four_bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, dtype),
                                                               bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.has_adapter = bool(adapter)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.lock = threading.Lock()
        self.device = next(self.model.parameters()).device

    def _prompt_ids(self, messages: list[dict], thinking: bool):
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        except Exception:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    def generate(self, req: dict) -> Iterator[str]:
        """Pieces of the answer as they are produced. One caller at a time."""
        import contextlib
        import torch
        from transformers import TextIteratorStreamer
        ids = self._prompt_ids(req["messages"], req["thinking"])
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        sampling = req["temperature"] > 0
        gen = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "max_new_tokens": req["max_tokens"], "streamer": streamer,
               "do_sample": sampling, "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id}
        if sampling:
            gen.update(temperature=req["temperature"], top_p=req["top_p"])
        use_adapter = req["use_adapter"] or not self.has_adapter
        errors: list[BaseException] = []

        def run():
            ctx = contextlib.nullcontext() if use_adapter else self.model.disable_adapter()
            try:
                with ctx, torch.inference_mode():
                    self.model.generate(**gen)
            except BaseException as e:           # reported to the caller; the streamer must still end
                errors.append(e)
                streamer.end()

        with self.lock:
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            for piece in streamer:
                yield piece
            worker.join()
        if errors:
            raise RuntimeError(str(errors[0])[:300])


def _chunk(content: str, finish: Optional[str] = None) -> bytes:
    delta = {"content": content} if content else {}
    return ("data: " + json.dumps({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n").encode()


def make_handler(host, info: dict, cap: int = MAX_NEW_TOKENS_CAP, peer_check=hub_guard.same_user_peer):
    """Request handler. ``host`` needs ``generate(req)`` yielding text pieces (see ModelHost); ``info`` is shown by /health."""

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "bhaskera-serve"

        def log_message(self, *args):
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _allowed(self) -> bool:
            if not peer_check(self.client_address, self.server.server_address):
                self._json(403, {"error": "private to the user who started this server"})
                return False
            if (self.headers.get("Host") or "").split(":")[0] not in ("localhost", "127.0.0.1"):
                self._json(421, {"error": "use localhost"})
                return False
            return True

        def do_GET(self):
            if not self._allowed():
                return
            if self.path == "/health":
                return self._json(200, {"status": "ready", **info})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": info.get("model", "model"), "object": "model"}]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._allowed():
                return
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    return self._json(413, {"error": "request too large"})
                req = parse_request(json.loads(self.rfile.read(length) or b"{}"), cap)
            except (ValueError, RequestError) as e:
                return self._json(400, {"error": str(e)})
            started, tokens = time.monotonic(), 0
            if req["stream"]:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    for piece in host.generate(req):
                        tokens += 1
                        self.wfile.write(_chunk(piece))
                        self.wfile.flush()
                    self.wfile.write(_chunk("", "stop") + b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    return                                            # the caller closed the page
                except Exception as e:
                    self.wfile.write(("data: " + json.dumps({"error": str(e)}) + "\n\n").encode())
                return
            try:
                text = "".join(host.generate(req))
            except Exception as e:
                return self._json(500, {"error": str(e)})
            self._json(200, {"object": "chat.completion", "model": info.get("model"),
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                             "bhaskera": {"adapter": req["use_adapter"] and bool(info.get("adapter")),
                                          "seconds": round(time.monotonic() - started, 2)}})

    return Handler


def serve(host, info: dict, port: int, cap: int = MAX_NEW_TOKENS_CAP, peer_check=hub_guard.same_user_peer) -> http.server.ThreadingHTTPServer:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(host, info, cap, peer_check))
    srv.daemon_threads = True
    return srv


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="bhaskera-serve", description="Chat with a fine-tuned model.")
    p.add_argument("--run", help="MLflow run id of the training run (finds the base model and the adapter)")
    p.add_argument("--model", help="base model folder")
    p.add_argument("--adapter", help="adapter folder")
    p.add_argument("--store", default=os.environ.get("MLFLOW_TRACKING_URI", "").replace("file://", "") or str(Path.home() / "mlflow-runs"))
    p.add_argument("--port", type=int, help="default: 8000 + (Slurm job id mod 1000)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--4bit", action="store_true", dest="four_bit")
    p.add_argument("--max-new-tokens-cap", type=int, default=MAX_NEW_TOKENS_CAP, dest="cap")
    p.add_argument("--trust-remote-code", action="store_true", dest="trust")
    args = p.parse_args(argv)
    model, adapter = args.model, args.adapter
    if args.run:
        from .evaluate import EvalError, from_run
        try:
            m, a = from_run(args.store, args.run)
        except EvalError as e:
            print(f"error: {e}")
            return 1
        model, adapter = model or m, adapter or a
    if not model:
        print("error: give --model <base model folder>, or --run <mlflow run id>")
        return 1
    port = args.port or port_for_job(os.environ.get("SLURM_JOB_ID", 0))
    print(f"loading {model}" + (f" with adapter {adapter}" if adapter else "") + " ...", flush=True)
    host = ModelHost(model, adapter, args.dtype, args.four_bit, args.trust)
    info = {"model": os.path.basename(model.rstrip("/")), "adapter": adapter_label(adapter),
            "host": socket.gethostname()}
    srv = serve(host, info, port, args.cap)
    print(f"=== serving on {socket.gethostname()}:{port} ===", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
