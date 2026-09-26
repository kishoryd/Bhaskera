"""
bhaskera.launcher.preflight
===========================
bhaskera-preflight --config <yaml> [--num-gpus N] [--gpu-mem-gb 80]

Login-node (CPU-only, seconds) checks that catch failures which otherwise
surface minutes to hours into a multi-node job:
  * the model directory and tokenizer load (a missing `sentencepiece` only
    appeared once Mixtral's tokenizer was used, after several failed jobs)
  * flash-attn / Liger import against the installed torch (ABI mismatches)
  * tokenized caches exist and match data.seq_len / model (see
    data/tokenize.py:_validate_tokenized_cache)
  * config combinations known to fail on this cluster
  * a per-GPU memory estimate, and the smallest GPU count that fits

The memory model is an estimate for planning, not a guarantee: sharded
weights + the gathered FSDP units in flight + decoder-level activation
checkpoints + one layer's recompute + fixed overhead. Exit code 1 on FAIL.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys

_GIB = 1024 ** 3
_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


class _Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str) -> None:
        self.rows.append((status, check, detail))
        print(f"[{status:4s}] {check:26s} {detail}", flush=True)

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for s, _, _ in self.rows)


class _WarningCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _check_model(cfg, rep: _Report) -> dict | None:
    path = cfg.model.name
    cfg_file = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_file):
        rep.add("FAIL", "model directory", f"no config.json at {path}")
        return None
    with open(cfg_file) as f:
        model_cfg = json.load(f)
    shards = glob.glob(os.path.join(path, "*.safetensors"))
    rep.add("PASS" if shards else "FAIL", "model weights",
            f"{len(shards)} safetensors file(s) in {path}")
    return model_cfg


def _check_tokenizer(cfg, rep: _Report) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            cfg.model.name, trust_remote_code=bool(cfg.model.trust_remote_code)
        )
        rep.add("PASS", "tokenizer", f"{type(tok).__name__}, vocab {len(tok)}")
    except Exception as e:  # noqa: BLE001 -- report any load failure verbatim
        rep.add("FAIL", "tokenizer", f"{type(e).__name__}: {str(e)[:200]}")


def _check_kernels(cfg, rep: _Report) -> None:
    if cfg.model.attn_impl == "flash_attention_2":
        try:
            import flash_attn
            import flash_attn_2_cuda  # noqa: F401 -- fails on a torch ABI mismatch

            rep.add("PASS", "flash-attn", f"{flash_attn.__version__} imports against this torch")
        except ImportError as e:
            rep.add("FAIL", "flash-attn", f"attn_impl=flash_attention_2 but: {e}")
    else:
        rep.add("INFO", "attention", f"attn_impl={cfg.model.attn_impl}")
    if cfg.model.use_liger_kernel:
        try:
            import liger_kernel  # noqa: F401

            rep.add("PASS", "liger-kernel", "imports")
        except ImportError as e:
            rep.add("FAIL", "liger-kernel", f"use_liger_kernel=true but: {e}")


def _check_caches(cfg, rep: _Report) -> None:
    from bhaskera.data.tokenize import _validate_tokenized_cache

    for field in ("tokenized_path", "val_tokenized_path"):
        path = getattr(cfg.data, field, None)
        if not path:
            rep.add("INFO", field, "not set")
            continue
        parquet = glob.glob(os.path.join(path, "*.parquet"))
        if not parquet:
            rep.add("FAIL", field, f"no parquet files at {path}")
            continue
        capture = _WarningCapture()
        logger = logging.getLogger("bhaskera.data.tokenize")
        logger.addHandler(capture)
        try:
            _validate_tokenized_cache(path, cfg)
        except ValueError as e:
            rep.add("FAIL", field, str(e)[:260])
            continue
        finally:
            logger.removeHandler(capture)
        if capture.messages:
            rep.add("WARN", field, capture.messages[0][:260])
        else:
            rep.add("PASS", field, f"{len(parquet)} parquet file(s), seq_len/model match")


def _check_config(cfg, model_cfg: dict, rep: _Report) -> None:
    fsdp = cfg.training.distributed.fsdp
    strategy = cfg.training.distributed.strategy.lower()
    num_experts = int(model_cfg.get("num_experts") or model_cfg.get("num_local_experts") or 0)
    if strategy == "fsdp" and num_experts >= 64 and fsdp.shard_experts_individually:
        layers = int(model_cfg.get("num_hidden_layers", 0))
        rep.add("WARN", "shard_experts_individually",
                f"true with {num_experts} experts x {layers} layers = "
                f"{num_experts * layers} tiny FSDP units; this hung all MoE runs in NCCL "
                "collectives here (engineering report 2.6). Use false.")
    ac = fsdp.activation_checkpointing if strategy == "fsdp" else cfg.training.distributed.ddp.activation_checkpointing
    if not ac:
        rep.add("WARN", "activation_checkpointing", "disabled; activation memory grows with layers x seq_len")
    if num_experts and cfg.lora.enabled and getattr(cfg.lora, "include_experts", False):
        rep.add("INFO", "lora.include_experts", "true: LoRA on every expert multiplies adapter count")


def _estimate_memory(cfg, model_cfg: dict, args, rep: _Report) -> None:
    index = os.path.join(cfg.model.name, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            stored = int(json.load(f).get("metadata", {}).get("total_size", 0))
    else:
        stored = sum(os.path.getsize(p) for p in glob.glob(os.path.join(cfg.model.name, "*.safetensors")))
    if stored <= 0:
        rep.add("WARN", "memory estimate", "could not determine checkpoint size")
        return
    stored_elem = _DTYPE_BYTES.get(str(model_cfg.get("torch_dtype", "bfloat16")), 2)
    train_elem = _DTYPE_BYTES.get(str(cfg.training.distributed.fsdp.param_dtype), 2)
    weights = stored / stored_elem * train_elem

    hidden = int(model_cfg.get("hidden_size", 0))
    layers = max(1, int(model_cfg.get("num_hidden_layers", 1)))
    vocab = int(model_cfg.get("vocab_size", 0))
    seq = int(cfg.data.seq_len)
    batch = int(cfg.training.batch_size)
    top_k = int(model_cfg.get("num_experts_per_tok", 0) or 0)
    moe_inter = int(model_cfg.get("moe_intermediate_size", 0) or 0)
    inter = int(model_cfg.get("intermediate_size", 0) or 0)
    if top_k and moe_inter:
        mlp_width, dispatch = top_k * moe_inter, 2 * top_k * hidden
    elif top_k:  # Mixtral-style: intermediate_size is per expert
        mlp_width, dispatch = top_k * inter, 2 * top_k * hidden
    else:
        mlp_width, dispatch = inter, 0

    embed = vocab * hidden * train_elem * (1 if model_cfg.get("tie_word_embeddings") else 2)
    unit = max(0.0, (weights - embed) / layers)  # one decoder layer's weights
    act_ckpt = layers * batch * seq * hidden * train_elem
    act_layer = batch * seq * train_elem * (12 * hidden + 3 * mlp_width + dispatch)
    logits = 0 if cfg.model.use_liger_kernel else batch * seq * vocab * 4
    overhead = 3.5 * _GIB  # CUDA context, NCCL buffers, LoRA params + AdamW state
    strategy = cfg.training.distributed.strategy.lower()
    fsdp = cfg.training.distributed.fsdp

    def per_gpu(n: int) -> float:
        if strategy == "ddp":
            w = weights
        else:
            gathered = 2 * unit  # current layer + prefetched/next
            root = embed if not fsdp.root_reshard_after_forward else 0
            w = weights / n + gathered + root
        return w + act_ckpt + act_layer + logits + overhead

    budget = 0.92 * args.gpu_mem_gb * _GIB
    rep.add("INFO", "weights", f"{weights / 1e9:.1f} GB in {cfg.training.distributed.fsdp.param_dtype} "
            f"({layers} layers, ~{unit / 1e9:.2f} GB per layer unit)")
    counts = [args.num_gpus] if args.num_gpus else [8, 16, 32, 64, 128, 256]
    for n in counts:
        need = per_gpu(n)
        status = "PASS" if need <= budget else "FAIL" if args.num_gpus else "INFO"
        rep.add(status, f"est. per-GPU @ {n} GPUs", f"{need / _GIB:.1f} GiB of {args.gpu_mem_gb} GiB")
    if strategy == "ddp":
        fits = per_gpu(1) <= budget
        rep.add("PASS" if fits else "FAIL", "min GPUs (DDP)",
                "any count (full replica fits)" if fits else "full replica does not fit one GPU; use fsdp")
    else:
        fixed = per_gpu(1) - weights
        min_n = math.inf if budget <= fixed else math.ceil(weights / (budget - fixed))
        rep.add("INFO", "min GPUs (FSDP est.)", "does not fit at any scale" if min_n == math.inf
                else f"{min_n} (round up to a multiple of GPUs/node)")


def main() -> None:
    p = argparse.ArgumentParser(description="Preflight checks for a Bhaskera training config")
    p.add_argument("--config", required=True)
    p.add_argument("--num-gpus", type=int, default=0, help="planned GPU count (default: show a table)")
    p.add_argument("--gpu-mem-gb", type=float, default=80.0, help="per-GPU memory (A100 80GB)")
    args = p.parse_args()

    from bhaskera.config import load_config

    cfg = load_config(args.config)
    rep = _Report()
    print(f"Preflight: {args.config}", flush=True)
    model_cfg = _check_model(cfg, rep)
    _check_tokenizer(cfg, rep)
    _check_kernels(cfg, rep)
    _check_caches(cfg, rep)
    if model_cfg is not None:
        _check_config(cfg, model_cfg, rep)
        _estimate_memory(cfg, model_cfg, args, rep)
    print("RESULT:", "FAIL" if rep.failed else "OK", flush=True)
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
