"""
bhaskera.evaluation.runner
==========================
Run a list of tasks on a (possibly FSDP-sharded) model and write results.

Called on every rank with identical arguments: task construction is
deterministic, scoring is collective, and every rank returns the same
results dict (so in-training evaluation can log from rank 0 and everyone
stays in step).
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
import time

import torch
import torch.distributed as dist

from bhaskera.evaluation import tasks as T
from bhaskera.evaluation.scoring import score

logger = logging.getLogger(__name__)


def _world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def evaluate(model, tokenizer, eval_cfg, device: torch.device,
             task_names: list[str] | None = None) -> dict:
    """{task: metrics} for each task; an unsupported task reports why."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prefix = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    results = {}
    for name in task_names or eval_cfg.tasks:
        try:
            run = T.build(name, tokenizer, eval_cfg)
        except T.TaskUnsupported as exc:
            logger.warning(f"eval {name}: unsupported for this model ({exc})")
            results[name] = {"unsupported": str(exc)}
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        scores = score(model, run.requests, batch_size=eval_cfg.batch_size,
                       max_length=eval_cfg.max_length, pad_id=pad_id,
                       prefix_token=prefix, device=device)
        seconds = time.perf_counter() - t0
        metrics = run.finalize(scores)
        # Tokens actually forwarded (context + continuation, after truncation).
        tokens = sum(min(len(r.context or [prefix]) + len(r.continuation), eval_cfg.max_length + 1) - 1
                     for r in run.requests)
        metrics.update(requests=len(run.requests), seconds=round(seconds, 2),
                       tokens_per_sec=round(tokens / seconds, 1) if seconds else math.nan,
                       gpus=_world())
        results[name] = metrics
        logger.info(f"eval {name}: {_headline(metrics)}  ({seconds:.0f}s)")
    return results


_HEADLINE = ("acc_norm", "acc", "exact_match", "tool_selection_acc", "word_perplexity")


def _headline(m: dict) -> str:
    return ", ".join(f"{k}={m[k]:.4f}" for k in _HEADLINE if k in m) or str(m)


def write_results(results: dict, cfg, out_dir: str, extra: dict | None = None) -> str:
    """rank 0: <out_dir>/<run_name>.json plus a markdown table on stdout."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cfg.logging.run_name}.json")
    payload = {
        "model": cfg.model.name,
        "run_name": cfg.logging.run_name,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "evaluation": {k: getattr(cfg.evaluation, k) for k in
                       ("batch_size", "max_length", "limit", "data_dir", "chat_template_kwargs")},
        "results": results,
        **(extra or {}),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    lines = ["| task | metric | value | n | tok/s |", "|---|---|---|---|---|"]
    for task, m in results.items():
        if "unsupported" in m:
            lines.append(f"| {task} | unsupported | {m['unsupported'][:60]} | | |")
            continue
        for k in _HEADLINE:
            if k in m:
                lines.append(f"| {task} | {k} | {m[k]:.4f} | {m.get('n', '')} | "
                             f"{m.get('tokens_per_sec', '')} |")
    print("\n".join(lines), flush=True)
    return path
