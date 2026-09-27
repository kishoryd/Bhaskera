"""
bhaskera.trainer.benchmark
==========================
End-of-run training benchmark: one JSON per run, comparable across models
(dense / MoE), strategies and node counts.

Records the throughput tracker's per-step output, skipping the first
`warmup` steps (compile, cache and allocator warm-up; the tracker itself
emits tokens/s from its second step), and reports medians -- a single slow
step (checkpoint, straggler) does not move a median the way it moves a
mean. Peak GPU memory is the maximum over all ranks, since the
largest rank is what decides whether a config fits.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import statistics

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_KEYS = ("throughput/tokens_per_sec", "throughput/tokens_per_sec_per_gpu",
         "throughput/mfu_pct", "throughput/step_time_s")


class TrainBenchmark:
    def __init__(self, path: str, warmup: int = 0):
        self.path = path
        self.warmup = max(0, int(warmup))
        self.seen = 0
        self.history: dict[str, list[float]] = {k: [] for k in _KEYS}

    def record(self, step_metrics: dict) -> None:
        self.seen += 1
        if self.seen <= self.warmup:
            return
        for k in _KEYS:
            if k in step_metrics:
                self.history[k].append(float(step_metrics[k]))

    def finish(self, cfg, profile, device: torch.device, rank: int, world: int,
               steps: int, extra: dict | None = None) -> None:
        """Collective (peak memory); rank 0 writes the JSON."""
        peak = torch.tensor(float(torch.cuda.max_memory_allocated(device)) / 2**30, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if rank != 0:
            return
        med = {k.split("/")[-1]: round(statistics.median(v), 4) if v else None
               for k, v in self.history.items()}
        payload = {
            "model": cfg.model.name,
            "run_name": cfg.logging.run_name,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "moe": bool(profile.is_moe),
            "strategy": cfg.training.distributed.strategy,
            "world_size": world,
            "seq_len": cfg.data.seq_len,
            "batch_size": cfg.training.batch_size,
            "grad_accum": cfg.training.grad_accum,
            "cpt": bool(getattr(cfg.data, "cpt", False)),
            "lora": bool(cfg.lora.enabled),
            "steps": steps,
            "warmup_steps": self.warmup,
            "measured_steps": len(self.history["throughput/step_time_s"]),
            "median": med,
            "peak_gpu_mem_gib": round(float(peak), 2),
            **(extra or {}),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"training benchmark -> {self.path}: {med}, peak {payload['peak_gpu_mem_gib']} GiB")
