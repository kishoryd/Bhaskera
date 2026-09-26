"""
bhaskera.trainer.step_timing
============================
Per-optimizer-step time breakdown and cross-rank straggler detection.

Why: at 64 nodes MFU collapsed to ~0-1% on both MoE models that trained,
with tokens/s swinging from thousands to double digits step to step, and
the logs could not say whether the time went to waiting on Ray Data, to
forward/backward (which includes FSDP's collectives), or to one slow node.

Each step is split into:
  data      host time blocked in next(loader)
  compute   forward+backward on the GPU (CUDA events; includes the FSDP
            all-gathers/reduce-scatters they wait on)
  optimizer grad clipping + optimizer/scheduler step (CUDA events)
  other     remainder of the wall-clock step (logging, host overhead)
GPU timings come from CUDA events read after the loop's existing loss
.item() sync, so no extra synchronisation is added. Every
straggler_every_n_steps, each rank's (step, data, compute) is all-gathered
and rank 0 reports the slowest rank and its host -- one bad node stalls every
collective, so it shows up as the rank the others wait for.
"""
from __future__ import annotations

import contextlib
import logging
import socket
import statistics
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class StepTimer:
    def __init__(
        self, *, enabled: bool, device: torch.device, rank: int,
        world_size: int, straggler_every_n_steps: int,
    ) -> None:
        self.enabled = bool(enabled) and device.type == "cuda"
        self.device, self.rank, self.world_size = device, rank, world_size
        self.straggler_every = int(straggler_every_n_steps or 0)
        self._hosts: list | None = None
        self._reset()

    def _reset(self) -> None:
        self._t0 = time.perf_counter()
        self._data = 0.0
        self._compute: list = []
        self._optim: tuple | None = None
        self._open: torch.cuda.Event | None = None

    def _event(self) -> torch.cuda.Event:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def start_window(self) -> None:
        if self.enabled:
            self._reset()

    @contextlib.contextmanager
    def data_wait(self):
        t = time.perf_counter()
        try:
            yield
        finally:
            if self.enabled:
                self._data += time.perf_counter() - t

    def compute_start(self) -> None:
        if self.enabled:
            self._open = self._event()

    def compute_end(self) -> None:
        if self.enabled and self._open is not None:
            self._compute.append((self._open, self._event()))
            self._open = None

    def optim_start(self) -> None:
        if self.enabled:
            self._open = self._event()

    def optim_end(self) -> None:
        if self.enabled and self._open is not None:
            self._optim = (self._open, self._event())
            self._open = None

    def finish(self, step: int) -> dict[str, float]:
        """Call on EVERY rank after a device sync; may run a collective."""
        if not self.enabled:
            return {}
        total = time.perf_counter() - self._t0
        compute = sum(s.elapsed_time(e) for s, e in self._compute) / 1000.0
        optim = self._optim[0].elapsed_time(self._optim[1]) / 1000.0 if self._optim else 0.0
        metrics = {
            "time/step_s": total,
            "time/data_wait_s": self._data,
            "time/compute_s": compute,
            "time/optimizer_s": optim,
            "time/other_s": max(0.0, total - self._data - compute - optim),
            "time/data_wait_pct": 100.0 * self._data / total if total > 0 else 0.0,
        }
        if (self.straggler_every > 0 and step % self.straggler_every == 0
                and dist.is_available() and dist.is_initialized() and self.world_size > 1):
            metrics.update(self._stragglers(total, self._data, compute, step))
        return metrics

    def _stragglers(self, total: float, data: float, compute: float, step: int) -> dict:
        if self._hosts is None:
            self._hosts = [None] * self.world_size
            dist.all_gather_object(self._hosts, socket.gethostname())
        mine = torch.tensor([total, data, compute], device=self.device, dtype=torch.float64)
        every = torch.empty(self.world_size, 3, device=self.device, dtype=torch.float64)
        dist.all_gather_into_tensor(every, mine)
        if self.rank != 0:
            return {}
        rows = every.tolist()
        steps = [r[0] for r in rows]
        slowest = max(range(len(rows)), key=lambda i: rows[i][0])
        slowest_data = max(range(len(rows)), key=lambda i: rows[i][1])
        median = statistics.median(steps)
        out = {
            "time/step_max_s": rows[slowest][0],
            "time/step_median_s": median,
            "time/slowest_rank": float(slowest),
            "time/data_wait_max_s": rows[slowest_data][1],
        }
        if rows[slowest][0] > 1.25 * median and rows[slowest][0] - median > 1.0:
            logger.warning(
                f"[step {step}] straggler: rank {slowest} on {self._hosts[slowest]} took "
                f"{rows[slowest][0]:.2f}s vs median {median:.2f}s "
                f"(its data wait {rows[slowest][1]:.2f}s, compute {rows[slowest][2]:.2f}s)"
            )
        return out
