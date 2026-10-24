"""Timestamped start-up milestones for a training worker.

The worker marks each stage it finishes (distributed ready, model loaded, model
wrapped, first optimizer step done). Rank 0 logs one line per mark with the time
since the previous mark and since the worker started, so a run's log answers
"how long did start-up take, and where?" without profiling.

Times are wall-clock for this worker process only. Ray start-up before the
worker exists and the data-set build in the driver are not included; those show
up in the driver's own log timestamps.
"""
from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger(__name__)

_t0 = time.time()
_last = _t0
_seen: set[str] = set()


def mark(name: str, rank: int) -> None:
    """Log the time since the previous mark and since start, once per name, on rank 0."""
    global _last
    if name in _seen:
        return
    _seen.add(name)
    if torch.cuda.is_available():
        torch.cuda.synchronize()   # GPU work done before the clock is read
    now = time.time()
    if rank == 0:
        logger.info(f"[startup] {name}: +{now - _last:.1f}s (total {now - _t0:.1f}s since worker start)")
    _last = now
