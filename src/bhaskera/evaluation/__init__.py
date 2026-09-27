"""
bhaskera.evaluation
===================
Benchmark evaluation by log-likelihood scoring; runs on the same
FSDP/DDP-wrapped model as training (see scoring.py, tasks.py, runner.py).
"""
from __future__ import annotations

from .runner import evaluate, write_results
from .tasks import TASKS

__all__ = ["TASKS", "evaluate", "write_results"]
