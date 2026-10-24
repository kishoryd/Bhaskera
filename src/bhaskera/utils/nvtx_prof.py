"""Per-step NVTX ranges and an Nsight Systems capture window for the train loop.

Everything here is a no-op unless BHASKERA_NSYS=1, so the default training path
pays nothing. With it on, the loop labels each phase of a step (data wait,
forward + loss, backward, gradient clipping, optimizer step) so the Nsight
Systems timeline shows where a step's wall time goes.

The capture window uses cudaProfilerStart/Stop, which Nsight Systems honours
under ``--capture-range=cudaProfilerApi``: only steps START..STOP-1 are
recorded, so the report stays small and skips CUDA/NCCL warm-up.

    BHASKERA_NSYS=1                  turn the markers and capture window on
    BHASKERA_NSYS_START_STEP=12      first captured step (0-based loop index)
    BHASKERA_NSYS_STOP_STEP=14       capture stops when this step begins
    BHASKERA_NSYS_FLUSH_SECONDS=10   pause after the capture ends so reports are written
"""
from __future__ import annotations

import contextlib
import os
import time

import torch

ENABLED = os.environ.get("BHASKERA_NSYS") == "1"
START_STEP = int(os.environ.get("BHASKERA_NSYS_START_STEP", "12"))
STOP_STEP = int(os.environ.get("BHASKERA_NSYS_STOP_STEP", "14"))
# Seconds to wait after the capture stops, so Nsight Systems on every node finishes
# writing its report before it shuts the job down (--capture-range-end=stop-shutdown).
FLUSH_SECONDS = float(os.environ.get("BHASKERA_NSYS_FLUSH_SECONDS", "10"))

_capturing = False


@contextlib.contextmanager
def nvtx_range(name: str):
    """Label the enclosed CPU-side region on the Nsight Systems timeline."""
    if not ENABLED:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def capture_window(step: int) -> None:
    """Start/stop the Nsight Systems capture as the loop reaches START/STOP."""
    global _capturing
    if not ENABLED:
        return
    if step == START_STEP and not _capturing:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        _capturing = True
    elif step >= STOP_STEP and _capturing:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        _capturing = False
        time.sleep(FLUSH_SECONDS)


def capture_end() -> None:
    """Close the capture if training ends before STOP_STEP."""
    global _capturing
    if ENABLED and _capturing:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        _capturing = False
        time.sleep(FLUSH_SECONDS)
