"""
bhaskera.trainer.checkpointing
==============================
Periodic-save and resume helpers.


"""
from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

import torch
import torch.distributed as dist

from bhaskera.distributed import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"step_(\d+)")


def save_and_prune(
    *,
    model,
    optimizer,
    step: int,
    avg_loss: float,
    ckpt_cfg,
    rank: int,
    best_ckpts: list[tuple[float, str]],
) -> list[tuple[float, str]]:
    """
    Save a checkpoint directory and prune down to ckpt_cfg.keep_last_n
    best-by-loss.  All ranks must call this — it runs a collective.
    """
    Path(ckpt_cfg.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(ckpt_cfg.save_dir, f"step_{step:07d}")

    # Synchronise before any collective save.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        path=ckpt_path,
        extra={"avg_loss": float(avg_loss)},
        rank=rank,
    )

    # Only rank 0 manages the "best" list on disk.
    if rank == 0:
        best_ckpts.append((avg_loss, ckpt_path))
        best_ckpts.sort(key=lambda x: x[0])  # ascending: lower loss is better
        while len(best_ckpts) > ckpt_cfg.keep_last_n:
            _, old_path = best_ckpts.pop()
            _safe_rmtree(old_path)
        logger.info(
            f"Checkpoint → {ckpt_path} | "
            f"kept: {[os.path.basename(p) for _, p in best_ckpts]}"
        )

    return best_ckpts


def maybe_resume(model, optimizer, save_dir: str) -> int:
    """Resume from the newest complete checkpoint under `save_dir`; return its step (0 if there is none).

    `load_checkpoint` takes the folder that *contains* the step_N folders and does the scan itself
    (skipping any without a .complete marker). This function used to pick the newest step_N folder and
    pass that, so the scan found nothing inside it and every resume silently restarted at step 0.
    """
    return load_checkpoint(model, optimizer, save_dir)


def fast_forward_scheduler(scheduler, steps: int) -> None:
    """Advance a freshly built learning-rate scheduler by `steps` optimizer steps."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # "scheduler.step() before optimizer.step()" is expected here
        for _ in range(max(0, int(steps))):
            scheduler.step()


def build_resumed_scheduler(optimizer, train_cfg, step: int):
    """Build the learning-rate scheduler and position it at `step` (0 for a fresh run).

    A resumed optimizer carries the learning rate of the saved step in its parameter groups, and no
    `initial_lr` (the checkpoint only restores keys the fresh optimizer already had). Building the
    scheduler on that would take the decayed rate as its base, so every group is first put back at the
    configured base rate, both groups of build_optimizer use `train_cfg.lr`, and the schedule is then
    replayed up to `step`.
    """
    from .optim import build_scheduler
    for group in optimizer.param_groups:
        group["lr"] = train_cfg.lr
        group.pop("initial_lr", None)
    scheduler = build_scheduler(optimizer, train_cfg)
    fast_forward_scheduler(scheduler, step)
    return scheduler


def _step_of(p: Path) -> int:
    m = _STEP_RE.search(p.name)
    return int(m.group(1)) if m else -1


def _safe_rmtree(path: str) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Failed to remove old checkpoint {path}: {e}")