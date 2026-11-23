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
    Save a checkpoint directory and keep the newest ckpt_cfg.keep_last_n of the completed checkpoints in the
    folder (0 keeps all). The newest one is never removed, and checkpoints left by an earlier run of the same
    job count too. All ranks must call this — it runs a collective.

    Older versions kept the N checkpoints with the lowest *training loss* from a list held in memory. That list
    was empty after a resume, so the checkpoint the run had resumed from stayed on disk for ever, and a newest
    checkpoint with a worse loss than an older one could be deleted, so a later resume lost progress.
    """
    Path(ckpt_cfg.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(ckpt_cfg.save_dir, f"step_{step:07d}")

    # Synchronise before any collective save.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Already saved at this step (a step checkpoint that falls on the end of an epoch or of the run): nothing to do.
    if os.path.exists(os.path.join(ckpt_path, ".complete")):
        return best_ckpts

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        path=ckpt_path,
        extra={"avg_loss": float(avg_loss)},
        rank=rank,
        keep_last_n=ckpt_cfg.keep_last_n,       # rank 0 removes the older checkpoints inside save_checkpoint
    )

    # The list (kept for the callers) now only names the checkpoints that are still on disk.
    if rank == 0:
        best_ckpts = [(loss, p) for loss, p in best_ckpts if os.path.isdir(p)]
        if all(p != ckpt_path for _, p in best_ckpts):
            best_ckpts.append((avg_loss, ckpt_path))
        logger.info(
            f"Checkpoint → {ckpt_path} | "
            f"kept: {sorted(d.name for d in Path(ckpt_cfg.save_dir).glob('step_*') if (d / '.complete').exists())}"
        )

    return best_ckpts


def maybe_resume(model, optimizer, save_dir: str, step: int = 0) -> int:
    """Resume from the newest complete checkpoint under `save_dir` (or from the one at `step`); return its step
    (0 if there is none).

    `load_checkpoint` takes the folder that *contains* the step_N folders and does the scan itself
    (skipping any without a .complete marker), so it is given `save_dir` itself, not a step_N folder.
    """
    return load_checkpoint(model, optimizer, save_dir, step)


def requested_resume_step(ckpt_cfg) -> int:
    """The step to resume from when one was asked for ($BHASKERA_RESUME_STEP, set by `submit.sh resume --step`, or
    checkpoint.resume_step), else 0 for the latest checkpoint."""
    raw = os.environ.get("BHASKERA_RESUME_STEP") or getattr(ckpt_cfg, "resume_step", 0) or 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def step_save_due(ckpt_cfg, step: int, max_steps: int) -> bool:
    """Whether a checkpoint is due after this optimizer step: every checkpoint.save_every_steps steps, not on the last
    step (the end of the run saves that one)."""
    every = int(getattr(ckpt_cfg, "save_every_steps", 0) or 0)
    return bool(ckpt_cfg.enabled and every > 0 and step > 0 and step % every == 0 and step < max_steps)


def final_save_due(ckpt_cfg, step: int, start_step: int) -> bool:
    """Whether the end of the run needs its own checkpoint: step saving is on and the run moved past where it started."""
    return bool(ckpt_cfg.enabled and int(getattr(ckpt_cfg, "save_every_steps", 0) or 0) > 0 and step > start_step)


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
    scheduler on that would take the decayed rate as its base. Every group is therefore put back at the
    configured base rate first (both groups made by build_optimizer use `train_cfg.lr`), and the schedule
    is then replayed up to `step`.
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