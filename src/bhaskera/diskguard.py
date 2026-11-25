"""Disk space checks, so that a job does not fail half way because a disk is full (home folders fill up fast on a cluster).

One place decides what "enough room" means, and every action that writes a lot uses it: new recipes (checkpoints), starting or resuming
training, preparing data, downloads, and a floor before any job (its logs and results).

The number checked is the free space of the file system that holds the folder. A quota on the user's account can be lower than that and
is not visible from here, so a message says so.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

GIB = 2 ** 30
FLOOR = 1 * GIB                  # a job needs this much anywhere it runs: logs, MLflow, a small checkpoint
MARGIN = 1.1                     # room beyond the estimate
LOW_GB = 20.0                    # below this the page warns on Home


class DiskError(ValueError):
    """Not enough room, in words for the person who chose the folder."""


def _existing(p: Path) -> Path:
    p = Path(p)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def free_bytes(path) -> int:
    try:
        return shutil.disk_usage(_existing(Path(path))).free
    except OSError:
        return -1                # unknown: never block on a disk we cannot read


def human(n: float) -> str:
    return f"{n / GIB:.1f} GB" if n >= GIB / 10 else f"{n / 2**20:.0f} MB"


def require(path, need: float, what: str, margin: float = MARGIN) -> int:
    """Raise DiskError unless the disk holding ``path`` has ``need`` bytes (plus a margin) free. Returns the free bytes."""
    free = free_bytes(path)
    if free >= 0 and need and free < need * margin:
        raise DiskError(f"there is not enough space for {what} in {path}: about {human(need)} is needed and {human(free)} is free on that disk "
                        "(a quota on your account can be lower). Choose a folder on a disk with more room.")
    return free


def floor(path, what: str = "this job's logs and results") -> int:
    """Refuse to start anything on a disk that is practically full."""
    free = free_bytes(path)
    if 0 <= free < FLOOR:
        raise DiskError(f"the disk holding {path} has only {human(free)} free, not enough for {what}. Free some space, or use a folder on another disk.")
    return free


def model_bytes(folder) -> int:
    """Size of a model's weight files (.safetensors and .bin) in bytes."""
    try:
        with os.scandir(folder) as it:
            return sum(e.stat().st_size for e in it if e.name.endswith((".safetensors", ".bin")) and e.is_file())
    except OSError:
        return 0


def checkpoint_need(model_folder, keep: int) -> tuple[int, int]:
    """(one checkpoint, all that are kept) in bytes. A checkpoint holds the whole wrapped model, so about the size of the weights.
    ``keep`` of 0 means keep all, which has no limit: it is counted as 3 here."""
    one = model_bytes(model_folder)
    return one, one * (keep if keep and keep > 0 else 3)


def note(path, need_one: int, need_all: int, keep: int) -> str:
    """One line for a preview: where the checkpoints go, how much room there is and how much they need."""
    free = free_bytes(path)
    if free < 0 or not need_one:
        return ""
    tail = "enough room" if free >= need_all * MARGIN else ("room for one, not for all of them" if free >= need_one * MARGIN else "NOT ENOUGH ROOM")
    return (f"Disk: {human(free)} free where the checkpoints go ({path}); each is about {human(need_one)} and {keep if keep else 'all'} "
            f"{'is' if keep == 1 else 'are'} kept: {tail}.")


def run_folder_status(folder: Optional[Path]) -> Optional[dict]:
    """{'path', 'free_gb', 'low'} for the disk of a run folder, for the home page; None when unknown."""
    if not folder:
        return None
    try:
        u = shutil.disk_usage(_existing(Path(folder)))
    except OSError:
        return None
    gb = u.free / GIB
    return {"path": str(folder), "free_gb": round(gb, 1), "total_gb": round(u.total / GIB, 1), "low": gb < LOW_GB}
