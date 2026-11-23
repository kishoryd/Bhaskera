"""keep_last_n keeps the newest checkpoints: never the 'best loss' ones, also across a resume."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from bhaskera.distributed import checkpoint as dckpt
from bhaskera.trainer import checkpointing


def _fake_save(**kw):
    """What save_checkpoint does to the disk: a folder with the .complete marker, then the clean-up of old ones."""
    path = Path(kw["path"])
    path.mkdir(parents=True)
    (path / ".complete").touch()
    dckpt._cleanup_old_checkpoints(str(path.parent), kw.get("keep_last_n", 3))


def _run(monkeypatch, tmp_path, steps_and_losses, keep, best=None):
    monkeypatch.setattr(checkpointing, "save_checkpoint", _fake_save)
    cfg = SimpleNamespace(save_dir=str(tmp_path), keep_last_n=keep)
    best = best or []
    for step, loss in steps_and_losses:
        best = checkpointing.save_and_prune(model=None, optimizer=None, step=step, avg_loss=loss, ckpt_cfg=cfg,
                                            rank=0, best_ckpts=best)
    return sorted(p.name for p in tmp_path.glob("step_*")), best


def test_the_newest_checkpoint_survives_even_with_the_worst_loss(monkeypatch, tmp_path):
    kept, _ = _run(monkeypatch, tmp_path, [(20, 1.0), (40, 1.5), (60, 2.0)], keep=1)
    assert kept == ["step_0000060"]                       # before: the newest (worst loss) was deleted


def test_keep_two_keeps_the_two_newest(monkeypatch, tmp_path):
    kept, best = _run(monkeypatch, tmp_path, [(10, 1.0), (20, 0.9), (30, 1.4), (40, 1.2)], keep=2)
    assert kept == ["step_0000030", "step_0000040"]
    assert sorted(os.path.basename(p) for _, p in best) == kept


def test_a_resumed_run_prunes_the_checkpoint_it_resumed_from(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, [(20, 1.2)], keep=1)                          # the first run
    kept, _ = _run(monkeypatch, tmp_path, [(24, 1.1)], keep=1)                # a resumed run, with an empty list
    assert kept == ["step_0000024"]                                           # before: both stayed (83 GB each)


def test_zero_means_keep_everything(monkeypatch, tmp_path):
    kept, _ = _run(monkeypatch, tmp_path, [(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0)], keep=0)
    assert len(kept) == 5                                                     # before: all of them were deleted


def test_only_completed_checkpoints_count(monkeypatch, tmp_path):
    (tmp_path / "step_0000010").mkdir()                                       # an interrupted save: no .complete
    kept, _ = _run(monkeypatch, tmp_path, [(20, 1.0), (30, 1.0)], keep=1)
    assert "step_0000030" in kept and "step_0000020" not in kept
