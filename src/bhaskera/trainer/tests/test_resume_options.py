"""When a job fails half way: checkpoints every N steps, resuming from a chosen step, and what happens to later checkpoints."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from bhaskera.distributed.checkpoint import list_checkpoints, save_checkpoint
from bhaskera.trainer import checkpointing
from bhaskera.trainer.checkpointing import (final_save_due, maybe_resume, requested_resume_step, save_and_prune,
                                            step_save_due)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(16, 16, bias=False)
        self.adapter = nn.Linear(16, 2, bias=False)
        self.base.weight.requires_grad_(False)

    def forward(self, x):
        return self.adapter(self.base(x))


def _opt(m):
    return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)


def _train(m, opt, n):
    for _ in range(n):
        opt.zero_grad()
        m(torch.randn(4, 16)).pow(2).mean().backward()
        opt.step()


def _make_checkpoints(tmp_path, steps):
    """A model trained a little more between each of the steps; returns the adapter weights at each step."""
    torch.manual_seed(0)
    m = _Tiny()
    opt = _opt(m)
    weights = {}
    for s in steps:
        _train(m, opt, 3)
        save_checkpoint(m, opt, step=s, path=str(tmp_path / f"step_{s:07d}"), rank=0)
        weights[s] = m.adapter.weight.detach().clone()
    return m, weights


def test_step_save_is_due_every_n_steps_but_not_on_the_last():
    cfg = SimpleNamespace(enabled=True, save_every_steps=100)
    due = [s for s in range(0, 501) if step_save_due(cfg, s, 500)]
    assert due == [100, 200, 300, 400]                  # 500 is the last step: the end of the run saves it
    assert not step_save_due(SimpleNamespace(enabled=True, save_every_steps=0), 100, 500)
    assert not step_save_due(SimpleNamespace(enabled=False, save_every_steps=100), 100, 500)


def test_a_final_save_is_due_only_when_the_run_moved_on():
    on = SimpleNamespace(enabled=True, save_every_steps=50)
    assert final_save_due(on, step=120, start_step=0)
    assert not final_save_due(on, step=100, start_step=100)         # a resumed run that did nothing
    assert not final_save_due(SimpleNamespace(enabled=True, save_every_steps=0), step=120, start_step=0)


def test_resume_from_the_latest_by_default(tmp_path):
    _m, w = _make_checkpoints(tmp_path, [100, 200, 300])
    m2 = _Tiny()
    m2.base.load_state_dict(_m.base.state_dict())
    assert maybe_resume(m2, _opt(m2), str(tmp_path)) == 300
    torch.testing.assert_close(m2.adapter.weight, w[300])
    assert list_checkpoints(str(tmp_path)) == [100, 200, 300]


def test_resume_from_an_earlier_step_sets_the_later_ones_aside(tmp_path):
    base, w = _make_checkpoints(tmp_path, [100, 200, 300])
    m2 = _Tiny()
    m2.base.load_state_dict(base.base.state_dict())
    assert maybe_resume(m2, _opt(m2), str(tmp_path), step=100) == 100
    torch.testing.assert_close(m2.adapter.weight, w[100])
    assert list_checkpoints(str(tmp_path)) == [100]                 # 200 and 300 no longer count as the newest
    assert sorted(p.name for p in Path(tmp_path).glob("rolledback-*")) == ["rolledback-0000200", "rolledback-0000300"]
    assert (tmp_path / "rolledback-0000300" / ".complete").exists(), "set aside, not deleted"


def test_a_step_that_was_not_saved_is_an_error_that_lists_the_ones_there(tmp_path):
    _make_checkpoints(tmp_path, [100, 200])
    with pytest.raises(FileNotFoundError, match="steps 100, 200"):
        maybe_resume(_Tiny(), _opt(_Tiny()), str(tmp_path), step=150)


def test_requested_step_comes_from_the_environment_or_the_config(monkeypatch):
    cfg = SimpleNamespace(resume_step=0)
    monkeypatch.delenv("BHASKERA_RESUME_STEP", raising=False)
    assert requested_resume_step(cfg) == 0
    assert requested_resume_step(SimpleNamespace(resume_step=200)) == 200
    monkeypatch.setenv("BHASKERA_RESUME_STEP", "300")
    assert requested_resume_step(cfg) == 300
    monkeypatch.setenv("BHASKERA_RESUME_STEP", "not a number")
    assert requested_resume_step(cfg) == 0


def test_saving_the_same_step_twice_is_harmless(monkeypatch, tmp_path):
    calls = []

    def fake(**kw):
        p = Path(kw["path"])
        p.mkdir(parents=True)
        (p / ".complete").touch()
        calls.append(kw["step"])

    monkeypatch.setattr(checkpointing, "save_checkpoint", fake)
    cfg = SimpleNamespace(save_dir=str(tmp_path), keep_last_n=3)
    best = save_and_prune(model=None, optimizer=None, step=40, avg_loss=1.0, ckpt_cfg=cfg, rank=0, best_ckpts=[])
    save_and_prune(model=None, optimizer=None, step=40, avg_loss=1.0, ckpt_cfg=cfg, rank=0, best_ckpts=best)
    assert calls == [40], "an epoch that ends on a step that was just saved must not save it again"
