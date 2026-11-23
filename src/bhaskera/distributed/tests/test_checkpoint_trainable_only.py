"""Checkpoints hold only trainable (LoRA) parameters; resume restores them exactly. CPU, single process."""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from bhaskera.distributed.checkpoint import maybe_resume, save_checkpoint


class _Tiny(nn.Module):
    """A frozen 'base' layer plus a trainable 'adapter', like a LoRA-wrapped model."""

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(256, 256, bias=False)      # 256 KiB, frozen
        self.adapter = nn.Linear(256, 4, bias=False)     # 4 KiB, trainable
        self.base.weight.requires_grad_(False)

    def forward(self, x):
        return self.adapter(self.base(x))


def _dir_bytes(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def _train_a_bit(model, opt):
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 256)).pow(2).mean().backward()
        opt.step()


def test_checkpoint_excludes_frozen_weights_and_resumes(tmp_path):
    torch.manual_seed(0)
    m = _Tiny()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    _train_a_bit(m, opt)

    ckpt = str(tmp_path / "step_0000003")
    save_checkpoint(m, opt, step=3, path=ckpt, rank=0)
    assert os.path.exists(os.path.join(ckpt, ".complete"))
    # the frozen 256 KiB layer must not be in the checkpoint (adapter + optimizer state are a few KiB)
    assert _dir_bytes(os.path.join(ckpt, "model")) < 64 * 1024

    saved_adapter = m.adapter.weight.detach().clone()
    base_before = m.base.weight.detach().clone()

    m2 = _Tiny()
    m2.base.load_state_dict(m.base.state_dict())          # base comes from the model folder
    opt2 = torch.optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=1e-2)
    assert maybe_resume(m2, opt2, str(tmp_path)) == 3
    torch.testing.assert_close(m2.adapter.weight, saved_adapter)
    torch.testing.assert_close(m2.base.weight, base_before)
    assert len(opt2.state) == 1                             # optimizer moments restored for the adapter
