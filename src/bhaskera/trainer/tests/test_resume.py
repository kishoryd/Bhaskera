"""Resume in the FSDP/DDP/QLoRA loop: the trainer-level maybe_resume finds the checkpoint, and the
learning-rate schedule continues instead of restarting. CPU, single process."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from bhaskera.distributed.checkpoint import _fill_missing_hyperparameters, save_checkpoint
from bhaskera.trainer.checkpointing import build_resumed_scheduler, fast_forward_scheduler, maybe_resume
from bhaskera.trainer.optim import build_scheduler


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(32, 32, bias=False)
        self.adapter = nn.Linear(32, 2, bias=False)
        self.base.weight.requires_grad_(False)

    def forward(self, x):
        return self.adapter(self.base(x))


def _opt(m):
    return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)


def test_trainer_maybe_resume_finds_the_checkpoint(tmp_path):
    torch.manual_seed(0)
    m = _Tiny()
    opt = _opt(m)
    for _ in range(2):
        opt.zero_grad()
        m(torch.randn(4, 32)).pow(2).mean().backward()
        opt.step()
    save_checkpoint(m, opt, step=7, path=str(tmp_path / "step_0000007"), rank=0)

    m2 = _Tiny()
    m2.base.load_state_dict(m.base.state_dict())
    # the folder passed in is the one that CONTAINS step_0000007, as in the training config
    assert maybe_resume(m2, _opt(m2), str(tmp_path)) == 7
    torch.testing.assert_close(m2.adapter.weight, m.adapter.weight)


def test_trainer_maybe_resume_without_checkpoint_returns_zero(tmp_path):
    assert maybe_resume(_Tiny(), _opt(_Tiny()), str(tmp_path)) == 0
    assert maybe_resume(_Tiny(), _opt(_Tiny()), str(tmp_path / "does-not-exist")) == 0


def test_scheduler_fast_forward_matches_stepping(tmp_path):
    cfg = SimpleNamespace(warmup_steps=5, max_steps=30)

    def fresh():
        m = _Tiny()
        o = _opt(m)
        return o, build_scheduler(o, cfg)

    o_ref, s_ref = fresh()
    for _ in range(12):
        o_ref.step()
        s_ref.step()
    o_new, s_new = fresh()
    fast_forward_scheduler(s_new, 12)
    assert s_new.get_last_lr() == s_ref.get_last_lr()
    # and it keeps going identically
    for _ in range(3):
        o_ref.step(); s_ref.step(); o_new.step(); s_new.step()
    assert s_new.get_last_lr() == s_ref.get_last_lr()


def test_resumed_schedule_matches_an_uninterrupted_run(tmp_path):
    """After resume the learning rate must equal what a run that never stopped has at that step
    (replaying the schedule on top of the restored optimizer would apply the warm-up twice)."""
    cfg = SimpleNamespace(warmup_steps=5, max_steps=30, lr=1e-2)
    torch.manual_seed(0)

    m_ref = _Tiny()
    o_ref = _opt(m_ref)
    s_ref = build_scheduler(o_ref, cfg)
    for _ in range(12):
        o_ref.zero_grad()
        m_ref(torch.randn(4, 32)).pow(2).mean().backward()
        o_ref.step()
        s_ref.step()
    save_checkpoint(m_ref, o_ref, step=12, path=str(tmp_path / "step_0000012"), rank=0)

    m2 = _Tiny()
    m2.base.load_state_dict(m_ref.base.state_dict())
    o2 = _opt(m2)
    step = maybe_resume(m2, o2, str(tmp_path))
    assert step == 12
    s2 = build_resumed_scheduler(o2, cfg, step)
    assert s2.get_last_lr() == s_ref.get_last_lr()
    for _ in range(4):                                  # and the two stay identical afterwards
        o_ref.step(); s_ref.step(); o2.step(); s2.step()
    assert s2.get_last_lr() == s_ref.get_last_lr()


def test_fresh_run_scheduler_unchanged():
    cfg = SimpleNamespace(warmup_steps=5, max_steps=30, lr=1e-2)
    o = _opt(_Tiny())
    s = build_resumed_scheduler(o, cfg, 0)
    assert s.get_last_lr() == build_scheduler(_opt(_Tiny()), cfg).get_last_lr()


def test_ddp_wrapped_model_saves_and_resumes(tmp_path):
    """DDP-wrapped model: PyTorch's set_model_state_dict raised KeyError('...embed_tokens.weight') here."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    try:
        torch.manual_seed(0)
        m = _Tiny()
        ddp = DDP(m)
        opt = _opt(m)
        for _ in range(2):
            opt.zero_grad()
            ddp(torch.randn(4, 32)).pow(2).mean().backward()
            opt.step()
        save_checkpoint(ddp, opt, step=5, path=str(tmp_path / "step_0000005"), rank=0)

        m2 = _Tiny()
        m2.base.load_state_dict(m.base.state_dict())
        ddp2 = DDP(m2)
        assert maybe_resume(ddp2, _opt(m2), str(tmp_path)) == 5
        torch.testing.assert_close(m2.adapter.weight, m.adapter.weight)
    finally:
        dist.destroy_process_group()


def test_missing_optimizer_settings_are_filled_in():
    """The optimizer load can return parameter groups without `betas` or `eps`; the next step must not fail."""
    opt = _opt(_Tiny())
    fresh = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
    for g in opt.param_groups:
        del g["betas"], g["eps"]
    _fill_missing_hyperparameters(opt, fresh)
    assert opt.param_groups[0]["betas"] == fresh[0]["betas"] and opt.param_groups[0]["eps"] == fresh[0]["eps"]
    opt.zero_grad()
    opt.param_groups[0]["params"][0].grad = torch.ones_like(opt.param_groups[0]["params"][0])
    opt.step()                                           # no KeyError
