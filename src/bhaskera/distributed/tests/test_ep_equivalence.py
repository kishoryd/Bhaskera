"""
Expert-parallel forward/backward must match the HF MoE block it replaces.

Multi-GPU (NCCL), so run with torchrun rather than plain pytest:
    torchrun --nproc_per_node=2 -m bhaskera.distributed.tests.test_ep_equivalence

Every rank builds the same tiny HF block (same seed), feeds it its own input,
and compares the reference single-rank block against the EP version for the
output, router logits, input gradient and router-weight gradient, in fp32.
Covers Qwen3Moe with and without norm_topk_prob and Mixtral, and inputs
small enough that some experts receive no tokens.
"""
from __future__ import annotations

import copy
import os
import sys
import types

import torch
import torch.distributed as dist
from torch import nn

from bhaskera.distributed.expert_parallel import apply_expert_parallel


def _blocks():
    from transformers.models.mixtral.configuration_mixtral import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    def qwen(norm):
        return Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(
            hidden_size=64, moe_intermediate_size=32, num_experts=8,
            num_experts_per_tok=2, norm_topk_prob=norm,
        ))

    def mixtral():
        return MixtralSparseMoeBlock(MixtralConfig(
            hidden_size=64, intermediate_size=32, num_local_experts=8,
            num_experts_per_tok=2, router_jitter_noise=0.0,
        ))

    return [("qwen3moe_norm", lambda: qwen(True)),
            ("qwen3moe_nonorm", lambda: qwen(False)),
            ("mixtral", mixtral)]


class _Wrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


def _check(name, make_block, shape, device, rank, world):
    torch.manual_seed(0)
    ref = make_block().to(device=device, dtype=torch.float32)
    for p in ref.experts.parameters():
        p.requires_grad_(False)
    model = _Wrapper(copy.deepcopy(ref))

    sd = model.state_dict() if rank == 0 else None
    profile = types.SimpleNamespace(model_dtype=torch.float32)
    apply_expert_parallel(model, profile, world, sd, device)

    torch.manual_seed(100 + rank)
    x = torch.randn(*shape, device=device)
    grad_out = torch.randn(*shape, device=device)

    x_ref = x.clone().requires_grad_(True)
    out_ref, logits_ref = ref(x_ref)
    out_ref.backward(grad_out)

    x_ep = x.clone().requires_grad_(True)
    out_ep, logits_ep = model.mlp(x_ep)
    out_ep.backward(grad_out)

    checks = {
        "output": (out_ep, out_ref),
        "router_logits": (logits_ep, logits_ref),
        "input_grad": (x_ep.grad, x_ref.grad),
        "router_grad": (model.mlp.gate.weight.grad, ref.gate.weight.grad),
    }
    ok = True
    for what, (a, b) in checks.items():
        err = (a - b).abs().max().item()
        passed = torch.allclose(a, b, rtol=1e-5, atol=1e-5)
        ok &= passed
        print(f"[rank {rank}] {name:16s} {shape!s:14} {what:14s} "
              f"max_abs_err={err:.2e} {'OK' if passed else 'FAIL'}", flush=True)
    return ok


def _check_checkpointed(name, make_block, device, rank, world):
    """EP block under activation checkpointing with recompute-only router noise:
    must not raise, and without noise must match the un-checkpointed EP block."""
    from torch.distributed._composable import checkpoint

    from bhaskera.distributed import moe_routing as mr

    def build(ac: bool, noise: bool):
        torch.manual_seed(0)
        blk = make_block().to(device=device, dtype=torch.float32)
        for p in blk.experts.parameters():
            p.requires_grad_(False)
        model = _Wrapper(blk)
        sd = model.state_dict() if rank == 0 else None
        apply_expert_parallel(model, types.SimpleNamespace(model_dtype=torch.float32), world, sd, device)
        if noise:
            def hook(_m, _i, out):
                return out + 3.0 * torch.randn_like(out) if getattr(mr._state, "phase", None) == "recompute" else out
            model.mlp.gate.register_forward_hook(hook)
        if ac:
            checkpoint(model.mlp, context_fn=mr.recompute_contexts)
        return model

    torch.manual_seed(200 + rank)
    x0 = torch.randn(2, 16, 64, device=device)
    g = torch.randn(2, 16, 64, device=device)
    grads = {}
    for tag, ac, noise in (("plain", False, False), ("ac", True, False), ("ac+noise", True, True)):
        model = build(ac, noise)
        x = x0.clone().requires_grad_(True)
        try:
            model.mlp(x)[0].backward(g)
            grads[tag] = x.grad
        except Exception as e:  # noqa: BLE001 -- report any failure as a test failure
            print(f"[rank {rank}] {name:16s} checkpointed {tag}: {type(e).__name__}: {str(e)[:120]} FAIL", flush=True)
            return False
    err = (grads["ac"] - grads["plain"]).abs().max().item()
    ok = err < 1e-5
    print(f"[rank {rank}] {name:16s} checkpointed: AC==plain max_abs_err={err:.2e}, "
          f"recompute noise survived {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> int:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    ok = True
    for name, make in _blocks():
        for shape in [(2, 16, 64), (1, 3, 64)]:  # (1,3): most experts get no tokens
            ok &= _check(name, make, shape, device, rank, world)

    for name, make in _blocks():
        ok &= _check_checkpointed(name, make, device, rank, world)

    flag = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("=== EP EQUIVALENCE:", "PASS" if flag.item() else "FAIL", "===", flush=True)
    dist.destroy_process_group()
    return 0 if flag.item() else 1


if __name__ == "__main__":
    sys.exit(main())
