"""
bhaskera.distributed.moe_experts
================================
Fast compute for frozen MoE experts (strategy: pipeline).

Why: HF's sparse MoE blocks loop over experts in Python and call torch.where per expert.
Each call has a data-dependent output size, so the host waits for the GPU once per expert:
128 waits per layer per micro-batch on Qwen3-MoE, paid again in the activation-checkpoint
recompute and in backward, on GEMMs of a few hundred rows. The layer ends up bound by
launches and host syncs, not FLOPs (Qwen3-30B-A3B ran at 4.7% MFU under the pipeline).

With LoRA the experts are frozen, so they need no weight gradients and may be re-stored in
any layout. stack_frozen_experts() moves each block's expert weights into three stacked
tensors [E, out, in] (same memory; the per-expert Linears are freed) and expert_forward()
computes all experts with one host sync per layer:

  sort the (token, expert) pairs by expert, then either
  - "loop": one GEMM chain per expert on its contiguous slice (exact rows, E launches), or
  - "bmm":  scatter into an [E, capacity, H] buffer and run three batched GEMMs
            (3 launches, but pads every expert to the busiest one's token count).
  "auto" uses bmm while the padding costs at most `max_pad` x the useful rows.

Routing still comes from moe_routing.select_experts, so the forward/recompute pinning is kept.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

# (gate, up, down) projection names inside one expert MLP
_LAYOUTS = (("gate_proj", "up_proj", "down_proj"),   # Qwen2/3-MoE, DeepSeek-style
            ("w1", "w3", "w2"))                       # Mixtral


def _layout(expert: nn.Module):
    for names in _LAYOUTS:
        mods = [getattr(expert, n, None) for n in names]
        if all(isinstance(m, nn.Linear) and m.bias is None for m in mods) and hasattr(expert, "act_fn"):
            return names
    return None


def stack_frozen_experts(block: nn.Module, attr: str, mode: str = "auto", max_pad: float = 1.5) -> bool:
    """Re-store block.<attr> (a ModuleList of expert MLPs) as stacked frozen tensors.
    Returns False (and leaves the block untouched) for a layout it does not know."""
    experts = block._modules.get(attr)
    if not isinstance(experts, nn.ModuleList) or not len(experts):
        return False
    names = _layout(experts[0])
    if names is None or any(_layout(e) != names for e in experts):
        return False
    if any(p.requires_grad for p in experts.parameters()):
        return False                       # LoRA on experts (lora.include_experts): keep modules
    for key, n in zip(("_bk_gate", "_bk_up", "_bk_down"), names):
        w = torch.stack([getattr(e, n).weight.detach() for e in experts])
        for e in experts:                  # free each expert's copy as soon as it is stacked
            getattr(e, n).weight = None
        block.register_buffer(key, w, persistent=False)
    block._bk_act = experts[0].act_fn
    block._bk_mode, block._bk_max_pad = mode, float(max_pad)
    del block._modules[attr]
    block._bk_num_experts = block._bk_gate.shape[0]
    return True


def expert_forward(block: nn.Module, x: torch.Tensor, selected: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
    """x [T, H], selected [T, k] expert ids, weights [T, k] routing weights -> [T, H]."""
    g, u, d, act = block._bk_gate, block._bk_up, block._bk_down, block._bk_act
    E, k = g.shape[0], selected.shape[1]
    flat = selected.reshape(-1)
    order = flat.argsort(stable=True)
    src = order // k                                   # token of each sorted pair
    counts = torch.bincount(flat, minlength=E)
    n = flat.numel()
    mode = block._bk_mode
    if mode != "loop":
        cap = int(counts.max())                        # the one host sync (bmm)
        if mode == "auto" and cap * E > block._bk_max_pad * n:
            mode = "loop"
    if mode == "loop":
        sizes = counts.tolist()                        # the one host sync (loop)
        outs = [F.linear(act(F.linear(c, g[e])) * F.linear(c, u[e]), d[e])
                for e, c in enumerate(x[src].split(sizes)) if sizes[e]]
        ys = torch.cat(outs)
    else:
        starts = torch.cumsum(counts, 0) - counts
        sel_sorted = flat[order]
        slot = sel_sorted * cap + (torch.arange(n, device=x.device) - starts[sel_sorted])
        buf = x.new_zeros(E * cap, x.shape[1]).index_copy(0, slot, x[src]).view(E, cap, -1)
        h = torch.bmm(act(torch.bmm(buf, g.transpose(1, 2))) * torch.bmm(buf, u.transpose(1, 2)),
                      d.transpose(1, 2))
        ys = h.reshape(E * cap, -1)[slot]
    ys = ys * weights.reshape(-1)[order, None].to(ys.dtype)
    return torch.zeros_like(x).index_add_(0, src, ys.to(x.dtype))
