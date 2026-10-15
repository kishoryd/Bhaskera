"""
bhaskera.distributed.moe_routing
================================
Keep MoE routing identical between a checkpointed layer's forward and its
recomputation.

Why: with decoder-layer activation checkpointing, the backward pass reruns
each MoE layer. HF's routed MoE blocks run only the experts that received
tokens, so the number and shapes of saved tensors depend on the top-k
routing. The recomputed router logits are not bitwise identical to the
original (bf16 GEMMs are not guaranteed reproducible across calls), and when
two experts are nearly tied the top-k choice flips; torch then aborts with
CheckpointError ("different number of tensors" / "different metadata"). This
killed Qwen3-30B-A3B on step 1 on 2 nodes.

Routing is a discrete, non-differentiable decision, so the recompute can
safely reuse the forward's choice: during the original forward each block
records its top-k expert indices; during recompute it pops them instead of
recomputing top-k. Routing *weights* are still recomputed from the current
logits (they are differentiable). Phases come from torch's checkpoint
context_fn hook, which runs the forward and the recompute under separate
context managers.
"""
from __future__ import annotations

import collections
import contextlib
import logging
import threading
import types

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

_EXPERT_ATTRS = ("experts", "local_experts", "routed_experts")
_state = threading.local()


@contextlib.contextmanager
def _phase(name: str):
    prev = getattr(_state, "phase", None)
    _state.phase = name
    try:
        yield
    finally:
        _state.phase = prev


def recompute_contexts():
    """context_fn for torch checkpoint: (forward ctx, recompute ctx)."""
    return _phase("forward"), _phase("recompute")


def select_experts(block: nn.Module, probs: torch.Tensor, top_k: int) -> torch.Tensor:
    """top-k expert indices for this block, reused verbatim on recompute."""
    phase = getattr(_state, "phase", None)
    cache = block.__dict__.setdefault("_bhaskera_route_cache", collections.deque())
    if phase == "recompute" and cache:
        return cache.popleft()
    # Detached: only the indices are needed (weights come from a gather that
    # carries the gradient). topk on the live tensor would save a tensor for
    # backward in the forward but not in the cached recompute -- the saved-
    # tensor counts would then differ, the very error this module prevents.
    _, selected = torch.topk(probs.detach(), top_k, dim=-1)
    if phase == "forward" and torch.is_grad_enabled():
        cache.clear()  # anything left belongs to a backward that never ran
        cache.append(selected)
    return selected


def routed_moe_blocks(model: nn.Module) -> tuple[list, list]:
    """(supported, unsupported) MoE blocks as (name, module, experts_attr)."""
    ok, other = [], []
    for name, module in model.named_modules():
        attr = next(
            (a for a in _EXPERT_ATTRS
             if isinstance(module._modules.get(a), nn.ModuleList) and len(module._modules[a])),
            None,
        )
        if attr is None:
            continue
        supported = (
            isinstance(getattr(module, "gate", None), nn.Linear)
            and hasattr(module, "top_k")
            and not any(hasattr(module, s) for s in ("shared_expert", "shared_experts"))
        )
        (ok if supported else other).append((name, module, attr))
    return ok, other


def _stable_forward(self, hidden_states: torch.Tensor):
    """HF Qwen3Moe/Mixtral SparseMoeBlock.forward, top-k via select_experts."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    jitter = float(getattr(self, "jitter_noise", 0.0) or 0.0)
    if self.training and jitter > 0:
        hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
            1.0 - jitter, 1.0 + jitter
        )
    x = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(x)
    probs = F.softmax(router_logits, dim=1, dtype=torch.float)
    selected = select_experts(self, probs, self.top_k)
    weights = probs.gather(1, selected)
    if getattr(self, "norm_topk_prob", True):
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(x.dtype)

    experts = self._modules[self._bhaskera_experts_attr]
    out = torch.zeros((batch_size * seq_len, hidden_dim), dtype=x.dtype, device=x.device)
    expert_mask = F.one_hot(selected, num_classes=len(experts)).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current = x[None, top_x].reshape(-1, hidden_dim)
        h = experts[expert_idx](current) * weights[top_x, idx, None]
        out.index_add_(0, top_x, h.to(x.dtype))
    return out.reshape(batch_size, seq_len, hidden_dim), router_logits


def make_routing_recompute_stable(model: nn.Module) -> int:
    """Patch supported MoE blocks in place; returns how many were patched."""
    ok, other = routed_moe_blocks(model)
    for _, block, attr in ok:
        block._bhaskera_experts_attr = attr
        block.forward = types.MethodType(_stable_forward, block)
    if other:
        logger.warning(
            f"{len(other)} MoE block(s) (e.g. {other[0][0]!r}) have a routing layout "
            "this module does not reproduce (no nn.Linear `gate`, no `top_k`, or shared "
            "experts); their recompute may hit CheckpointError if top-k flips."
        )
    return len(ok)
