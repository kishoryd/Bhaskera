"""
bhaskera.distributed.expert_parallel
====================================
Expert parallelism (EP) for routed-MoE blocks under FSDP2.

Why: under FSDP every rank all-gathers *every* expert of a layer to run it,
even though each token uses only top_k of them -- 4.83GB per layer for
Qwen3-235B-A22B (128 experts x 3 x 4096 x 1536 x 2B). EP instead keeps a
fixed 1/ep_size slice of the experts resident on each rank and moves the
*tokens* to them with two all-to-alls per MoE layer (dispatch + combine):
~T * top_k * hidden * 2B each way, 4-24x less traffic than the gather for
the MoE models benchmarked here.

Layout: ranks are split into consecutive EP groups of ep_size. EP rank p
owns global experts [p*L, (p+1)*L), L = num_experts / ep_size. Everything
that is not an expert (attention, router, norms, embeddings, LoRA adapters)
stays under FSDP exactly as before.

Scope, enforced at apply time rather than assumed:
  * Routed blocks shaped like HF's Qwen3Moe/Mixtral SparseMoeBlock: a Linear
    router at `gate`, an `nn.ModuleList` of experts, `top_k`, softmax top-k
    routing with renormalisation controlled by `norm_topk_prob` (absent ->
    always renormalise, the Mixtral behaviour). Blocks with shared experts
    (Qwen2Moe, DeepSeek) are rejected -- their routing differs.
  * Frozen experts only (LoRA with include_experts: false). Experts leave
    FSDP, so their gradients would need their own all-reduce across EP
    groups; with frozen experts there are none. Gradients still flow
    *through* the experts to earlier layers via the differentiable
    all-to-all below.
  * Experts are removed from the registered module tree, so they are absent
    from model.state_dict() and therefore from DCP checkpoints. Being frozen
    pretrained weights, a resume reloads them from the base model.
"""
from __future__ import annotations

import logging
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from bhaskera.introspect import ModelProfile

logger = logging.getLogger(__name__)

_EXPERT_ATTRS = ("experts", "local_experts", "routed_experts")
_BROADCAST_BUCKET_BYTES = 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# Differentiable all-to-all
# ---------------------------------------------------------------------------

class _AllToAll(torch.autograd.Function):
    """all_to_all_single whose backward is the transposed all-to-all."""

    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        ctx.out_splits, ctx.in_splits, ctx.group = out_splits, in_splits, group
        out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = grad_out.new_empty((sum(ctx.in_splits),) + tuple(grad_out.shape[1:]))
        dist.all_to_all_single(
            grad_in, grad_out.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group
        )
        return grad_in, None, None, None


def _inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv


# ---------------------------------------------------------------------------
# EP forward
# ---------------------------------------------------------------------------

def _ep_forward(self, hidden_states: torch.Tensor):
    """Drop-in for the block's forward; returns (output, router_logits)."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    ep = self._ep
    if self.training and ep["jitter_noise"] > 0:
        hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
            1.0 - ep["jitter_noise"], 1.0 + ep["jitter_noise"]
        )
    x = hidden_states.reshape(-1, hidden_dim)
    num_tokens = x.shape[0]
    top_k, num_experts, P, L = ep["top_k"], ep["num_experts"], ep["size"], ep["local"]

    # Routing -- identical arithmetic to the HF block it replaces.
    router_logits = self.gate(x)
    weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    weights, selected = torch.topk(weights, top_k, dim=-1)
    if ep["norm_topk_prob"]:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(x.dtype)

    # Token-major flattening: row i*top_k + j is (token i, its j-th expert).
    flat_experts = selected.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    token_of_row = torch.arange(num_tokens, device=x.device).repeat_interleave(top_k)
    dispatched = x[token_of_row[order]]  # sorted by global expert id

    counts = torch.bincount(flat_experts, minlength=num_experts)  # [E]
    recv_counts = torch.empty_like(counts)
    dist.all_to_all_single(recv_counts, counts, group=ep["group"])  # [P*L] from each src
    send_splits = counts.view(P, L).sum(1).tolist()
    recv_matrix = recv_counts.view(P, L)
    recv_splits = recv_matrix.sum(1).tolist()

    received = _AllToAll.apply(dispatched, recv_splits, send_splits, ep["group"])

    # Received rows are grouped by source rank, then by local expert. Regroup
    # by local expert, run each expert once, and restore the received order.
    local_of_row = torch.arange(L, device=x.device).repeat(P).repeat_interleave(
        recv_matrix.reshape(-1)
    )
    by_expert = torch.argsort(local_of_row, stable=True)
    chunks = torch.split(received[by_expert], recv_matrix.sum(0).tolist())
    outputs = [
        expert(chunk) if chunk.shape[0] else chunk
        for expert, chunk in zip(ep["experts"], chunks)
    ]
    expert_out = torch.cat(outputs)[_inverse_permutation(by_expert)]

    combined = _AllToAll.apply(expert_out, send_splits, recv_splits, ep["group"])
    per_choice = combined[_inverse_permutation(order)]  # back to token-major
    out = (per_choice * weights.reshape(-1, 1)).view(num_tokens, top_k, hidden_dim).sum(1)
    return out.view(batch_size, seq_len, hidden_dim), router_logits


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _find_moe_blocks(model: nn.Module) -> list[tuple[str, nn.Module, str]]:
    blocks = []
    for name, module in model.named_modules():
        for attr in _EXPERT_ATTRS:
            experts = module._modules.get(attr)
            if isinstance(experts, nn.ModuleList) and len(experts) > 0:
                blocks.append((name, module, attr))
                break
    return blocks


def _validate_block(name: str, block: nn.Module, ep_size: int, num_experts: int) -> None:
    if not isinstance(getattr(block, "gate", None), nn.Linear):
        raise NotImplementedError(
            f"Expert parallelism: MoE block {name!r} has no nn.Linear router at "
            "`gate`; only softmax-top-k routed blocks (Qwen3Moe/Mixtral-style) are supported."
        )
    if any(hasattr(block, a) for a in ("shared_expert", "shared_experts")):
        raise NotImplementedError(
            f"Expert parallelism: MoE block {name!r} has shared experts, whose "
            "routing this implementation does not reproduce."
        )
    if not hasattr(block, "top_k"):
        raise NotImplementedError(f"Expert parallelism: MoE block {name!r} has no `top_k`.")
    if num_experts % ep_size:
        raise ValueError(
            f"Expert parallelism: num_experts={num_experts} is not divisible by "
            f"expert_parallel_size={ep_size}."
        )


def _new_ep_group(ep_size: int):
    """Consecutive-rank EP groups; every rank must create every group."""
    world, rank = dist.get_world_size(), dist.get_rank()
    if world % ep_size:
        raise ValueError(
            f"expert_parallel_size={ep_size} must divide world_size={world}."
        )
    mine = None
    for start in range(0, world, ep_size):
        ranks = list(range(start, start + ep_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            mine = group
    return mine, rank % ep_size


def apply_expert_parallel(
    model: nn.Module,
    profile: ModelProfile,
    ep_size: int,
    rank0_state_dict: dict | None,
    device: torch.device,
) -> None:
    """
    Convert every routed MoE block to EP in place. Collective -- every rank
    must call it with the same model structure, and before fully_shard(), so
    that FSDP never sees the (now unregistered) experts.
    """
    blocks = _find_moe_blocks(model)
    if not blocks:
        raise RuntimeError("Expert parallelism requested but no MoE blocks were found.")
    for name, block, attr in blocks:
        _validate_block(name, block, ep_size, len(block._modules[attr]))
        experts = block._modules[attr]
        if any(p.requires_grad for p in experts.parameters()):
            raise NotImplementedError(
                f"Expert parallelism requires frozen experts, but {name!r}.{attr} has "
                "trainable parameters (set lora.include_experts: false)."
            )

    group, ep_rank = _new_ep_group(ep_size)
    dtype = profile.model_dtype
    rank = dist.get_rank()

    # Load owned experts: rank 0 broadcasts every expert tensor (bucketed);
    # each rank keeps only its own slice. Same traffic as the existing
    # full-model broadcast in wrap_fsdp2, which experts no longer take part in.
    plan: list[tuple[nn.Parameter | None, str, torch.Size]] = []
    local_by_block = []
    expert_param_total = 0
    for name, block, attr in blocks:
        experts = block._modules[attr]
        L = len(experts) // ep_size
        owned = range(ep_rank * L, (ep_rank + 1) * L)
        for e, expert in enumerate(experts):
            expert_param_total += sum(p.numel() for p in expert.parameters())
            if e in owned:
                expert.to_empty(device=device)
                expert.to(dtype)
            for pname, p in expert.named_parameters():
                fqn = f"{name}.{attr}.{e}.{pname}" if name else f"{attr}.{e}.{pname}"
                plan.append((p if e in owned else None, fqn, p.shape))
        local_by_block.append(experts[ep_rank * L:(ep_rank + 1) * L])

    pg = dist.distributed_c10d._get_default_group()
    bucket: list[tuple[nn.Parameter | None, torch.Tensor]] = []
    bucket_bytes = 0

    def _flush():
        if not bucket:
            return
        dist._broadcast_coalesced(pg, [buf for _, buf in bucket], _BROADCAST_BUCKET_BYTES, 0)
        with torch.no_grad():
            for dst, buf in bucket:
                if dst is not None:
                    dst.copy_(buf)
        bucket.clear()

    for dst, fqn, shape in plan:
        if rank == 0:
            src = rank0_state_dict[fqn]
            buf = src.detach().to(device=device, dtype=dtype)
        else:
            buf = torch.empty(shape, dtype=dtype, device=device)
        bucket.append((dst, buf))
        bucket_bytes += buf.numel() * buf.element_size()
        if bucket_bytes >= _BROADCAST_BUCKET_BYTES:
            _flush()
            bucket_bytes = 0
    _flush()

    # Unregister experts; keep owned ones as a plain list (invisible to FSDP,
    # optimizer and state_dict) and move non-owned ones to meta so profile
    # references cost no memory while still reporting param counts.
    for (name, block, attr), local in zip(blocks, local_by_block):
        experts = block._modules.pop(attr)
        num_experts = len(experts)
        for e, expert in enumerate(experts):
            if expert not in local:
                expert.to("meta")
            for p in expert.parameters():
                p.requires_grad_(False)
        block._ep = {
            "group": group,
            "size": ep_size,
            "local": num_experts // ep_size,
            "num_experts": num_experts,
            "top_k": int(block.top_k),
            "norm_topk_prob": bool(getattr(block, "norm_topk_prob", True)),
            "jitter_noise": float(getattr(block, "jitter_noise", 0.0) or 0.0),
            "experts": list(local),
        }
        block.forward = types.MethodType(_ep_forward, block)

    # Experts are no longer model.parameters(); the MFU meter adds this back.
    model._bhaskera_ep_expert_params = expert_param_total
    if rank == 0:
        L = blocks[0][1]._ep["local"]
        logger.info(
            f"Expert parallelism: {len(blocks)} MoE blocks, ep_size={ep_size}, "
            f"{L} of {blocks[0][1]._ep['num_experts']} experts/layer resident per rank, "
            f"{dist.get_world_size() // ep_size} EP group(s)"
        )
