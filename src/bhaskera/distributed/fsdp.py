"""
bhaskera.distributed.fsdp
=========================
FSDP2 (torch.distributed._composable.fsdp.fully_shard) wrap + mixed precision.

Design notes:
  * We call fully_shard three times (experts → layers → root) so that the
    MoE forward only all-gathers the activated experts, not the full set.
  * MixedPrecisionPolicy is the SINGLE source of mixed-precision truth.
    The training loop no longer wraps forward in torch.autocast.
  * "auto" dtype resolution reads from the ModelProfile produced by
    introspect.py, so no layer-name hardcoding is required.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from bhaskera.introspect import ModelProfile
from .activation_ckpt import apply_activation_checkpointing

logger = logging.getLogger(__name__)

# Bucket size for the rank-0 weight broadcast in wrap_fsdp2(). Caps the extra
# GPU memory that broadcast costs (peak = one bucket, not the model) while
# keeping the collective count sane for MoE checkpoints with tens of
# thousands of small per-expert tensors.
_BROADCAST_BUCKET_BYTES = 256 * 1024 * 1024

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}


def wrap_fsdp2(
    model: nn.Module,
    cfg,
    local_rank: int,
    profile: ModelProfile,
) -> nn.Module:
    try:
        from torch.distributed._composable.fsdp import (
            fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy, OffloadPolicy,
        )
    except ImportError as e:
        raise ImportError(
            "FSDP2 requires PyTorch >= 2.4. "
            "Either upgrade PyTorch or set training.distributed.strategy: ddp."
        ) from e

    fsdp_cfg = cfg.training.distributed.fsdp
    mp_policy = _build_mp_policy(fsdp_cfg, profile, MixedPrecisionPolicy)
    decoder_cls = _resolve_decoder_cls(model, fsdp_cfg, profile)

    # Snapshot rank 0's real weights BEFORE any sharding/AC transformation
    # touches the module tree, so these FQNs match the model's natural
    # (pre-FSDP) parameter names. model.state_dict() detaches views, it
    # does not clone storage, so this costs no extra host RAM on rank 0 —
    # it's the same tensors `model` already holds, just referenced from a
    # second dict so they survive fully_shard() reassigning `model`'s own
    # attributes to sharded DTensor storage. Every other rank's local
    # model is a `meta`-device build (see models/loader.py) with zero real
    # data — this is what actually fixes the redundant-per-worker-CPU-load
    # problem behind the MoE host-RAM OOMs: at most rank 0 ever holds a
    # full real copy.
    rank = dist.get_rank()
    _pretrained_state_dict = model.state_dict() if rank == 0 else None

    # None for FULL_SHARD (fully_shard's default 1D mesh over the whole
    # world); a 2D (replicate, shard) mesh for HYBRID_SHARD. Passed to every
    # fully_shard() below -- mixing meshes across units is not allowed.
    mesh = _build_device_mesh(fsdp_cfg)

    # fsdp_cfg.cpu_offload: offloads sharded params/grads/optimizer state to
    # host RAM between uses, trading step time for a lower GPU memory floor.
    # Applied uniformly across all three fully_shard() calls below — mixing
    # offload and non-offload units in one model is allowed by the API but
    # not something this config exposes a reason to do.
    offload_policy = CPUOffloadPolicy() if fsdp_cfg.cpu_offload else OffloadPolicy()

    # Step 1 — per-expert sharding (MoE only).
    if profile.is_moe and fsdp_cfg.shard_experts_individually and profile.expert_modules:
        for expert in profile.expert_modules:
            fully_shard(expert, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)
        logger.info(
            "FSDP2: per-expert sharding applied to "
            f"{len(profile.expert_modules)} expert modules "
            f"(class={profile.expert_module_cls.__name__ if profile.expert_module_cls else 'N/A'})"
        )

    # Step 2 — per-decoder-layer sharding.
    #
    # What each layer keeps between its forward and its backward decides how
    # expensive the backward all-gather is (measured, Llama-3.3-70B on 128 nodes:
    # 80% of every step is weight all-gathers, half of them in backward):
    #   True   -> back to a 1/world shard; backward re-gathers across all ranks.
    #   int K  -> keep a 1/K shard; backward gathers inside a block of K ranks,
    #             a much shorter ring (fsdp.reshard_after_forward_group).
    #   False  -> keep the whole layer; no backward gather at all. Used for the
    #             last fsdp.keep_gathered_layers layers, which backward needs first.
    rank_world = dist.get_world_size()
    group = int(getattr(fsdp_cfg, "reshard_after_forward_group", 0) or 0)
    if group and not (1 < group < rank_world and rank_world % group == 0):
        logger.warning(
            f"fsdp.reshard_after_forward_group={group} must divide world size "
            f"{rank_world} and lie strictly between 1 and it; using full resharding"
        )
        group = 0
    if group and mesh is not None:
        logger.warning(
            "fsdp.reshard_after_forward_group has no effect with HYBRID_SHARD "
            "(the shard group is already the small group); ignoring it"
        )
        group = 0
    keep_last = max(0, int(getattr(fsdp_cfg, "keep_gathered_layers", 0) or 0))
    decoder_layers = (
        [m for m in model.modules() if isinstance(m, decoder_cls)] if decoder_cls is not None else []
    )
    layers: list[nn.Module] = []
    if decoder_cls is not None:
        for i, module in enumerate(decoder_layers):
            if i >= len(decoder_layers) - keep_last:
                raf = False
            else:
                raf = group if group else True
            fully_shard(module, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy,
                        reshard_after_forward=raf)
            layers.append(module)
        if group or keep_last:
            logger.info(
                f"FSDP2: backward all-gather over blocks of {group or rank_world} ranks; "
                f"last {min(keep_last, len(layers))} layers kept gathered between forward and backward"
            )
        logger.info(
            f"FSDP2: per-layer sharding applied to {len(layers)} "
            f"{decoder_cls.__name__} layers"
        )
        _apply_prefetch_hints(layers, fsdp_cfg)
    else:
        logger.warning(
            "No decoder layer class found — applying fully_shard to root only. "
            "Memory use will be suboptimal."
        )

    # Step 3 — root. reshard_after_forward defaults False: root params are
    # needed again immediately at the start of backward, so resharding them
    # after forward just to re-gather microseconds later in backward is
    # pure wasted communication.
    fully_shard(
        model, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy,
        reshard_after_forward=fsdp_cfg.root_reshard_after_forward,
    )

    # Activation checkpointing.
    if fsdp_cfg.activation_checkpointing:
        apply_activation_checkpointing(model, profile, decoder_cls)

    # Fill in every rank's shard with rank 0's real values. Composable
    # fully_shard()/checkpoint() preserve the module hierarchy (unlike
    # FSDP1's FlatParameter), so the FQNs in `_pretrained_state_dict`
    # (captured pre-shard) still match this now-sharded model exactly.
    # This is a collective call -- every rank must reach it.
    _broadcast_pretrained_weights(
        model, _pretrained_state_dict, torch.device(f"cuda:{local_rank}"),
        ckpt_dir=cfg.model.name if getattr(fsdp_cfg, "sharded_load", True) else None,
    )

    logger.info(
        f"FSDP2 wrap complete | param_dtype={mp_policy.param_dtype} | "
        f"reduce_dtype={mp_policy.reduce_dtype} | moe={profile.is_moe} | "
        f"ac={fsdp_cfg.activation_checkpointing} | cpu_offload={fsdp_cfg.cpu_offload} | "
        f"mesh={'1D world (FULL_SHARD)' if mesh is None else tuple(mesh.shape)}"
    )
    return model


def _checkpoint_key(name: str) -> str:
    """Model FQN -> safetensors key: drop the PEFT and activation-checkpoint wrappers."""
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    for seg in (".base_layer.", "._checkpoint_wrapped_module.", "._fsdp_wrapped_module."):
        name = name.replace(seg, ".")
    return name


def _sharded_load_plan(named: list, ckpt_dir: Optional[str]) -> dict:
    """
    Map tensor FQN -> (safetensors file, key) for every tensor that can be read
    straight from the checkpoint with a matching shape. Every rank computes the
    same plan from the same files, so the remaining broadcasts stay in lockstep.
    Returns {} (all-broadcast, the old behaviour) whenever the checkpoint is not
    a local safetensors directory.
    """
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return {}
    index = os.path.join(ckpt_dir, "model.safetensors.index.json")
    single = os.path.join(ckpt_dir, "model.safetensors")
    try:
        from safetensors import safe_open
        if os.path.exists(index):
            with open(index) as f:
                weight_map = json.load(f)["weight_map"]
        elif os.path.exists(single):
            with safe_open(single, framework="pt") as f:
                weight_map = {k: "model.safetensors" for k in f.keys()}
        else:
            return {}
        shapes: dict[str, tuple] = {}
        for fname in sorted(set(weight_map.values())):
            with safe_open(os.path.join(ckpt_dir, fname), framework="pt") as f:
                for k in f.keys():
                    shapes[k] = tuple(f.get_slice(k).get_shape())
    except Exception as e:  # noqa: BLE001 — same files on every rank, so same outcome
        logger.warning(f"sharded load disabled ({e}); falling back to rank-0 broadcast")
        return {}
    plan = {}
    for name, tensor in named:
        key = _checkpoint_key(name)
        if key in weight_map and shapes.get(key) == tuple(tensor.shape):
            plan[name] = (os.path.join(ckpt_dir, weight_map[key]), key)
    return plan


def _build_device_mesh(fsdp_cfg):
    """
    Resolve fsdp.sharding_strategy to the mesh fully_shard() should use.

    FSDP2 has no ShardingStrategy enum -- the strategy is the mesh. A 1D mesh
    is ZeRO-3 over every rank; a 2D (replicate, shard) mesh is HSDP: params
    are sharded along "shard" and replicated along "replicate", so per-layer
    all-gathers span only one shard group and the only cross-group collective
    is the gradient all-reduce. SHARD_GRAD_OP / NO_SHARD have no mesh form
    and stay no-ops (warned) as before.
    """
    import os

    strategy = (getattr(fsdp_cfg, "sharding_strategy", None) or "FULL_SHARD").upper()
    if strategy == "FULL_SHARD":
        return None
    if strategy != "HYBRID_SHARD":
        logger.warning(
            f"fsdp.sharding_strategy={strategy!r} has no FSDP2 equivalent and is "
            "ignored -- running FULL_SHARD. Supported: FULL_SHARD, HYBRID_SHARD."
        )
        return None

    world = dist.get_world_size()
    shard = int(getattr(fsdp_cfg, "hsdp_shard_size", 0) or 0)
    if shard <= 0:
        shard = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    if shard <= 0 or world % shard != 0:
        raise ValueError(
            f"HYBRID_SHARD: hsdp_shard_size={shard} must be > 0 and divide "
            f"world_size={world}."
        )
    if shard == world:
        logger.warning(
            f"HYBRID_SHARD with shard size == world_size ({world}) is FULL_SHARD; "
            "using the default 1D mesh."
        )
        return None

    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh(
        "cuda", (world // shard, shard), mesh_dim_names=("replicate", "shard")
    )
    _log_shard_group_placement(mesh)
    return mesh


def _log_shard_group_placement(mesh) -> None:
    """
    Log how many nodes each shard group spans. HSDP's benefit depends on it:
    a group inside one node keeps weight all-gathers off the network
    entirely; a group spanning nodes still pays InfiniBand for them. Rank
    ordering comes from Ray Train, so this is checked rather than assumed.
    Collective -- every rank must call it.
    """
    import socket

    hosts: list = [None] * dist.get_world_size()
    dist.all_gather_object(hosts, socket.gethostname())
    if dist.get_rank() != 0:
        return
    spans = [
        len({hosts[r] for r in group})
        for group in mesh.mesh.tolist()  # rows = shard groups
    ]
    logger.info(
        f"HSDP mesh: {mesh.shape[0]} replica groups x {mesh.shape[1]} GPUs/shard group; "
        f"nodes spanned per shard group: min={min(spans)} max={max(spans)}"
    )


def _broadcast_pretrained_weights(
    model: nn.Module, rank0_state_dict: Optional[dict], device: torch.device,
    ckpt_dir: Optional[str] = None,
) -> None:
    """
    Fill every rank's sharded (DTensor) parameters from rank 0's real values,
    one tensor at a time, with bounded GPU memory.

    Do NOT use torch.distributed.checkpoint's
    set_model_state_dict(..., broadcast_from_rank0=True) here. It is correct
    but its peak GPU cost is the size of the WHOLE unsharded model on EVERY
    rank, regardless of world_size, which OOMs any model bigger than one GPU
    (measured: Qwen3-30B-A3B, 60GB in bf16, OOM'd at 74.35GiB on an 80GB
    A100). The reason is in torch/distributed/_state_dict_utils.py:
    _broadcast_tensors() moves each full tensor to the GPU, then
    _distribute_tensors() takes the local shard as `full_tensor[slices]` --
    a VIEW, which pins the full tensor's entire storage -- and stores the
    resulting DTensor in a dict that accumulates across every key. So each
    parameter's full GPU tensor stays alive until the whole loop finishes.

    Here each full tensor is broadcast, the local shard is `.clone()`d so it
    owns its own storage, and the full tensor is dropped immediately. Peak
    GPU is therefore (this rank's shard) + (the single largest tensor),
    not the whole model.

    Sharded load (fsdp.sharded_load, default on): tensors present in the
    checkpoint's safetensors files are not broadcast at all -- every rank reads
    only its own slice straight from disk, in parallel. The broadcast made rank 0
    read the whole checkpoint alone and push it through one GPU: 600 s of wrap
    time for Llama-3.3-70B on 8 nodes. Only tensors with no checkpoint entry
    (LoRA adapters, non-persistent buffers such as RoPE inv_freq, tied weights)
    still come from rank 0.
    """
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    rank = dist.get_rank()
    sd = rank0_state_dict if rank0_state_dict is not None else {}

    # Ranks that built on `meta` have correctly-shaped shards with no real
    # storage. to_empty() materializes them through nn.Module._apply, which
    # is the sanctioned path: it keeps FSDP2's internal FSDPParam references
    # pointing at the same objects. Assigning fresh parameters instead leaks
    # the shards FSDP still holds (measured: a rank that built real ended at
    # 64.30GB GPU, exactly 2x its 30GB shard, while a meta-built rank sat at
    # 32.13GB) and desyncs FSDP's bookkeeping.
    if any(p.is_meta for p in model.parameters()):
        model.to_empty(device=device, recurse=True)

    # Every rank has the identical module structure, so this iteration order
    # and every shape/dtype below is the same on all ranks -- no metadata
    # exchange needed, and the broadcasts stay in lockstep.
    named = list(model.named_parameters()) + list(model.named_buffers())

    # One collective per tensor is unusable at MoE scale: Qwen3-30B-A3B has
    # 18,867 tensors and measured 478s of wrap time, almost all of it
    # per-collective latency on tiny payloads (each expert's weights are a
    # separate tensor). Bucket them instead -- peak extra GPU memory is the
    # bucket, not the model, and the collective count drops by ~3 orders of
    # magnitude.
    bucket_bytes = _BROADCAST_BUCKET_BYTES
    pg = dist.distributed_c10d._get_default_group()

    missing: list[str] = []
    bucket: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    bucket_size = 0

    def _flush(bucket):
        if not bucket:
            return
        dist._broadcast_coalesced(pg, [full for _, _, full in bucket], bucket_bytes, 0)
        # .detach() is required, not cosmetic: DTensor.to_local() hands back a
        # view produced by an autograd custom Function
        # (_ToTorchTensorBackward), and writing into that view raises "view is
        # being modified inplace ... forbidden". detach() shares the same
        # storage with no autograd edge, so the write lands on the real shard.
        with torch.no_grad():
            for _, tensor, full in bucket:
                if isinstance(tensor, DTensor):
                    shape, offset = compute_local_shape_and_global_offset(
                        full.shape, tensor.device_mesh, tensor.placements
                    )
                    slices = tuple(slice(o, o + s) for s, o in zip(shape, offset))
                    tensor.to_local().detach().copy_(full[slices])
                else:
                    tensor.detach().copy_(full)
        bucket.clear()

    plan = _sharded_load_plan(named, ckpt_dir)
    open_files: dict = {}
    if plan:
        from safetensors import safe_open

    for name, tensor in named:
        if name in plan:
            fname, key = plan[name]
            if fname not in open_files:
                open_files[fname] = safe_open(fname, framework="pt", device="cpu")
            if isinstance(tensor, DTensor):
                shape, offset = compute_local_shape_and_global_offset(
                    tensor.shape, tensor.device_mesh, tensor.placements
                )
                slices = tuple(slice(o, o + s) for s, o in zip(shape, offset))
                dst = tensor.to_local().detach()
            else:
                slices, dst = tuple(slice(None) for _ in tensor.shape), tensor.detach()
            if dst.numel():
                with torch.no_grad():
                    dst.copy_(open_files[fname].get_slice(key)[slices].to(device=device, dtype=dst.dtype))
            continue
        if rank == 0:
            src = sd.get(name)
            # Non-persistent buffers (e.g. rotary_emb.inv_freq) are absent
            # from state_dict(), but rank 0 built the model for real, so its
            # live value is correct. Broadcasting torch.empty() here instead
            # overwrote RoPE frequencies with garbage on every rank, including
            # rank 0, and FSDP runs started at loss ~9-15 vs ~1.4 under DDP.
            if src is None and not isinstance(tensor, DTensor) and not tensor.is_meta:
                src = tensor
            if src is None:
                missing.append(name)
                full = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
            else:
                full = src.detach().to(device=device, dtype=tensor.dtype, copy=True)
        else:
            full = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)

        bucket.append((name, tensor, full))
        bucket_size += full.numel() * full.element_size()
        if bucket_size >= bucket_bytes:
            _flush(bucket)
            bucket_size = 0

    _flush(bucket)
    open_files.clear()

    if rank == 0 and plan:
        logger.info(
            f"FSDP2 sharded load: {len(plan)} of {len(named)} tensors read per rank "
            f"from {ckpt_dir}; {len(named) - len(plan)} broadcast from rank 0"
        )
    if rank == 0 and missing:
        logger.warning(
            f"{len(missing)} model tensors had no entry in the pretrained "
            f"state dict and were left uninitialised, e.g. {missing[:5]}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_prefetch_hints(layers: list[nn.Module], fsdp_cfg) -> None:
    """
    Explicit forward/backward prefetch scheduling for a per-layer-wrapped
    transformer stack.

    Without this, FSDP2 falls back to implicit prefetch heuristics that
    often fail to overlap the next layer's all-gather with the current
    layer's compute in a plain per-layer wrap — every layer ends up
    blocking on its own gather instead. Each layer here is an FSDPModule
    (fully_shard() return value), which exposes set_modules_to_forward
    /backward_prefetch() directly.
    """
    if len(layers) < 2:
        return
    if fsdp_cfg.forward_prefetch:
        for i, layer in enumerate(layers[:-1]):
            layer.set_modules_to_forward_prefetch([layers[i + 1]])
    if fsdp_cfg.backward_prefetch:
        # Backward runs in reverse layer order, so each layer prefetches
        # the PREVIOUS layer's params.
        for i, layer in enumerate(layers[1:], start=1):
            layer.set_modules_to_backward_prefetch([layers[i - 1]])


def _build_mp_policy(fsdp_cfg, profile: ModelProfile, MixedPrecisionPolicy):
    if fsdp_cfg.param_dtype == "auto":
        param_dtype = profile.model_dtype
    else:
        param_dtype = _DTYPE_MAP.get(fsdp_cfg.param_dtype, torch.bfloat16)

    # reduce_dtype=float32 is safer for MoE because gradients are sparse
    # and low-precision reductions amplify routing noise.
    if fsdp_cfg.reduce_dtype == "auto":
        reduce_dtype = torch.float32 if profile.is_moe else param_dtype
    else:
        reduce_dtype = _DTYPE_MAP.get(fsdp_cfg.reduce_dtype, torch.bfloat16)

    if fsdp_cfg.buffer_dtype == "auto":
        output_dtype = param_dtype
    else:
        output_dtype = _DTYPE_MAP.get(fsdp_cfg.buffer_dtype, torch.bfloat16)

    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
    )


def _resolve_decoder_cls(model: nn.Module, fsdp_cfg, profile: ModelProfile) -> Optional[type]:
    """
    Priority:
        1. explicit names in fsdp_cfg.transformer_layer_cls (back-compat)
        2. auto-detected class from introspection profile
    """
    if fsdp_cfg.transformer_layer_cls:
        found = _find_layer_classes_by_name(model, fsdp_cfg.transformer_layer_cls)
        if found:
            return found[0]
        logger.warning(
            f"Manual transformer_layer_cls {fsdp_cfg.transformer_layer_cls} "
            "not found in model — falling back to auto-detection"
        )
    return profile.decoder_layer_cls


def _find_layer_classes_by_name(model: nn.Module, names: list[str]) -> list[type]:
    found: dict[str, type] = {}
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name in names and cls_name not in found:
            found[cls_name] = module.__class__
    missing = set(names) - set(found)
    if missing:
        logger.warning(f"FSDP2: layer classes not found in model: {missing}")
    return list(found.values())