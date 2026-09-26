"""
bhaskera.distributed.activation_ckpt
====================================
Activation checkpointing wrappers, FSDP2-composable with legacy fallback.

Strategy: checkpoint each decoder layer, for dense and MoE alike. Expert-only
checkpointing is a fallback for when no decoder layer class was found.

Why not per-expert for MoE: checkpointing only the expert MLPs leaves
everything else in the layer un-checkpointed -- attention Q/K/V/output, LoRA
adapter inputs, the router, and the top-k token dispatch (each token copied
to k experts, so k x hidden_states per layer). On Qwen3-235B-A22B (94 layers,
hidden 4096, top-8, seq 4096) that is roughly 0.7-1GB of saved activations
per layer, ~65-90GB total per GPU -- a per-GPU cost that does not shrink with
more GPUs, which is why that model OOM'd at ~78GB identically at 16, 32, 128
and 256 GPUs. Checkpointing the whole decoder layer keeps only its input
(~33MB/layer there) and recomputes the rest, matching what HF's own gradient
checkpointing does for Qwen3MoE/Mixtral.

Recomputing a whole MoE layer requires the recompute to route tokens exactly
as the forward did, or torch aborts with CheckpointError; moe_routing.py pins
the top-k choice across the two (passed as the checkpoint context_fn).
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import torch.nn as nn

from bhaskera.introspect import ModelProfile

from .moe_routing import make_routing_recompute_stable, recompute_contexts

logger = logging.getLogger(__name__)


def apply_activation_checkpointing(
    model: nn.Module,
    profile: ModelProfile,
    decoder_cls: Optional[type],
) -> None:
    """Preferred: FSDP2-native composable API; fallback to legacy if missing."""
    # Composable API (torch >= 2.4).
    try:
        from torch.distributed._composable import checkpoint as composable_checkpoint
        _apply_composable(model, profile, decoder_cls, composable_checkpoint)
        return
    except ImportError:
        pass

    # Legacy path.
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing as _legacy_apply,
            checkpoint_wrapper,
            CheckpointImpl,
        )
        _apply_legacy(model, profile, decoder_cls, _legacy_apply, checkpoint_wrapper, CheckpointImpl)
    except ImportError:
        logger.warning("No activation checkpointing API available — skipping.")


def _moe_context_fn(model, profile):
    """For MoE, pin routing across forward/recompute (see moe_routing.py)."""
    if not profile.is_moe:
        return None
    patched = make_routing_recompute_stable(model)
    logger.info(f"AC: routing pinned across recompute for {patched} MoE blocks")
    return recompute_contexts


def _apply_composable(model, profile, decoder_cls, composable_checkpoint):
    if decoder_cls is not None:
        context_fn = _moe_context_fn(model, profile)
        kwargs = {"context_fn": context_fn} if context_fn else {}
        count = 0
        for module in model.modules():
            if isinstance(module, decoder_cls):
                composable_checkpoint(module, **kwargs)
                count += 1
        logger.info(f"AC (composable): applied to {count} {decoder_cls.__name__} layers")
        return

    if profile.is_moe and profile.expert_modules:
        for expert in profile.expert_modules:
            composable_checkpoint(expert)
        logger.warning(
            f"AC (composable): no decoder layer class found; falling back to "
            f"{len(profile.expert_modules)} expert modules. Attention, router and "
            f"dispatch activations are NOT checkpointed in this mode."
        )
        return

    logger.warning("AC: no target modules found — skipping")


def _apply_legacy(model, profile, decoder_cls, apply_fn, checkpoint_wrapper, CheckpointImpl):
    wrapper_fn = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    if decoder_cls is not None:
        context_fn = _moe_context_fn(model, profile)
        if context_fn:
            wrapper_fn = partial(wrapper_fn, context_fn=context_fn)
        check_fn = lambda m: isinstance(m, decoder_cls)
        label = decoder_cls.__name__
    elif profile.is_moe and profile.expert_module_cls:
        check_fn = lambda m: isinstance(m, profile.expert_module_cls)
        label = profile.expert_module_cls.__name__
    else:
        logger.warning("AC (legacy): no target classes — skipping")
        return

    apply_fn(model, checkpoint_wrapper_fn=wrapper_fn, check_fn=check_fn)
    logger.info(f"AC (legacy): applied to {label}")