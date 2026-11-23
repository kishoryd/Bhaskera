"""
bhaskera.models.loader
======================
Load an HF `AutoModelForCausalLM`, or `AutoModelForImageTextToText` for image-text models
(or a registered custom loader), run
introspection, and optionally attach LoRA.


"""
from __future__ import annotations

import logging
from typing import Callable, Tuple

import torch
from transformers import AutoConfig

from bhaskera.introspect import ModelProfile, introspect_model
from bhaskera.models.quantization import build_quantization_config, param_numel
from bhaskera.models.multimodal import (encoder_param_counts, find_encoders, freeze_encoders, model_class_and_config,
                                        resolve_kind, strip_undeclared_output_keys)

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}

_CUSTOM_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    """Register a custom non-HF model loader under `name`."""
    def _wrap(fn: Callable):
        _CUSTOM_REGISTRY[name] = fn
        return fn
    return _wrap


def _maybe_apply_liger_kernel(model: torch.nn.Module, cfg) -> bool:
    """
    Apply Liger Kernel's Triton-fused RMSNorm / RoPE / SwiGLU / CrossEntropy
    kernels to an already-instantiated model.

    This is the generalized integration path: `_apply_liger_kernel_to_instance`
    dispatches on `model.config.model_type`, so it transparently supports every
    architecture Liger ships (Llama, Mistral, Mixtral, Qwen2/2.5/3, Gemma 1/2/3,
    Phi3, Granite, Olmo2, and more), and is a safe no-op for anything else.
    Compatible with FSDP2, DDP, gradient checkpointing, and flash-attn.

    Returns True iff kernels were actually applied. Never raises — a missing
    package, unsupported architecture, or internal Liger error is downgraded
    to a warning so training continues on the vanilla HF implementation.
    """
    if not getattr(cfg.model, "use_liger_kernel", True):
        logger.info("Liger Kernel disabled via config (model.use_liger_kernel=false).")
        return False

    try:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
    except ImportError:
        logger.warning(
            "liger-kernel is not installed — falling back to the standard "
            "HuggingFace kernels. Install with `pip install liger-kernel` "
            "(or re-run setup.sh) to enable Triton-fused RMSNorm/RoPE/SwiGLU/"
            "CrossEntropy and unlock ~20% throughput / ~60% memory savings."
        )
        return False

    model_type = getattr(getattr(model, "config", None), "model_type", "") or "<unknown>"
    try:
        _apply_liger_kernel_to_instance(model=model)
    except Exception as e:
        # Liger raises for model types it doesn't patch. This is expected for
        # custom / research architectures and must NOT abort training.
        logger.warning(
            f"Liger Kernel could not patch model_type={model_type!r} "
            f"({type(e).__name__}: {e}). Continuing with standard HF kernels."
        )
        return False

    logger.info(f"Liger Kernel applied (model_type={model_type}).")
    return True


def _warn_if_dequantizing(name: str, model_config, rank: int) -> None:
    """
    Loud warning when a quantized checkpoint is about to be silently
    dequantized to bf16 at load time, which multiplies host RAM by ~4x.

    Found the hard way on gpt-oss-120b: its checkpoint is MXFP4 (4-bit,
    183GB on disk), but transformers logs
        "MXFP4 quantization requires Triton and kernels installed:
         CUDA requires Triton >= 3.4.0 ... we will default to
         dequantizing the model to bf16"
    and proceeds. 120B params in bf16 is ~240GB of ANONYMOUS host memory
    per loading worker -- unsurvivable on a 192GB node, and the cause of
    the worker kills previously filed as a generic MoE host-RAM OOM
    (measured: one RayTrainWorker at 159.13GB, 158.64GB of it anonymous,
    killed at checkpoint shard 8/15). It is not visible in mmap-based
    loads, where weights stay file-backed and cost almost nothing.
    """
    quant = getattr(model_config, "quantization_config", None)
    if not quant:
        return
    method = (
        quant.get("quant_method") if isinstance(quant, dict)
        else getattr(quant, "quant_method", None)
    )
    if str(method).lower() != "mxfp4":
        return

    try:
        import triton
        triton_ver = tuple(int(x) for x in triton.__version__.split(".")[:2])
    except Exception:
        triton_ver = (0, 0)
    try:
        import kernels  # noqa: F401
        have_kernels = True
    except ImportError:
        have_kernels = False

    if triton_ver >= (3, 4) and have_kernels:
        return  # stays quantized, nothing to warn about

    n_params = getattr(model_config, "num_parameters", None)
    est = f"~{n_params * 2 / 1e9:.0f}GB" if n_params else "roughly 2 bytes/param"
    if rank == 0:
        logger.warning(
            f"{name} is an MXFP4 checkpoint, but transformers will DEQUANTIZE "
            f"it to bf16 at load: triton={'.'.join(map(str, triton_ver))} "
            f"(needs >=3.4.0), kernels={'present' if have_kernels else 'MISSING'}. "
            f"That materializes the full model as anonymous host RAM ({est}), "
            "per loading worker, before FSDP shards anything. On a 192GB node "
            "this OOM-kills the worker partway through loading. Either install "
            "triton>=3.4.0 plus the `kernels` package so it stays 4-bit, or "
            "schedule this model onto the cluster's large-memory (768GB) nodes."
        )


def _build_empty_model(model_config, kwargs: dict, auto_cls=None):
    """Architecture only (meta device, no weights), with the same tied-embedding structure as rank 0's
    real model.

    Some models (custom-code ones such as Param2-17B, and small ones such as Qwen3-VL-2B) come out of
    `from_config` with separate input and output embeddings although the config ties them, so this rank
    would hold one more parameter than rank 0 (which loaded real weights and tied them). The ranks then
    disagree on the model structure and the weight broadcast at start-up never completes.
    """
    from accelerate import init_empty_weights
    if auto_cls is None:
        from transformers import AutoModelForCausalLM as auto_cls
    with init_empty_weights():
        if hasattr(auto_cls, "from_config"):
            model = auto_cls.from_config(model_config, **kwargs)
        else:   # concrete classes (the Omni thinker) have only _from_config, which rejects trust_remote_code
            model = auto_cls._from_config(model_config, **{k: v for k, v in kwargs.items() if k != "trust_remote_code"})
    # outside the context: inside it, accelerate re-wraps every assigned parameter, which breaks the tie.
    # tie_weights() follows the config, so an untied model stays untied.
    model.tie_weights()
    return model


def build_model(
    cfg, device: torch.device, rank: int = 0
) -> Tuple[torch.nn.Module, ModelProfile]:
    """
    Load model, introspect it, optionally apply LoRA.

    For FSDP2, callers should pass device=torch.device("cpu") — the FSDP
    wrap step will shard and migrate the params to the correct GPU.
    For DDP, pass the target CUDA device directly.

    `rank` (the GLOBAL torch.distributed rank, not local_rank) matters only
    under FSDP: every non-zero rank builds the model on the `meta` device
    (architecture only, zero real weights, zero host RAM) instead of
    redundantly loading a full real copy. Rank 0 still loads real weights
    exactly as before. distributed.fsdp.wrap_fsdp2() shards all ranks'
    (real or meta) local models identically, then broadcasts rank 0's real
    values into every other rank's shard via
    torch.distributed.checkpoint.state_dict.set_model_state_dict(...,
    broadcast_from_rank0=True).

    Why this matters: previously every GPU worker independently loaded a
    full model copy before FSDP sharded it. Measured on the real
    Qwen3-30B-A3B checkpoint (57GB, mmap'd via low_cpu_mem_usage=True):
    this saves ~6.5GB of host RSS per 2-GPU node (11.0GB -> 4.5GB) --
    real, but not large, because mmap'd weights are file-backed pages,
    not anonymous RAM, regardless of how many workers open the file.
    This is NOT the cause of the multi-hundred-GB MoE OOMs seen in
    production -- see _warn_if_dequantizing() below for that (a
    dequantized-to-bf16 MXFP4 checkpoint IS fully anonymous RAM, and
    that is what actually exhausts a node).
    """
    name = cfg.model.name
    trust_remote_code = getattr(cfg.model, "trust_remote_code", False)
    if trust_remote_code:
        from .compat import patch_transformers_for_remote_code
        patch_transformers_for_remote_code()
    strategy = cfg.training.distributed.strategy.lower()
    meta_build = strategy == "fsdp" and rank != 0

    raw_dtype = getattr(cfg.model, "dtype", "bfloat16")
    if raw_dtype == "auto":
        load_dtype: "str | torch.dtype" = "auto"
    else:
        load_dtype = _DTYPE_MAP.get(raw_dtype, torch.bfloat16)

    kwargs: dict = dict(
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        torch_dtype=load_dtype,
    )
    if cfg.model.attn_impl:
        kwargs["attn_implementation"] = cfg.model.attn_impl

    # ── Load model ──────────────────────────────────────────────────
    if name in _CUSTOM_REGISTRY:
        model = _CUSTOM_REGISTRY[name](cfg, device)
    else:
        model_config = AutoConfig.from_pretrained(
            name, trust_remote_code=trust_remote_code
        )
        _warn_if_dequantizing(name, model_config, rank)
        # Text-only models load through AutoModelForCausalLM (unchanged); image-text models
        # (Qwen3-VL and similar, which have a vision_config) through the image-text class.
        model_kind = resolve_kind(getattr(cfg.model, "kind", "auto"), model_config)
        auto_cls, build_config = model_class_and_config(model_kind, model_config)
        if rank == 0:
            logger.info(f"Model kind: {model_kind} ({auto_cls.__name__})")
        # NOTE: we deliberately do NOT set max_position_embeddings nor
        # output_router_logits here.  Truncation to seq_len is the
        # tokenizer's job, and router logits are requested per-forward
        # by the training loop when MoE aux loss is needed.
        if meta_build:
            # Must mirror from_pretrained's kwargs exactly (minus
            # low_cpu_mem_usage, meaningless with no checkpoint to read) —
            # any structural difference (e.g. attn_implementation choosing
            # a different attention module class) would make this rank's
            # module tree diverge from rank 0's, which fully_shard()'s
            # collectives assume never happens.
            meta_kwargs = {
                k: v for k, v in kwargs.items() if k != "low_cpu_mem_usage"
            }
            model = _build_empty_model(build_config, meta_kwargs, auto_cls)
        else:
            quant_cfg = build_quantization_config(cfg)   # raises under FSDP
            if quant_cfg is not None:
                # Quantize shard by shard straight onto this rank's GPU. Loading to
                # CPU first (the bf16 DDP path) would need the whole bf16 model in
                # host RAM per rank: ~141 GB for a 70B model, two ranks per node.
                kwargs["quantization_config"] = quant_cfg
                kwargs["device_map"] = {"": device}
            model = auto_cls.from_pretrained(
                name, config=build_config, **kwargs
            )
        if device.type != "cpu" and "quantization_config" not in kwargs:
            model = model.to(device)

    # Multimodal MoE outputs can carry keys their output class does not declare (router_logits),
    # which breaks FSDP2's post-forward hook; see strip_undeclared_output_keys.
    if name not in _CUSTOM_REGISTRY and resolve_kind(getattr(cfg.model, "kind", "auto"), model_config) == "multimodal":
        strip_undeclared_output_keys(model)

    # ── Introspect (never hardcode layer classes) ───────────────────
    profile = introspect_model(model)
    if raw_dtype != "auto":
        profile.model_dtype = _DTYPE_MAP.get(raw_dtype, torch.bfloat16)

    param_count = sum(param_numel(p) for p in model.parameters())
    logger.info(
        f"Loaded {name} | params={param_count:,} | "
        f"dtype={profile.model_dtype} | moe={profile.is_moe}"
    )

    # ── Encoders (multimodal models) ────────────────────────────────
    if profile.is_multimodal:
        to_freeze = getattr(cfg.model, "freeze_encoders", ["all"])
        freeze_encoders(model, to_freeze)
        for modality, (total, trainable) in encoder_param_counts(model).items():
            logger.info(f"Encoder {modality}: {total:,} parameters, "
                        f"{'frozen' if trainable == 0 else f'{trainable:,} trainable'}")

    # ── Liger Kernel patching ───────────────────────────────────────
    # Done BEFORE LoRA so PEFT wraps the already-fused modules (LoRA
    # adapters attach to Linear children, which Liger does not touch).
    # Done BEFORE FSDP wrap (handled later by distributed.wrap_model)
    # so that sharding sees the final module classes.
    _maybe_apply_liger_kernel(model, cfg)

    # ── LoRA ────────────────────────────────────────────────────────
    if cfg.lora.enabled:
        from .lora import apply_lora
        if meta_build:
            # New LoRA A/B parameters must also land on meta here, or this
            # rank would materialize real (random, per-rank-inconsistent)
            # adapter weights while rank 0 materializes its own — two
            # different random draws for what fully_shard() must treat as
            # one coherent parameter. Rank 0's actual values, LoRA included,
            # get broadcast into every rank's shard right after wrap_fsdp2()
            # (see distributed/fsdp.py), so what happens here is irrelevant
            # data-wise — it only needs to produce the right shapes/names.
            from accelerate import init_empty_weights
            with init_empty_weights():
                model = apply_lora(model, cfg, profile)
        else:
            model = apply_lora(model, cfg, profile)

        # get_peft_model freezes every parameter outside the adapters, including encoders that
        # model.freeze_encoders left trainable (freeze_encoders: [] would silently train nothing
        # but the LoRA weights). Give those encoders their gradients back.
        if profile.is_multimodal:
            frozen = set(getattr(cfg.model, "freeze_encoders", ["all"]) or [])
            if "all" not in frozen:
                for enc in find_encoders(model):
                    if enc.modality not in frozen:
                        for p in enc.tower.parameters():
                            p.requires_grad_(True)
                        logger.info(f"Encoder {enc.modality} trainable alongside LoRA "
                                    f"({sum(p.numel() for p in enc.tower.parameters()):,} parameters)")

    n_train = sum(param_numel(p) for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters after setup: {n_train:,}")
    return model, profile
