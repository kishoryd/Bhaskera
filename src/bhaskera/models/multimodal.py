"""
bhaskera.models.multimodal
==========================
Helpers for models that read more than text (Qwen3-VL for images and video,
Qwen2.5-Omni for audio as well): telling them apart from text-only models,
finding every *encoder* (the vision tower, the audio tower), and the settings
checks that must fail before any GPU is allocated.

An encoder turns one modality into embeddings that the language model reads next
to the text. Each one is a tower of identical transformer blocks plus a few small
layers (patch embedding, merger). ExaAI treats them all the same way:

  * every encoder block is its own FSDP shard unit (distributed/fsdp.py),
  * encoders are frozen by default and are never LoRA targets by default,
  * the data path has one processor plugin per modality (data/).

So adding a modality means adding one row to `ENCODER_SPECS` and one data plugin,
not new code in the trainer or the sharding.

A text-only model is built from `AutoModelForCausalLM`, an image-text model (a
`vision_config` in its Hugging Face config) from `AutoModelForImageTextToText`, an Omni
model from its thinker class (see `model_class_and_config`).
Everything else in ExaAI reads the ModelProfile, which carries `encoders`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)

# modality -> (attribute of the HF config that marks it, names the tower goes by in the module tree)
ENCODER_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "vision": ("vision_config", ("visual", "vision_tower", "vision_model")),
    "audio":  ("audio_config",  ("audio_tower", "audio_encoder", "audio_model")),
}
MODALITIES = tuple(ENCODER_SPECS)               # encoder modalities a model can have
DATA_MODALITIES = ("text", "image", "video", "audio")   # what a data row can carry
_BLOCK_LIST_NAMES = ("blocks", "layers")


@dataclass
class Encoder:
    """One encoder tower of a multimodal model."""
    modality: str                                   # a key of ENCODER_SPECS
    tower: nn.Module
    block_cls: Optional[type] = None                # class of the tower's repeated blocks
    blocks: list[nn.Module] = field(default_factory=list)


# ── config helpers ─────────────────────────────────────────────────────

def _root_config(config):
    """Qwen2.5/3-Omni keep everything under thinker_config (the talker makes speech, which
    ExaAI does not train); other models are their own root."""
    return getattr(config, "thinker_config", None) or config


def text_config(config):
    """The text sub-config of a multimodal model, or the config itself for a text model."""
    root = _root_config(config)
    sub = getattr(root, "text_config", None)
    return sub if sub is not None else root


def config_modalities(config) -> list[str]:
    """Encoder modalities this config declares, e.g. ["vision"] or ["vision", "audio"]."""
    root = _root_config(config)
    return [m for m, (attr, _) in ENCODER_SPECS.items() if getattr(root, attr, None) is not None]


def is_multimodal_config(config) -> bool:
    return bool(config_modalities(config))


_KIND_ALIASES = {"image_text": "multimodal"}


def normalize_kind(kind: str) -> str:
    kind = _KIND_ALIASES.get((kind or "auto").lower(), (kind or "auto").lower())
    if kind not in ("auto", "causal_lm", "multimodal"):
        raise ValueError(f"model.kind must be auto, causal_lm or multimodal, got {kind!r}")
    return kind


def resolve_kind(kind: str, model_config) -> str:
    """`auto` -> multimodal when the config declares an encoder, else causal_lm."""
    kind = normalize_kind(kind)
    if kind == "auto":
        return "multimodal" if is_multimodal_config(model_config) else "causal_lm"
    return kind


# Omni models: ExaAI trains the thinker (reads text, images, video, audio; writes text).
# The talker / token2wav (speech output) are not loaded.
_THINKER_CLASSES = {
    "qwen2_5_omni":   "Qwen2_5OmniThinkerForConditionalGeneration",
    "qwen3_omni_moe": "Qwen3OmniMoeThinkerForConditionalGeneration",
}


def model_class_and_config(kind: str, model_config):
    """(model class, config to build it with) for the resolved kind.

    * text: AutoModelForCausalLM, config unchanged
    * Omni (thinker_config): the thinker class with config.thinker_config
    * image(-video) models: AutoModelForImageTextToText
    * any other multimodal model (audio-only and the like): the class named in
      config.architectures, when transformers has it
    """
    if kind != "multimodal":
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM, model_config
    import transformers

    model_type = getattr(model_config, "model_type", "")
    if getattr(model_config, "thinker_config", None) is not None:
        name = _THINKER_CLASSES.get(model_type)
        if name is None or not hasattr(transformers, name):
            raise NotImplementedError(f"No thinker class known for {model_type!r}; register a loader "
                                      "with @register_model")
        return getattr(transformers, name), model_config.thinker_config
    if "vision" in config_modalities(model_config):
        return auto_model_class(kind, model_config), model_config
    for arch in getattr(model_config, "architectures", None) or []:
        if hasattr(transformers, arch):
            return getattr(transformers, arch), model_config
    raise NotImplementedError(f"No model class found for the {'+'.join(config_modalities(model_config))} "
                              f"model {model_type!r}; register a loader with @register_model")


def auto_model_class(kind: str, model_config=None):
    """Hugging Face Auto class for text (AutoModelForCausalLM) or image-text models."""
    if kind == "multimodal":
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        except ImportError:                       # transformers < 4.45
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM


# ── finding encoders in the module tree ────────────────────────────────

def _find_tower(model: nn.Module, names: tuple[str, ...]) -> Optional[nn.Module]:
    """The tower, found by name so DDP / FSDP / PEFT wrappers do not matter."""
    for name, mod in model.named_modules():
        if name.split(".")[-1] in names and any(True for _ in mod.parameters()):
            return mod
    return None


def _find_blocks(tower: nn.Module) -> tuple[Optional[type], list[nn.Module]]:
    """Largest same-class list of blocks inside the tower (its transformer layers)."""
    best: list[nn.Module] = []
    for name, mod in tower.named_modules():
        if isinstance(mod, nn.ModuleList) and name.split(".")[-1] in _BLOCK_LIST_NAMES:
            kids = list(mod.children())
            if len(kids) >= 1 and all(type(k) is type(kids[0]) for k in kids) and len(kids) > len(best):
                best = kids
    return (type(best[0]) if best else None), best


def find_encoders(model: nn.Module, config=None) -> list[Encoder]:
    """Every encoder of the model. With a config, only the modalities it declares are looked for
    (and a missing tower is reported); without one, all known modalities are tried."""
    wanted = config_modalities(config) if config is not None else list(MODALITIES)
    found: list[Encoder] = []
    for modality in wanted:
        tower = _find_tower(model, ENCODER_SPECS[modality][1])
        if tower is None:
            if config is not None:
                logger.warning(f"Config declares a {modality} encoder but no tower was found in the module tree")
            continue
        cls, blocks = _find_blocks(tower)
        found.append(Encoder(modality=modality, tower=tower, block_cls=cls, blocks=blocks))
    return found


def encoder_block_classes(encoders: Iterable[Encoder]) -> list[type]:
    """Distinct block classes across encoders, in order (one FSDP unit per instance of each)."""
    out: list[type] = []
    for e in encoders:
        if e.block_cls is not None and e.block_cls not in out:
            out.append(e.block_cls)
    return out


# ── model output hygiene ───────────────────────────────────────────────

def strip_undeclared_output_keys(model: nn.Module) -> nn.Module:
    """
    Make the model's forward return an output holding only the fields its class declares.

    Hugging Face's output-recording decorator adds collected tensors (e.g. `router_logits` when
    output_router_logits=True) to the returned ModelOutput as extra keys. For Qwen3-VL-MoE the
    output class has no such field, and FSDP2's post-forward hook flattens and rebuilds the output
    through the class constructor: `TypeError: ...__init__() got an unexpected keyword argument
    'router_logits'`. The router load-balancing loss is already inside the output as `aux_loss`,
    so dropping the raw logits loses nothing the trainer reads. A no-op for outputs without extras.
    """
    import dataclasses
    import functools

    if getattr(model, "_bhaskera_clean_output", False):
        return model
    orig = model.forward

    @functools.wraps(orig)
    def forward(*args, **kwargs):
        out = orig(*args, **kwargs)
        if dataclasses.is_dataclass(out):
            declared = {f.name for f in dataclasses.fields(out)}
            if set(out.keys()) - declared:
                out = type(out)(**{n: getattr(out, n, None) for n in declared})
        return out

    model.forward = forward
    model._bhaskera_clean_output = True
    return model


# ── freezing and counting ──────────────────────────────────────────────

def _selected(encoders: list[Encoder], modalities) -> list[Encoder]:
    if modalities is None or modalities == "all" or "all" in list(modalities):
        return encoders
    unknown = set(modalities) - set(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown encoder modality {sorted(unknown)}; known: {list(MODALITIES)}")
    return [e for e in encoders if e.modality in set(modalities)]


def encoder_param_counts(model: nn.Module) -> dict[str, tuple[int, int]]:
    """{modality: (total, trainable)} parameter counts of every encoder."""
    out = {}
    for e in find_encoders(model):
        params = list(e.tower.parameters())
        out[e.modality] = (sum(p.numel() for p in params),
                           sum(p.numel() for p in params if p.requires_grad))
    return out


def freeze_encoders(model: nn.Module, modalities=None) -> dict[str, int]:
    """requires_grad=False on the given encoders' parameters (all by default).
    Returns {modality: number of tensors changed}."""
    changed: dict[str, int] = {}
    for e in _selected(find_encoders(model), modalities):
        n = 0
        for p in e.tower.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                n += 1
        changed[e.modality] = n
    return changed


def encoder_exclude_regex(modalities: Iterable[str]) -> Optional[str]:
    """Regex (fully matched against a module path) selecting every module inside the given
    encoders, for PEFT's `exclude_modules`."""
    names = [n for m in modalities for n in ENCODER_SPECS[m][1]]
    return rf"(.*\.)?({'|'.join(names)})\..*" if names else None


# ── settings that cannot work with non-text data ───────────────────────

def check_multimodal_settings(cfg) -> None:
    """
    Fail early, with a plain message, on settings that cannot work with image, video or
    audio data. Called by the launcher on the driver and by every worker.
    """
    modalities = [str(m).lower() for m in (getattr(cfg.data, "modalities", None) or ["text"])]
    unknown = [m for m in modalities if m not in DATA_MODALITIES]
    if unknown:
        raise ValueError(f"data.modalities has unknown entries {unknown}; allowed: {list(DATA_MODALITIES)}")
    kind = normalize_kind(getattr(cfg.model, "kind", "auto"))
    non_text = [m for m in modalities if m != "text"]
    if not non_text and kind != "multimodal":
        return
    strategy = str(cfg.training.distributed.strategy).lower()
    problems = []
    if non_text and kind == "causal_lm":
        problems.append(f"data.modalities has {non_text} but model.kind is causal_lm")
    if strategy == "pipeline":
        problems.append("training.distributed.strategy: pipeline does not support multimodal models yet (use fsdp or ddp)")
    if getattr(cfg.data, "runtime_packing", False):
        problems.append("data.runtime_packing is text-only; set it to false for multimodal data")
    if str(getattr(cfg.model, "quantization", "none")).lower() != "none":
        problems.append("model.quantization (QLoRA) is not supported for multimodal models yet")
    if non_text and not getattr(cfg.data, "tokenized_path", None) and not getattr(cfg.data, "cache_dir", None):
        # live tokenisation works but re-decodes every image / audio file on every run
        logger.warning("multimodal data without data.tokenized_path decodes every media file at start-up; "
                       "run bhaskera-tokenize first")
    if problems:
        raise ValueError("Multimodal settings problem(s):\n  - " + "\n  - ".join(problems))
