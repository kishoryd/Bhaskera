
from __future__ import annotations

import ray.data

from bhaskera.data.registry import register, register_raw
from bhaskera.data.tokenize import tokenize_dataset


_HF_SPLIT = {"train": "train_sft", "val": "test_sft"}


@register_raw("ultrachat", text_col="prompt")
def _build_raw(cfg, split: str | None = None) -> ray.data.Dataset:
    """Return the raw (un-tokenized) UltraChat dataset.

    split="train" -> HF "train_sft"; split="val" -> HF "test_sft" (the
    dataset's held-out set); split=None defaults to "train_sft" so existing
    callers that don't pass split keep working unchanged.
    """
    from datasets import load_dataset
    hf_split = _HF_SPLIT.get(split, "train_sft")
    hf_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=hf_split)
    return ray.data.from_huggingface(hf_ds)


@register("ultrachat")
def build(cfg, world_size: int = 1) -> ray.data.Dataset:
    """Return the tokenized UltraChat dataset, using cache if available."""
    return tokenize_dataset(_build_raw(cfg), cfg, "prompt", world_size=world_size)