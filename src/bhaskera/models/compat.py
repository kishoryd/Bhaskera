"""Compatibility shims for models whose custom code (trust_remote_code) targets another transformers version.

Only names that newer transformers releases removed are re-created, and only when they are missing, so
nothing changes for models that do not need them.
"""
from __future__ import annotations

from typing import Optional, TypedDict


def patch_transformers_for_remote_code() -> None:
    import transformers.utils as tu

    if not hasattr(tu, "LossKwargs"):
        # Removed in transformers 4.56; BharatGen Param-1-7B (written for 4.51) imports it.
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: Optional[int]

        tu.LossKwargs = LossKwargs
