"""A few readable evaluation examples for MLflow.

During evaluation the model is run on the validation data with the correct previous tokens as input
(teacher forcing). For the first rows of the evaluation this module keeps the model's most likely next
token at each answer position, then writes them out as text next to the reference answer, so a person can
see what the model gets right and where it goes wrong. It adds no communication between ranks.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

IGNORE = -100
_MAX_CHARS = 600


def capture_rows(input_ids: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor, count: int) -> list[dict]:
    """The first ``count`` rows of a batch as plain lists: the ids, the labels, and the model's most likely token
    for every position that has a label (position t is predicted from the tokens before it)."""
    rows = []
    for i in range(min(count, input_ids.shape[0])):
        predicted = logits[i, :-1].argmax(-1).cpu()
        rows.append({"input_ids": input_ids[i].cpu().tolist(), "labels": labels[i].cpu().tolist(),
                     "predicted": [IGNORE] + predicted.tolist()})        # aligned with labels
    return rows


def _first_answer(labels: list[int]) -> tuple[int, int]:
    """Start and end (exclusive) of the first run of labelled tokens, or (0, 0) when there is none."""
    start = next((i for i, v in enumerate(labels) if v != IGNORE), None)
    if start is None:
        return 0, 0
    end = start
    while end < len(labels) and labels[end] != IGNORE:
        end += 1
    return start, end


def _text(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode([i for i in ids if i >= 0], skip_special_tokens=False)


def render_rows(rows: list[dict], tokenizer: Any) -> list[dict]:
    """Text for each captured row: the prompt, the reference answer, the model's prediction and the share of
    answer tokens it got right."""
    out = []
    for r in rows:
        start, end = _first_answer(r["labels"])
        start = max(start, 1)              # the very first token has nothing before it to be predicted from
        if end <= start:
            continue
        ref, pred = r["labels"][start:end], r["predicted"][start:end]
        right = sum(1 for a, b in zip(ref, pred) if a == b)
        out.append({
            "prompt": _text(tokenizer, r["input_ids"][:start])[-_MAX_CHARS:],
            "reference": _text(tokenizer, ref)[:_MAX_CHARS],
            "prediction": _text(tokenizer, pred)[:_MAX_CHARS],
            "match": round(right / max(1, len(ref)), 4),
        })
    return out


_TOKENIZERS: dict[str, Any] = {}


def load_tokenizer(cfg) -> Optional[Any]:
    """The tokenizer of the model being trained (cached), or None when it cannot be loaded."""
    name = cfg.model.name
    if name not in _TOKENIZERS:
        try:
            from transformers import AutoTokenizer
            _TOKENIZERS[name] = AutoTokenizer.from_pretrained(
                name, trust_remote_code=bool(getattr(cfg.model, "trust_remote_code", False)))
        except Exception as e:
            logger.warning(f"evaluation samples skipped: cannot load the tokenizer of {name!r}: {e}")
            _TOKENIZERS[name] = None
    return _TOKENIZERS[name]
