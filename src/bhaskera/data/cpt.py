"""
bhaskera.data.cpt
=================
Continual pre-training (CPT) row layout: concatenate documents and cut the
token stream into full seq_len rows.

Why not the SFT layout: SFT tokenizes each example into its own row and
pads it to seq_len. For pre-training corpora that wastes most of the compute
on padding (web/book documents vary from a few tokens to far beyond
seq_len) and truncates long documents. CPT instead treats the corpus as one
token stream -- each document followed by EOS so the model sees where it
ends -- and cuts it into rows of exactly seq_len tokens. Every position is
a real token and a training target, so labels == input_ids and the
attention mask is all ones.

Attention may cross an EOS into the next document within a row. That is the
standard pre-training setup (GPT-2/3, Llama); the EOS is the boundary signal.

Packing runs per Ray Data batch (map_batches is stateless across calls), so
each batch's final partial row is dropped. With data.tokenize_batch_size
documents per batch the loss is < 1 row per batch; pack() reports it.
"""
from __future__ import annotations

import numpy as np


def pack(
    docs: list[list[int]], seq_len: int, eos_id: int | None
) -> tuple[np.ndarray, int]:
    """Concatenate token lists (EOS after each) and cut into full rows.

    Returns (rows [n, seq_len] int64, dropped_tokens). Rows are filled in
    document order; tokens that do not complete a final row are dropped.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    parts = []
    for ids in docs:
        if len(ids) == 0:
            continue
        parts.append(np.asarray(ids, dtype=np.int64))
        if eos_id is not None and ids[-1] != eos_id:
            parts.append(np.array([eos_id], dtype=np.int64))
    if not parts:
        return np.zeros((0, seq_len), dtype=np.int64), 0
    stream = np.concatenate(parts)
    n_rows = len(stream) // seq_len
    kept = n_rows * seq_len
    return stream[:kept].reshape(n_rows, seq_len), int(len(stream) - kept)


def cpt_batch(
    docs: list[list[int]], seq_len: int, eos_id: int | None
) -> tuple[dict, int]:
    """pack() as the tokenized-cache columns training expects, plus the
    dropped-token count (kept out of the columns: Ray Data needs every
    column to have one entry per row)."""
    rows, dropped = pack(docs, seq_len, eos_id)
    cols = {
        "input_ids": rows,
        "attention_mask": np.ones_like(rows),
        "labels": rows.copy(),
    }
    return cols, dropped
