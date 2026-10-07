"""
bhaskera.data.packing
=====================
Runtime sequence packing for pre-tokenized, padded caches (data.runtime_packing).

Why: the Nemotron chat cache at seq_len 4096 averages 2,338 real tokens per
row -- 43% of every row is padding that still costs attention and MLP
compute. Packing concatenates several examples into one seq_len row so that
compute goes to real tokens.

How:
  * Within each Ray Data batch, examples are stripped of padding (by
    attention_mask, so left or right padding both work) and placed into
    seq_len rows first-fit-decreasing. An example already seq_len long gets
    a row to itself.
  * Example boundaries are marked by position_ids restarting at 0. The label
    at each example's first token is -100, so the causal shift never trains a
    prediction from one example into the next. Leftover space is a separate,
    fully masked segment (ids 0, labels -100, positions from 0).
  * attention_mask is kept only to count real tokens (throughput/MFU); the
    model must NOT receive it. With no mask and batch size 1, transformers'
    flash-attention path switches to per-example varlen attention from the
    position_ids (modeling_flash_attention_utils._is_packed_sequence) -- for
    batch > 1 it silently does not, and packed examples would attend to each
    other. trainer/loop.py therefore flattens each packed micro-batch to
    [1, batch * seq_len]. Requires attn_impl: flash_attention_2.
"""
from __future__ import annotations

import numpy as np


def _rows(col) -> list[np.ndarray]:
    return [np.asarray(r) for r in col]


def pack_batch(batch: dict, seq_len: int) -> dict:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for ids, mask, lab in zip(
        _rows(batch["input_ids"]), _rows(batch["attention_mask"]), _rows(batch["labels"])
    ):
        keep = mask.astype(bool)
        ids, lab = ids[keep][:seq_len], lab[keep][:seq_len]
        if ids.size:
            segments.append((ids, lab))

    order = sorted(range(len(segments)), key=lambda i: -len(segments[i][0]))
    bins: list[list[int]] = []
    room: list[int] = []
    for i in order:
        n = len(segments[i][0])
        for b, r in enumerate(room):
            if n <= r:
                bins[b].append(i)
                room[b] -= n
                break
        else:
            bins.append([i])
            room.append(seq_len - n)

    rows = len(bins)
    input_ids = np.zeros((rows, seq_len), dtype=np.int64)
    labels = np.full((rows, seq_len), -100, dtype=np.int64)
    position_ids = np.zeros((rows, seq_len), dtype=np.int64)
    attention_mask = np.zeros((rows, seq_len), dtype=np.int64)
    for b, members in enumerate(bins):
        pos = 0
        for i in members:
            ids, lab = segments[i]
            n = len(ids)
            input_ids[b, pos:pos + n] = ids
            labels[b, pos:pos + n] = lab
            labels[b, pos] = -100
            position_ids[b, pos:pos + n] = np.arange(n)
            attention_mask[b, pos:pos + n] = 1
            pos += n
        if pos < seq_len:
            position_ids[b, pos:] = np.arange(seq_len - pos)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def pack_dataset(ds, cfg):
    """Apply pack_batch over a Ray Dataset (train only; eval stays unpacked)."""
    if cfg.model.attn_impl != "flash_attention_2":
        raise ValueError(
            "data.runtime_packing requires model.attn_impl: flash_attention_2 -- other "
            "attention paths would let packed examples attend to each other "
            f"(got {cfg.model.attn_impl!r})."
        )
    return ds.map_batches(
        pack_batch,
        batch_format="numpy",
        batch_size=int(cfg.data.packing_window_rows),
        fn_kwargs={"seq_len": int(cfg.data.seq_len)},
    )
