"""data.format_options.chunk: long documents become several seq_len rows instead of being cut. Needs Ray and numpy (installed on the cluster)."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("ray.data")
pytest.importorskip("torch")
from bhaskera.data.tokenize import TokenizerActor  # noqa: E402


class Tok:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, texts, truncation=False, padding=False):
        return {"input_ids": [[int(w) for w in t.split()] for t in texts]}


def actor(seq_len=4):
    a = object.__new__(TokenizerActor)
    a.tokenizer, a.seq_len = Tok(), seq_len
    return a


def test_a_long_document_is_cut_into_rows_and_ends_with_eos():
    out = actor(4)._chunked(["1 2 3 4 5 6 7 8 9"])           # 9 tokens + eos = 10 -> 4, 4, 2
    assert out["input_ids"].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 99, 0, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0]]
    assert out["labels"].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 99, -100, -100]]


def test_each_document_starts_a_new_row_and_empty_ones_are_skipped_safely():
    out = actor(4)._chunked(["1 2", "3"])
    assert out["input_ids"].tolist() == [[1, 2, 99, 0], [3, 99, 0, 0]]
    none = actor(4)._chunked([])
    assert none["labels"].min() == -100 and none["input_ids"].shape == (1, 4)
