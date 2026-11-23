"""CPU tests for data/packing.py:  pytest src/bhaskera/data/tests/test_packing.py"""
from __future__ import annotations

from itertools import pairwise

import numpy as np

from bhaskera.data.packing import pack_batch

SEQ = 16


def _padded_batch(lengths, left_pad=False, seed=0):
    rng = np.random.default_rng(seed)
    ids, mask, lab, originals = [], [], [], []
    for n in lengths:
        tok = rng.integers(1, 1000, size=n)
        lbl = tok.copy()
        lbl[: n // 3] = -100                      # a masked "prompt" part
        pad = SEQ - n
        if left_pad:
            ids.append(np.concatenate([np.zeros(pad, int), tok]))
            mask.append(np.concatenate([np.zeros(pad, int), np.ones(n, int)]))
            lab.append(np.concatenate([np.full(pad, -100), lbl]))
        else:
            ids.append(np.concatenate([tok, np.zeros(pad, int)]))
            mask.append(np.concatenate([np.ones(n, int), np.zeros(pad, int)]))
            lab.append(np.concatenate([lbl, np.full(pad, -100)]))
        originals.append((tok, lbl))
    obj = lambda rows: np.array(rows + [None], dtype=object)[:-1]  # object column, like Ray
    return {"input_ids": obj(ids), "attention_mask": obj(mask), "labels": obj(lab)}, originals


def _segments(out):
    """Recover (ids, labels) per packed example from position_ids restarts."""
    segs = []
    for ids, lab, pos, m in zip(out["input_ids"], out["labels"], out["position_ids"], out["attention_mask"]):
        starts = [i for i in range(SEQ) if pos[i] == 0] + [SEQ]
        for a, b in pairwise(starts):
            if m[a:b].all():
                segs.append((ids[a:b], lab[a:b]))
            else:
                assert not m[a:b].any() and (lab[a:b] == -100).all(), "tail must be fully masked"
    return segs


def _check(lengths, left_pad=False):
    batch, originals = _padded_batch(lengths, left_pad)
    out = pack_batch(batch, SEQ)
    assert out["input_ids"].shape[1] == SEQ
    segs = _segments(out)
    assert len(segs) == len(originals)
    got = sorted(tuple(s[0]) for s in segs)
    want = sorted(tuple(o[0]) for o in originals)
    assert got == want, "every real token preserved exactly once"
    for ids, lab in segs:
        _, lbl = next(o for o in originals if tuple(o[0]) == tuple(ids))
        assert lab[0] == -100, "first label of each example masked"
        assert (lab[1:] == lbl[1:]).all(), "other labels unchanged"
    assert out["attention_mask"].sum() == sum(lengths)
    return out


def test_right_padded():
    out = _check([5, 7, 3, 9, 16, 2, 11])
    assert out["input_ids"].shape[0] < 7  # fewer rows than examples


def test_left_padded():
    _check([4, 12, 6, 1], left_pad=True)


def test_full_length_rows_stay_alone():
    out = _check([16, 16, 3])
    assert out["input_ids"].shape[0] == 3
