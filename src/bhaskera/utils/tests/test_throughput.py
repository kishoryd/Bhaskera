"""MFU arithmetic for ThroughputTracker (CPU only, no GPU or model needed)."""
from __future__ import annotations

import pytest

from bhaskera.utils import throughput as tp


def _tracker(**kw):
    base = dict(params_for_flops=10_000_000_000, world_size=16, peak_flops_per_gpu=312e12,
                warmup_steps=0)
    base.update(kw)
    return tp.ThroughputTracker(**base)


def _one_step(t, *, dt, real, processed, seq_len=4096):
    t._last_t = 100.0
    # step() reads the clock; patch it for a fixed step time
    orig = tp.time.perf_counter
    tp.time.perf_counter = lambda: 100.0 + dt
    try:
        return t.step(tokens_in_step=real, samples_in_step=8, seq_len=seq_len,
                      processed_tokens_in_step=processed)
    finally:
        tp.time.perf_counter = orig


def test_full_finetune_is_6n():
    out = _one_step(_tracker(), dt=10.0, real=30_000, processed=30_000)
    expected = 100 * 6 * 10e9 * 3_000 / 312e12
    assert out["throughput/mfu_pct"] == pytest.approx(expected)


def test_lora_frozen_base_is_about_4n():
    full = _one_step(_tracker(), dt=10.0, real=30_000, processed=30_000)["throughput/mfu_pct"]
    lora = _one_step(_tracker(trainable_params=40_000_000), dt=10.0, real=30_000,
                     processed=30_000)["throughput/mfu_pct"]
    assert lora / full == pytest.approx((4 * 10e9 + 2 * 40e6) / (6 * 10e9))


def test_padding_counts_toward_mfu_but_not_tokens_per_sec():
    out = _one_step(_tracker(), dt=10.0, real=20_000, processed=40_000)
    assert out["throughput/padding_pct"] == pytest.approx(50.0)
    assert out["throughput/tokens_per_sec_per_gpu"] == pytest.approx(2_000)
    assert out["throughput/mfu_pct"] == pytest.approx(100 * 6 * 10e9 * 4_000 / 312e12)


def test_hfu_adds_recompute_only_with_checkpointing():
    off = _one_step(_tracker(recompute=False), dt=10.0, real=30_000, processed=30_000)
    on = _one_step(_tracker(recompute=True), dt=10.0, real=30_000, processed=30_000)
    assert off["throughput/hfu_pct"] == pytest.approx(off["throughput/mfu_pct"])
    assert on["throughput/hfu_pct"] > on["throughput/mfu_pct"]


def test_attention_term_added_when_shape_known():
    base = _one_step(_tracker(), dt=10.0, real=30_000, processed=30_000)["throughput/mfu_pct"]
    withattn = _one_step(_tracker(num_layers=40, hidden_size=5120), dt=10.0, real=30_000,
                         processed=30_000)["throughput/mfu_pct"]
    assert withattn > base
