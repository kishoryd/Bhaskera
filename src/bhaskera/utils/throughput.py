"""
bhaskera.utils.throughput
=========================
Step-time, tokens/sec, samples/sec, and an MFU estimate for LLM
fine-tuning loops.

The MFU calculation counts the matrix-multiply FLOPs a step really needs:
    forward                      2 * N                 per token
    backward, activation grads   2 * N                 (needed even when weights are frozen)
    backward, weight grads       2 * N_trainable       (zero for frozen base weights)
    attention (causal)           6 * layers * hidden * seq_len
so full fine-tuning is the usual 6 * N, while LoRA (frozen base) is about 4 * N.
N is the full parameter count for dense models and the active count for MoE.

MFU divides those model FLOPs by peak, using the tokens the GPU actually
processed (rows x seq_len, padding included). Padding costs the same compute as
real text, so leaving it out understated MFU by the padding share (about 43% on
the Nemotron cache). ``tokens_per_sec`` still counts real tokens only, and
``padding_pct`` reports the share that was padding, so both views are available.

``hfu_pct`` (hardware FLOPs utilization) additionally counts the forward pass that
activation checkpointing recomputes during backward.

Counting convention (the one thing to get right here):
    ``step()`` takes **per-rank** token and sample counts, because the
    training loop sums its own ``attention_mask`` and the train dataset is
    sharded per rank.  ``mfu_pct`` and ``tokens_per_sec_per_gpu`` are
    therefore that rate as-is; ``tokens_per_sec`` and ``samples_per_sec``
    scale it up by ``world_size`` to describe the whole job.

Notes:
    * The first ``warmup_steps`` step times are dropped from the
      moving averages — they include compile / cache warmup and would
      otherwise drag the EMA down for the rest of the run.
    * ``peak_flops_per_gpu`` defaults to A100-bf16 (312 TFLOPS).
      Override per-GPU-type via the ``monitoring.peak_tflops_per_gpu``
      config field.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional


class ThroughputTracker:
    """Lightweight tracker — call ``step()`` once per optimizer step."""

    def __init__(
        self,
        *,
        params_for_flops: int,
        world_size: int,
        peak_flops_per_gpu: float = 312e12,  # A100 bf16
        window: int = 50,
        warmup_steps: int = 5,
        trainable_params: Optional[int] = None,
        num_layers: int = 0,
        hidden_size: int = 0,
        recompute: bool = False,
    ) -> None:
        self._params = max(1, int(params_for_flops))
        # None -> every weight trains (full fine-tuning), the old 6 * N behaviour.
        self._trainable = self._params if trainable_params is None else min(max(0, int(trainable_params)), self._params)
        self._layers = max(0, int(num_layers))
        self._hidden = max(0, int(hidden_size))
        self._recompute = bool(recompute)
        self._world  = max(1, int(world_size))
        self._peak   = max(1.0, float(peak_flops_per_gpu))
        self._window = max(1, int(window))
        self._warmup = max(0, int(warmup_steps))

        self._step_times: deque[float] = deque(maxlen=self._window)
        self._last_t: Optional[float] = None
        self._steps_seen = 0

    def reset_step_clock(self) -> None:
        """Call right before the first forward of a new step."""
        self._last_t = time.perf_counter()

    def step(
        self,
        *,
        tokens_in_step: int,
        samples_in_step: int,
        seq_len: int,
        processed_tokens_in_step: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Close out one optimizer step and emit derived metrics.

        Returns a dict like::

            {
                "throughput/step_time_s":        0.412,
                "throughput/step_time_ema_s":    0.418,
                "throughput/tokens_per_sec":     19880.0,
                "throughput/tokens_per_sec_per_gpu": 2485.0,
                "throughput/samples_per_sec":    9.7,
                "throughput/mfu_pct":            41.2,
                "throughput/total_steps":        137.0,
            }
        """
        now = time.perf_counter()
        out: dict[str, float] = {}
        self._steps_seen += 1
        out["throughput/total_steps"] = float(self._steps_seen)

        if self._last_t is None:
            self._last_t = now
            return out
        dt = now - self._last_t
        self._last_t = now

        if dt <= 0:
            return out

        out["throughput/step_time_s"] = dt
        if self._steps_seen > self._warmup:
            self._step_times.append(dt)
        if self._step_times:
            ema = sum(self._step_times) / len(self._step_times)
            out["throughput/step_time_ema_s"] = ema
            ref_dt = ema
        else:
            ref_dt = dt

        # Throughput uses the smoothed dt so the panel doesn't jitter.
        #
        # `tokens_in_step` / `samples_in_step` are THIS RANK's counts: the
        # training loop sums its own attention_mask, and "train" is sharded
        # per rank, so what arrives here is already a per-GPU quantity.  The
        # cluster-wide figure is therefore the local rate scaled *up* by
        # world size — derived rather than all-reduced, because a collective
        # on every optimizer step would cost more than the metric is worth.
        tps_per_gpu = tokens_in_step / ref_dt if tokens_in_step > 0 else 0.0
        if tokens_in_step > 0:
            out["throughput/tokens_per_sec"] = tps_per_gpu * self._world
            out["throughput/tokens_per_sec_per_gpu"] = tps_per_gpu
        if samples_in_step > 0:
            out["throughput/samples_per_sec"] = samples_in_step / ref_dt * self._world

        # MFU — model FLOPs utilization, per GPU (convention in the module docstring).
        #
        # `tps_per_gpu` is one GPU's token rate, so it divides straight into
        # one GPU's peak — no world-size term belongs in this ratio.  It was
        # previously divided by world a second time, under-reporting MFU by
        # exactly world_size (a 16-GPU run read 0.18% for a true 2.87%).
        #
        # MFU counts the tokens the GPU processed (padding included); the real-token
        # rate above is unchanged.  With sequence packing the two coincide.
        processed = processed_tokens_in_step if processed_tokens_in_step else tokens_in_step
        if processed > 0:
            attn = 6.0 * self._layers * self._hidden * max(0, int(seq_len))  # causal, fwd+bwd
            model_flops_per_token = 4.0 * self._params + 2.0 * self._trainable + attn
            proc_per_sec = processed / ref_dt
            out["throughput/mfu_pct"] = 100.0 * model_flops_per_token * proc_per_sec / self._peak
            # This step alone (raw dt, not the running average above), so one slow
            # step shows up as one slow step instead of dragging every later value.
            out["throughput/mfu_step_pct"] = 100.0 * model_flops_per_token * (processed / dt) / self._peak
            if self._recompute:
                # activation checkpointing reruns the forward pass inside backward
                hw_flops_per_token = model_flops_per_token + 2.0 * self._params + attn / 3.0
            else:
                hw_flops_per_token = model_flops_per_token
            out["throughput/hfu_pct"] = 100.0 * hw_flops_per_token * proc_per_sec / self._peak
            if tokens_in_step > 0:
                out["throughput/padding_pct"] = 100.0 * max(0.0, 1.0 - tokens_in_step / processed)
        return out
