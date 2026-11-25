"""Which metrics reach MLflow.

A run used to write about 170 curves, most of them system readings of one node. By default only the curves a
user reads while fine-tuning are written; the rest can be switched on with ``logging.mlflow_metrics``:

* ``essential`` (default): loss, learning rate, gradient norm, evaluation (loss, accuracy, perplexity), speed
  and MFU, the share of time spent waiting for data, and GPU load, memory and temperature over the whole job.
* ``detailed``: also every speed, step-time, loss-statistic, job-wide GPU and GPU-memory curve.
* ``full``: everything, including the readings of each GPU and node, and the process and host statistics.
"""
from __future__ import annotations

LEVELS = ("essential", "detailed", "full")

ESSENTIAL = frozenset({
    "loss", "loss_running_avg", "lr", "grad_norm", "epoch",
    "eval_loss", "eval_accuracy", "eval_perplexity",
    "throughput_tokens_per_sec", "throughput_mfu_pct", "time_step_s", "time_data_wait_pct",
    "cluster_nodes", "cluster_gpus", "cluster_gpu_util_pct_min", "cluster_gpu_util_pct_mean",
    "cluster_gpu_mem_used_gib_max", "cluster_gpu_temp_c_max", "cluster_gpu_power_w_sum",
    "model_total_params", "model_trainable_params",
})

# Added by "detailed": whole families of job-wide curves.
_DETAILED_PREFIXES = ("throughput_", "time_", "loss_", "cluster_", "cuda_", "model_", "eval_")


def normalize_level(value) -> str:
    """The level named in the recipe; anything unknown falls back to the default."""
    level = str(value or "essential").strip().lower()
    return level if level in LEVELS else "essential"


def wanted(name: str, level: str) -> bool:
    """Whether the metric ``name`` (already written with underscores) is written at ``level``."""
    if level == "full":
        return True
    if name in ESSENTIAL:
        return True
    return level == "detailed" and name.startswith(_DETAILED_PREFIXES)
