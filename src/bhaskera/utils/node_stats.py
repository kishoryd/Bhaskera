"""GPU and host telemetry for every node of a multi-node run, collected on rank 0.

`system_stats()` reads the node the calling process runs on, and only rank 0 logs to MLflow, so on a
multi-node job the MLflow run used to show the first node alone. Here the first rank of each node reads
its node's GPUs (NVML) and host (CPU, memory) and the readings are gathered to rank 0 every few steps.

Metrics written to MLflow, all in the same run as the training metrics:

* ``cluster/*``: minimum, mean and maximum over every GPU of the job (utilisation, memory, temperature),
  the summed power draw, and the node and GPU counts. These stay small at any scale.
* ``node/<host>/gpu<i>/*`` and ``node/<host>/*``: one set of curves per node, only while the job has at
  most ``per_node_max`` nodes (a 128-node job would otherwise write more than a thousand metrics each time).

The live per-node and per-GPU view for any number of nodes is the Ray Dashboard (see ``bhaskera-dashboard ray``).
"""
from __future__ import annotations

import logging
import socket
from typing import Any

import torch.distributed as dist

from .system_stats import system_stats

logger = logging.getLogger(__name__)

_GPU_KEYS = ("util_pct", "mem_used_mib", "temp_c", "power_w")
_HOST_KEYS = {"sys/cpu_pct": "cpu_pct", "sys/mem_used_pct": "mem_used_pct", "sys/loadavg_1m": "loadavg_1m"}


def local_report(*, gpu: bool = True, cpu: bool = True) -> dict[str, Any]:
    """This node's readings: the host name and a small dict of numbers."""
    raw = system_stats(gpu=gpu, cpu=cpu)
    stats: dict[str, float] = {}
    for key, value in raw.items():
        if key.startswith("gpu/"):
            parts = key.split("/")
            if len(parts) == 3 and parts[2] in _GPU_KEYS:
                stats[key] = float(value)
        elif key in _HOST_KEYS:
            stats[key] = float(value)
    return {"host": socket.gethostname(), "stats": stats}


def gather_node_reports(local_rank: int, *, gpu: bool = True, cpu: bool = True) -> list[dict[str, Any]] | None:
    """Collective: every rank must call it at the same step. The first rank on each node contributes a
    report; rank 0 gets the list (one entry per node), every other rank gets None."""
    report = local_report(gpu=gpu, cpu=cpu) if local_rank == 0 else None
    if not (dist.is_available() and dist.is_initialized()):
        return [report] if report else None
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, report)
    if dist.get_rank() != 0:
        return None
    return [r for r in gathered if r]


def flatten(reports: list[dict[str, Any]], per_node_max: int = 16) -> dict[str, float]:
    """Turn the per-node reports into the flat metric dict that is logged."""
    util: list[float] = []
    mem_gib: list[float] = []
    temp: list[float] = []
    power: list[float] = []
    for r in reports:
        for key, value in r["stats"].items():
            if not key.startswith("gpu/"):
                continue
            name = key.split("/")[2]
            if name == "util_pct":
                util.append(value)
            elif name == "mem_used_mib":
                mem_gib.append(value / 1024.0)
            elif name == "temp_c":
                temp.append(value)
            elif name == "power_w":
                power.append(value)

    out: dict[str, float] = {"cluster/nodes": float(len(reports)), "cluster/gpus": float(len(util))}
    for label, values in (("gpu_util_pct", util), ("gpu_mem_used_gib", mem_gib), ("gpu_temp_c", temp)):
        if values:
            out[f"cluster/{label}_min"] = min(values)
            out[f"cluster/{label}_mean"] = sum(values) / len(values)
            out[f"cluster/{label}_max"] = max(values)
    if power:
        out["cluster/gpu_power_w_sum"] = sum(power)

    if len(reports) <= per_node_max:
        for r in reports:
            host = r["host"]
            for key, value in r["stats"].items():
                if key.startswith("gpu/"):
                    _, index, name = key.split("/")
                    shown = value / 1024.0 if name == "mem_used_mib" else value
                    name = "mem_used_gib" if name == "mem_used_mib" else name
                    out[f"node/{host}/gpu{index}/{name}"] = shown
                else:
                    out[f"node/{host}/{_HOST_KEYS[key]}"] = value
    return out


def summary_line(step: int, metrics: dict[str, float]) -> str:
    """One line for the job log, in the same '[step N]' form as the training line."""
    if "cluster/gpu_util_pct_mean" not in metrics:
        return f"[step {step}] nodes={int(metrics.get('cluster/nodes', 0))}"
    return (f"[step {step}] nodes={int(metrics['cluster/nodes'])} gpus={int(metrics['cluster/gpus'])} "
            f"gpu_util={metrics['cluster/gpu_util_pct_min']:.0f}/{metrics['cluster/gpu_util_pct_mean']:.0f}/"
            f"{metrics['cluster/gpu_util_pct_max']:.0f}% (min/mean/max) "
            f"gpu_mem_max={metrics.get('cluster/gpu_mem_used_gib_max', 0):.1f}GiB "
            f"temp_max={metrics.get('cluster/gpu_temp_c_max', 0):.0f}C")
