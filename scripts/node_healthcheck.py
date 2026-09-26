#!/usr/bin/env python3
"""
Per-node GPU smoke test, one process per node (see scripts/preflight_nodes.sh).

Why: individual bad nodes on this cluster killed or hung multi-node jobs
repeatedly -- "CUDA error: invalid device ordinal", "CUDA error: unrecognized
error code" inside NCCL all-gathers, and an 8-node group on which two runs
hung for hours. Those faults show up in a ~1 minute local test, long before a
model load finishes.

Checks every visible GPU: expected count, a bf16 matmul (finite result,
throughput, and a checksum that must agree across the node's GPUs since the
inputs are identical), and a device-to-device copy between local GPUs.
Prints one JSON line: {"host", "ok", "gpus", "tflops", "p2p_gbps", "errors"}.
"""
from __future__ import annotations

import argparse
import json
import socket
import time


def _gpu_test(torch, idx: int, n: int, iters: int) -> tuple[float, float]:
    dev = torch.device("cuda", idx)
    torch.cuda.set_device(dev)
    gen = torch.Generator(device=dev).manual_seed(1234)
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16, generator=gen)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16, generator=gen)
    c = a @ b  # warm-up
    torch.cuda.synchronize(dev)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize(dev)
    secs = start.elapsed_time(end) / 1000.0
    if not torch.isfinite(c).all():
        raise RuntimeError("non-finite matmul result")
    tflops = 2 * n ** 3 * iters / secs / 1e12
    return tflops, float(c.float().sum().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expected-gpus", type=int, default=2)
    p.add_argument("--size", type=int, default=8192)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    result = {"host": socket.gethostname(), "ok": True, "gpus": 0,
              "tflops": [], "p2p_gbps": None, "errors": []}
    try:
        import torch

        n_gpus = torch.cuda.device_count()
        result["gpus"] = n_gpus
        if n_gpus != args.expected_gpus:
            result["errors"].append(f"expected {args.expected_gpus} GPUs, found {n_gpus}")
        checksums = []
        for i in range(n_gpus):
            try:
                tflops, checksum = _gpu_test(torch, i, args.size, args.iters)
                result["tflops"].append(round(tflops, 1))
                checksums.append(checksum)
            except Exception as e:  # noqa: BLE001 -- any CUDA fault marks the node bad
                result["tflops"].append(None)
                result["errors"].append(f"gpu{i}: {type(e).__name__}: {str(e)[:160]}")
        if len(set(checksums)) > 1:
            result["errors"].append(f"matmul checksums disagree across GPUs: {checksums}")
        if n_gpus >= 2 and not any(r is None for r in result["tflops"]):
            src = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
            dst = torch.empty_like(src, device="cuda:1")
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(5):
                dst.copy_(src)
            torch.cuda.synchronize()
            result["p2p_gbps"] = round(5 * src.numel() / (time.perf_counter() - t) / 1e9, 1)
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
    result["ok"] = not result["errors"]
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
