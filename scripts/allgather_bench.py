"""All-gather bandwidth benchmark that mirrors FSDP's weight gathers, without Ray.

One process per GPU (launched by srun). Each round all-gathers one Llama-3.3-70B
decoder layer's worth of bf16 weights (~1.75 GB) the way FSDP2 does:
all_gather_into_tensor on the default stream. Reports, per variant:

    algbw  = bytes gathered / time                       (what one GPU receives)
    busbw  = algbw * (n - 1) / n                         (what each link carries in a ring)

Variants (BENCH_GROUPS, comma-free so it survives `sbatch --export`):
    world     gather across all ranks (FSDP today)
    group:K   every rank gathers inside its block of K consecutive ranks, all blocks
              at once -- the backward gather when reshard_after_forward=K

NCCL settings (NCCL_ALGO, NCCL_PROTO, NCCL_IB_QPS_PER_CONNECTION, ...) are taken
from the environment as usual, so each job measures one setting.
"""
from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist

LAYER_BYTES = int(float(os.environ.get("BENCH_BYTES", 1.75e9)))
ITERS = int(os.environ.get("BENCH_ITERS", "10"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))


def bench(group, n: int, rank0: bool, label: str) -> None:
    per_rank = LAYER_BYTES // 2 // n                      # bf16 elements per rank
    shard = torch.randn(per_rank, dtype=torch.bfloat16, device="cuda")
    full = torch.empty(per_rank * n, dtype=torch.bfloat16, device="cuda")
    for _ in range(WARMUP):
        dist.all_gather_into_tensor(full, shard, group=group)
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        dist.all_gather_into_tensor(full, shard, group=group)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    # slowest rank decides the collective's real duration
    worst = torch.tensor(sorted(times)[len(times) // 2], device="cuda")
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    if rank0:
        nbytes = full.numel() * full.element_size()
        algbw = nbytes / worst.item() / 1e9
        print(f"[bench] {label:<12} ranks={n:<4} time={worst.item()*1e3:8.1f} ms  "
              f"algbw={algbw:6.2f} GB/s  busbw={algbw*(n-1)/n:6.2f} GB/s  "
              f"({algbw*(n-1)/n*8:5.1f} Gb/s per link)", flush=True)


def main() -> None:
    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    if rank == 0:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
        print(f"[bench] host={socket.gethostname()} ranks={world} nccl={nccl} "
              f"layer={LAYER_BYTES/1e9:.2f} GB  env: " + " ".join(
                  f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")), flush=True)
    for spec in os.environ.get("BENCH_GROUPS", "world").split():
        if spec == "world":
            bench(None, world, rank == 0, "world")
        else:
            k = int(spec.split(":")[1])
            groups = [dist.new_group(list(range(s, s + k))) for s in range(0, world, k)]
            bench(groups[rank // k], k, rank == 0, f"group:{k}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
