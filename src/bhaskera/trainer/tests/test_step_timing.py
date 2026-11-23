"""
StepTimer must split a step into data / compute / optimizer time and name the
slowest rank. Multi-GPU (NCCL); run with:
    torchrun --nproc_per_node=2 -m bhaskera.trainer.tests.test_step_timing
Rank 1 is made slower on purpose; rank 0 must report it as the straggler.
"""
from __future__ import annotations

import logging
import os
import sys
import time

import torch
import torch.distributed as dist

from bhaskera.trainer.step_timing import StepTimer


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", torch.cuda.current_device())
    timer = StepTimer(enabled=True, device=device, rank=rank, world_size=world,
                      straggler_every_n_steps=1)
    a = torch.randn(4096, 4096, device=device)
    ok = True
    for step in range(1, 4):
        timer.start_window()
        with timer.data_wait():
            time.sleep(0.2 + (1.5 if rank == 1 else 0.0))   # rank 1: slow data
        timer.compute_start()
        for _ in range(20):
            a = (a @ a).tanh()
        timer.compute_end()
        timer.optim_start()
        a = a * 0.5
        timer.optim_end()
        a.sum().item()                                          # the loop's sync point
        m = timer.finish(step)
        if rank == 0:
            parts = m["time/data_wait_s"] + m["time/compute_s"] + m["time/optimizer_s"] + m["time/other_s"]
            print(f"step {step}: step={m['time/step_s']:.3f} data={m['time/data_wait_s']:.3f} "
                  f"compute={m['time/compute_s']:.3f} opt={m['time/optimizer_s']:.4f} "
                  f"slowest=r{int(m['time/slowest_rank'])}", flush=True)
            ok &= abs(parts - m["time/step_s"]) < 0.05 or m["time/other_s"] == 0.0
            ok &= int(m["time/slowest_rank"]) == 1 and m["time/compute_s"] > 0
            ok &= 0.15 < m["time/data_wait_s"] < 0.4
    flag = torch.tensor([1 if ok else 0], device=device)
    dist.broadcast(flag, 0)
    if rank == 0:
        print("=== STEP TIMING:", "PASS" if flag.item() else "FAIL", "===", flush=True)
    dist.destroy_process_group()
    return 0 if flag.item() else 1


if __name__ == "__main__":
    sys.exit(main())
