"""
HSDP (sharding_strategy: HYBRID_SHARD) must match FULL_SHARD numerically and
shard as intended. Needs >=2 nodes x 2 GPUs, one process per GPU, launched by
srun with the torch env set (see .claude_tmp/hsdp_unittest.sbatch):
    python -m bhaskera.distributed.tests.test_hsdp

On a tiny Qwen3 model in fp32, from identical initial weights and per-rank
data:
  1. placement: FULL_SHARD holds 1/world of each param per GPU; HSDP holds
     1/shard_size with (Replicate, Shard(0)) placements, and every shard group
     sits on one node
  2. loss and post-step weights equal between FULL_SHARD and HSDP
  3. after the step, HSDP replica groups hold identical weights
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from bhaskera.distributed.fsdp import _build_device_mesh


def _model(device):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, attn_implementation="eager")
    return Qwen3ForCausalLM(cfg).to(device)


def _shard(model, strategy, shard_size):
    fsdp_cfg = types.SimpleNamespace(sharding_strategy=strategy, hsdp_shard_size=shard_size)
    mesh = _build_device_mesh(fsdp_cfg)
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model, mesh


def _train_step(model, ids):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    opt.step()
    avg = loss.detach().clone()
    dist.all_reduce(avg)
    return avg.item() / dist.get_world_size()


def main() -> int:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    shard = int(os.environ.get("LOCAL_WORLD_SIZE", "2"))
    torch.manual_seed(100 + rank)
    ids = torch.randint(0, 256, (2, 32), device=device)
    ok = True

    full, _ = _shard(_model(device), "FULL_SHARD", 0)
    hsdp, mesh = _shard(_model(device), "HYBRID_SHARD", shard)

    p_full = next(p for n, p in full.named_parameters() if n.endswith("q_proj.weight"))
    p_hsdp = next(p for n, p in hsdp.named_parameters() if n.endswith("q_proj.weight"))
    full_frac = p_full.to_local().numel() / p_full.numel()
    hsdp_frac = p_hsdp.to_local().numel() / p_hsdp.numel()
    placements = tuple(type(pl).__name__ for pl in p_hsdp.placements)
    placed = (abs(full_frac - 1 / world) < 1e-9 and abs(hsdp_frac - 1 / shard) < 1e-9
              and mesh is not None and mesh.ndim == 2 and placements == ("Replicate", "Shard"))
    if rank == 0:
        print(f"1. placement: FULL_SHARD holds {full_frac:.3f}/param, HSDP {hsdp_frac:.3f}/param, "
              f"HSDP mesh {tuple(mesh.shape) if mesh is not None else None}, placements {placements}  "
              f"{'OK' if placed else 'FAIL'}", flush=True)
    ok &= placed

    loss_full = _train_step(full, ids)
    loss_hsdp = _train_step(hsdp, ids)
    max_diff = 0.0
    for (n, a), (_, b) in zip(full.named_parameters(), hsdp.named_parameters()):
        diff = (a.full_tensor() - b.full_tensor()).abs().max().item()
        max_diff = max(max_diff, diff)
    same = abs(loss_full - loss_hsdp) < 1e-5 and max_diff < 1e-5
    if rank == 0:
        print(f"2. loss FULL {loss_full:.6f} vs HSDP {loss_hsdp:.6f}; post-step weights "
              f"max |diff| {max_diff:.2e}  {'OK' if same else 'FAIL'}", flush=True)
    ok &= same

    local_sum = torch.stack([p.to_local().double().sum() for p in hsdp.parameters()]).sum()
    replica_group = mesh.get_group("replicate")
    sums = [torch.zeros_like(local_sum) for _ in range(dist.get_world_size(replica_group))]
    dist.all_gather(sums, local_sum, group=replica_group)
    agree = all(torch.equal(s, sums[0]) for s in sums)
    if rank == 0:
        print(f"3. HSDP replicas hold identical shards after the step: {agree}  "
              f"{'OK' if agree else 'FAIL'}", flush=True)
    ok &= agree

    flag = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("=== HSDP:", "PASS" if flag.item() else "FAIL", "===", flush=True)
    dist.destroy_process_group()
    return 0 if flag.item() else 1


if __name__ == "__main__":
    sys.exit(main())
