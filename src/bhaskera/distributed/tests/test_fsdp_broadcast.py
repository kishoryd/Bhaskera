"""
The FSDP load path must reproduce the pretrained model exactly on every rank:
rank 0 builds real weights, other ranks build on `meta`, the model is sharded,
then _broadcast_pretrained_weights fills every rank from rank 0.

Regression: non-persistent buffers (rotary_emb.inv_freq) are absent from
state_dict(), and were broadcast as torch.empty() garbage, including over
rank 0's own correct copy. RoPE broke on every rank and FSDP runs started at
loss ~9-15 instead of ~1.4.

    torchrun --nproc_per_node=2 -m bhaskera.distributed.tests.test_fsdp_broadcast
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from bhaskera.distributed.fsdp import _broadcast_pretrained_weights


def _build(device):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, attn_implementation="eager")
    with torch.device(device):
        return Qwen3ForCausalLM(cfg)


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    ref = _build("cpu").to(device)
    ref_tensors = {n: t.detach().clone() for n, t in
                   list(ref.named_parameters()) + list(ref.named_buffers())}

    model = _build("cpu" if rank == 0 else "meta")
    sd = model.state_dict() if rank == 0 else None
    for layer in model.model.layers:
        fully_shard(layer)
    fully_shard(model)
    _broadcast_pretrained_weights(model, sd, device)

    ok = True
    wrong = []
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        full = t.full_tensor() if hasattr(t, "full_tensor") else t
        if not torch.equal(full.to(device), ref_tensors[n]):
            wrong.append(n)
    ok &= not wrong
    inv = dict(model.named_buffers())["model.rotary_emb.inv_freq"]
    inv_ok = torch.equal(inv, ref_tensors["model.rotary_emb.inv_freq"])
    print(f"[rank {rank}] 1. tensors equal to reference: {not wrong} "
          f"(wrong: {wrong[:5]}); inv_freq correct: {inv_ok}  "
          f"{'OK' if not wrong else 'FAIL'}", flush=True)

    ids = torch.randint(0, 256, (2, 32), device=device, generator=torch.Generator(device).manual_seed(1))
    with torch.no_grad():
        loss = model(input_ids=ids, labels=ids).loss.item()
        ref_loss = ref(input_ids=ids, labels=ids).loss.item()
    same = abs(loss - ref_loss) < 1e-5
    print(f"[rank {rank}] 2. loss sharded {loss:.6f} vs reference {ref_loss:.6f}  "
          f"{'OK' if same else 'FAIL'}", flush=True)
    ok &= same

    flag = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("=== FSDP BROADCAST:", "PASS" if flag.item() else "FAIL", "===", flush=True)
    dist.destroy_process_group()
    return 0 if flag.item() else 1


if __name__ == "__main__":
    sys.exit(main())
