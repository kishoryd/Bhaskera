"""Pipeline-parallel LoRA benchmark: each GPU keeps its slice of the frozen model.

Why this exists: under FSDP every GPU receives the whole frozen model twice per step
(~280 GB for Llama-3.3-70B), which over one 100 Gb/s link per node costs ~47 s per step
at any node count. Splitting the model by layers across a few GPUs (pipeline parallel)
keeps the frozen weights where they are: only activations cross a stage boundary
(~67 MB per 4,096-token sequence for Llama-70B) and only the LoRA gradients are averaged
across pipeline copies (data parallel).

Launch: one process per GPU under srun, see scripts/pp_lora_bench.sbatch.
  --check   tiny Llama in fp32: compares the loss and every LoRA gradient against an
            unsplit single-GPU reference on rank 0; exit code 1 on mismatch.
  default   throughput on a real model config with random weights (speed does not depend
            on weight values, and it skips a ~141 GB checkpoint read). bf16 + FlashAttention-2
            + activation checkpointing per layer. Reports step time, tokens/s and MFU with
            the FLOP formula of bhaskera.utils.throughput (4N + 2*N_trainable + 6*L*h*S).

Rank layout: rank = replica * pp + stage, so a pipeline is pp consecutive ranks; with
2 GPUs per node, stages 0-1 share a node (PCIe) and 1->2 crosses InfiniBand.
Limits of this prototype: unpacked random sequences (no runtime packing), equal layer
split (the last stage also runs the LM head), 1F1B or GPipe schedule only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import zlib
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SCHEDULES = {"1f1b": Schedule1F1B, "gpipe": ScheduleGPipe}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="", help="HF model directory (config only is read)")
    p.add_argument("--pp", type=int, default=4, help="pipeline stages (GPUs per model copy)")
    p.add_argument("--microbatches", type=int, default=16, help="micro-batches per step per pipeline")
    p.add_argument("--mb-size", type=int, default=1, help="sequences per micro-batch")
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3, help="steps left out of the medians")
    p.add_argument("--schedule", default="1f1b", choices=sorted(SCHEDULES))
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-ac", action="store_true", help="disable activation checkpointing")
    p.add_argument("--check", action="store_true", help="tiny-model correctness test")
    p.add_argument("--peak-tflops", type=float, default=312.0)
    p.add_argument("--nccl-timeout-min", type=float, default=10.0)
    return p.parse_args()


def tiny_config() -> LlamaConfig:
    return LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=344,
                       num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
                       max_position_embeddings=512, tie_word_embeddings=False)


def split_layers(n_layers: int, pp: int) -> list[list[int]]:
    base, extra = divmod(n_layers, pp)
    sizes = [base + (1 if 0 < s <= extra else 0) for s in range(pp)]   # remainder to middle stages
    if sum(sizes) != n_layers:
        sizes[-1] += n_layers - sum(sizes)
    out, start = [], 0
    for size in sizes:
        out.append(list(range(start, start + size)))
        start += size
    return out


def canon(name: str, layer_ids=None) -> str:
    """Stage-local or wrapped parameter name -> the unsplit model's name."""
    n = name.replace("._checkpoint_wrapped_module", "").replace(".base_layer", "")
    if layer_ids is not None:
        if n.startswith("layers."):
            j, rest = n[len("layers."):].split(".", 1)
            n = f"model.layers.{layer_ids[int(j)]}.{rest}"
        elif n.startswith(("embed_tokens.", "norm.")):
            n = "model." + n
    return n


def det_fill_(named_params, layer_ids=None) -> None:
    """Same values for the same canonical name on every rank (pipeline vs reference)."""
    with torch.no_grad():
        for name, p in named_params:
            c = canon(name, layer_ids)
            g = torch.Generator().manual_seed(zlib.crc32(c.encode()))
            t = torch.randn(p.shape, generator=g, dtype=torch.float32)
            t = 1.0 + 0.02 * t if c.endswith("norm.weight") else 0.05 * t
            p.copy_(t.to(p.dtype))


class Stage(nn.Module):
    """A contiguous slice of a Llama/Mistral-style decoder: [embed] layers [norm].
    The LM head is kept here but applied in the loss (fused with cross-entropy)."""

    def __init__(self, full, layer_ids, first: bool, last: bool):
        super().__init__()
        inner = full.model
        self.embed_tokens = inner.embed_tokens if first else None
        self.layers = nn.ModuleList(inner.layers[i] for i in layer_ids)
        self.norm = inner.norm if last else None
        self.lm_head = full.lm_head if last else None
        self.rotary = None   # built on the GPU after materialization (inv_freq is not a weight)

    def forward(self, x):
        h = self.embed_tokens(x) if self.embed_tokens is not None else x
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0).expand(h.shape[0], -1)
        pe = self.rotary(h, pos)
        for layer in self.layers:
            h = layer(h, position_ids=pos, position_embeddings=pe)
        return self.norm(h) if self.norm is not None else h


def build_full(cfg, attn: str, dtype):
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(cfg, attn_implementation=attn, dtype=dtype)


def add_lora(module: nn.Module, args, check: bool) -> nn.Module:
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                      target_modules=TARGETS, init_lora_weights=not check)
    module = inject_adapter_in_model(lcfg, module)
    for n, p in module.named_parameters():
        p.requires_grad = "lora_" in n
    return module


def make_batch(step: int, replica: int, rows: int, seq: int, vocab: int, device):
    g = torch.Generator().manual_seed(1000 * step + replica)
    x = torch.randint(0, vocab, (rows, seq), generator=g)
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = -100
    return x.to(device), y.to(device)


def make_loss(stage: Stage, m: int, fused: bool):
    if fused:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        flce = LigerFusedLinearCrossEntropyLoss()

        def loss_fn(h, target):   # never materializes the [tokens, vocab] logits
            return flce(stage.lm_head.weight, h.reshape(-1, h.shape[-1]), target.reshape(-1)) / m
    else:
        def loss_fn(h, target):
            logits = stage.lm_head(h).float()
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)) / m
    return loss_fn


def main() -> None:
    args = parse()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    # No device_id: binding it makes new_group() split the default communicator, and the
    # pipeline's lazily created 2-rank send/recv communicators then collided across
    # pipeline copies ("rank 1 of 2 ranks has already checked in"); pp > 1 with dp > 1 hung.
    dist.init_process_group("nccl", timeout=timedelta(minutes=args.nccl_timeout_min))
    pp = args.pp
    assert world % pp == 0, f"world {world} not divisible by pp {pp}"
    dp = world // pp
    mesh = init_device_mesh("cuda", (dp, pp), mesh_dim_names=("dp", "pp"))
    replica, s = rank // pp, rank % pp
    pp_group, dp_group = mesh["pp"].get_group(), mesh["dp"].get_group()
    first, last = s == 0, s == pp - 1
    m = args.microbatches
    rows = m * args.mb_size

    if args.check:
        cfg, attn, dtype = tiny_config(), "sdpa", torch.float32
    else:
        cfg, attn, dtype = AutoConfig.from_pretrained(args.model), "flash_attention_2", torch.bfloat16
    full = build_full(cfg, attn, dtype)
    n_total = sum(p.numel() for p in full.parameters())
    layer_ids = split_layers(cfg.num_hidden_layers, pp)[s]
    rotary_cls = type(full.model.rotary_emb)

    stage = Stage(full, layer_ids, first, last)
    del full
    stage.to_empty(device=dev)
    if args.check:
        det_fill_(stage.named_parameters(), layer_ids)
    else:
        with torch.no_grad():
            for n, p in stage.named_parameters():
                p.fill_(1.0) if n.endswith("norm.weight") else p.normal_(0.0, 0.02)
    stage.rotary = rotary_cls(config=cfg, device=dev)
    stage = add_lora(stage, args, args.check)
    if args.check:
        det_fill_([(n, p) for n, p in stage.named_parameters() if "lora_" in n], layer_ids)
    if not args.no_ac:
        for j in range(len(stage.layers)):
            stage.layers[j] = checkpoint_wrapper(stage.layers[j], checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    lora = [p for p in stage.parameters() if p.requires_grad]
    n_train = torch.tensor(sum(p.numel() for p in lora), device=dev)
    dist.all_reduce(n_train, group=pp_group)
    n_train = int(n_train.item())
    opt = torch.optim.AdamW(lora, lr=1e-4, weight_decay=0.0, fused=True)

    pstage = PipelineStage(stage, stage_index=s, num_stages=pp, device=dev, group=pp_group)
    sched = SCHEDULES[args.schedule](pstage, n_microbatches=m,
                                     loss_fn=make_loss(stage, m, fused=not args.check))
    seq = args.seq if not args.check else 64
    flops_tok = 4 * n_total + 2 * n_train + 6 * cfg.num_hidden_layers * cfg.hidden_size * seq
    if rank == 0:
        print(json.dumps({"event": "setup", "world": world, "pp": pp, "dp": dp, "microbatches": m,
                          "mb_size": args.mb_size, "seq": seq, "schedule": args.schedule,
                          "params": n_total, "lora_params": n_train,
                          "layers_per_stage": [len(x) for x in split_layers(cfg.num_hidden_layers, pp)],
                          "ac": not args.no_ac, "bubble_1f1b": round((pp - 1) / (m + pp - 1), 3)}),
              flush=True)

    rows_log = []
    steps = 1 if args.check else args.steps
    for step in range(steps):
        x, y = make_batch(step, replica, rows, seq, cfg.vocab_size, dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses: list = []
        if first:
            sched.step(x)
        elif last:
            sched.step(target=y, losses=losses)
        else:
            sched.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if dp > 1:   # average LoRA grads across pipeline copies (the only cross-copy traffic)
            flat = torch._utils._flatten_dense_tensors([p.grad for p in lora])
            dist.all_reduce(flat, group=dp_group)
            flat /= dp
            for p, g in zip(lora, torch._utils._unflatten_dense_tensors(flat, [p.grad for p in lora])):
                p.grad.copy_(g)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if args.check:
            break
        sq = torch.stack([p.grad.float().pow(2).sum() for p in lora]).sum()
        dist.all_reduce(sq, group=pp_group)
        clip = min(1.0, 1.0 / (sq.sqrt().item() + 1e-6))
        if clip < 1.0:
            torch._foreach_mul_([p.grad for p in lora], clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        loss = torch.tensor(sum(float(v) for v in losses) if last else 0.0, device=dev)
        dist.all_reduce(loss, group=pp_group)
        step_s = t3 - t0
        tok_gpu = dp * rows * seq / step_s / world
        rec = {"event": "step", "step": step, "step_s": round(step_s, 3),
               "pipeline_s": round(t1 - t0, 3), "dp_allreduce_s": round(t2 - t1, 3),
               "optim_s": round(t3 - t2, 3), "tokens_per_s": round(tok_gpu * world),
               "mfu_pct": round(100 * tok_gpu * flops_tok / (args.peak_tflops * 1e12), 2),
               "loss": round(loss.item(), 4)}
        rows_log.append(rec)
        if rank == 0:
            print(json.dumps(rec), flush=True)

    if args.check:
        ok = _check(args, cfg, rank, s, replica, first, last, losses, stage, lora, layer_ids,
                    dp, m, rows, seq, dev, pp_group)
        _finish(0 if ok else 1)

    mem = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=dev)
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    if rank == 0:
        keep = rows_log[args.warmup:] or rows_log
        med = {k: statistics.median(r[k] for r in keep)
               for k in ("step_s", "pipeline_s", "dp_allreduce_s", "optim_s", "tokens_per_s", "mfu_pct")}
        print(json.dumps({"event": "summary", "nodes": world // 2, "pp": pp, "dp": dp,
                          "microbatches": m, "global_batch_seqs": dp * rows,
                          "steps_in_median": len(keep), **med,
                          "max_mem_gib": round(mem.item(), 1)}), flush=True)
    _finish(0)


def _finish(code: int) -> None:
    """Every rank waits for the others (rank 0 may still be running the reference), then
    exits without NCCL teardown, which hung the first check run until its time limit."""
    dist.barrier()
    os._exit(code)


def _check(args, cfg, rank, s, replica, first, last, losses, stage, lora, layer_ids,
           dp, m, rows, seq, dev, pp_group) -> bool:
    """Pipeline (+ data-parallel average) vs one unsplit model over the same global batch.
    Returns the verdict on rank 0, True elsewhere."""
    mine = {}
    if replica == 0:
        mine = {canon(n, layer_ids): p.grad.detach().float().cpu()
                for n, p in stage.named_parameters() if p.requires_grad}
    loss_here = sum(float(v) for v in losses) if last else None
    grads, loss_vals = [None] * dist.get_world_size(), [None] * dist.get_world_size()
    dist.all_gather_object(grads, mine)
    dist.all_gather_object(loss_vals, loss_here)
    if rank != 0:
        return True
    pipe_grads = {k: v for d in grads for k, v in d.items()}
    pipe_loss = sum(v for v in loss_vals if v is not None) / dp

    ref = build_full(cfg, "sdpa", torch.float32)
    ref.to_empty(device=dev)
    det_fill_(ref.named_parameters())
    ref.model.rotary_emb = type(ref.model.rotary_emb)(config=cfg, device=dev)
    ref = add_lora(ref, args, check=True)
    det_fill_([(n, p) for n, p in ref.named_parameters() if "lora_" in n])
    total = 0.0
    for r in range(dp):
        x, y = make_batch(0, r, rows, seq, cfg.vocab_size, dev)
        logits = ref(input_ids=x).logits.float()
        loss_r = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)) / dp
        loss_r.backward()
        total += loss_r.item()
    ref_grads = {canon(n): p.grad.detach().float().cpu() for n, p in ref.named_parameters() if p.requires_grad}

    missing = sorted(set(ref_grads) ^ set(pipe_grads))
    worst, worst_name = 0.0, ""
    for k in ref_grads.keys() & pipe_grads.keys():
        rel = (pipe_grads[k] - ref_grads[k]).abs().max().item() / max(ref_grads[k].abs().max().item(), 1e-12)
        if rel > worst:
            worst, worst_name = rel, k
    loss_rel = abs(pipe_loss - total) / abs(total)
    ok = not missing and worst < 1e-3 and loss_rel < 1e-5
    print(json.dumps({"event": "check", "ok": ok, "pp": args.pp, "dp": dp, "microbatches": m,
                      "loss_pipeline": pipe_loss, "loss_reference": total, "loss_rel_err": loss_rel,
                      "lora_tensors": len(ref_grads), "unmatched_names": missing[:5],
                      "max_grad_rel_err": worst, "worst_tensor": worst_name}), flush=True)
    return ok


if __name__ == "__main__":
    main()
