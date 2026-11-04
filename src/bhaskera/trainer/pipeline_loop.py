"""
bhaskera.trainer.pipeline_loop
==============================
Training loop for strategy: pipeline (see distributed/pipeline.py for the why).

One optimizer step:
  1. Every rank pulls one batch from its Ray Data shard (all ranks keep pulling in lockstep,
     so the shared stream never stalls on a consumer that stopped reading); only the first
     stage's batch is used, and it is broadcast to the rest of its pipeline.
  2. The batch is cut into `microbatches` micro-batches of training.batch_size rows; each is
     flattened to one [1, rows*S] row with its position ids, as the DDP/FSDP loop does with
     runtime packing, so FlashAttention keeps packed examples apart.
  3. The pipeline schedule (1F1B by default) runs forward/backward; the last stage computes
     a fused LM-head + cross-entropy loss per micro-batch, divided by the micro-batch count.
  4. LoRA grads are averaged across pipeline copies, clipped by the norm over all stages,
     then the optimizer and scheduler step.

Logs the same line and metric names as trainer/loop.py. Periodic eval (training.eval_steps,
forward-only through the pipeline on the validation shard), checkpoint save and resume are
supported; MoE aux loss is not. Resume restores the LoRA weights, optimizer, scheduler and
step; the data stream restarts (it is shuffled, as in the other strategies).

Data note: every rank reads its own shard but only the first stage's is trained on, so an
epoch covers 1/stages of the dataset; set max_steps rather than num_epochs.
"""
from __future__ import annotations

import logging
import math
import os
import re
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

from bhaskera.distributed.pipeline import _DTYPES, PipelineInfo, Stage, canonical_name
from bhaskera.utils.throughput import ThroughputTracker

from .optim import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)

_SCHEDULES = {"1f1b": Schedule1F1B, "gpipe": ScheduleGPipe}


def _make_loss(stage: Stage, microbatches: int):
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        flce = LigerFusedLinearCrossEntropyLoss()
    except Exception:          # noqa: BLE001 -- fall back to materialized logits
        flce = None

    def loss_fn(h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Same convention as HF ForCausalLMLoss on the flattened [1, rows*S] micro-batch:
        # position t predicts label t+1, the last position has no target.
        h = h.reshape(-1, h.shape[-1])
        y = labels.reshape(-1)
        y = torch.cat([y[1:], y.new_full((1,), -100)])
        if flce is not None:
            loss = flce(stage.lm_head.weight, h, y)
        else:
            loss = F.cross_entropy(stage.lm_head(h).float(), y, ignore_index=-100)
        return loss / microbatches
    return loss_fn


def _save_lora(stage: Stage, info: PipelineInfo, save_dir: str, step: int, rank: int,
               optimizer=None, scheduler=None) -> None:
    """Copy 0 of every stage writes its LoRA tensors (under unsplit-model names) and, when
    given, its optimizer/scheduler state; rank 0 writes the .complete sentinel last (same
    convention as the FSDP/DDP checkpoints). Copies hold identical weights and optimizer
    state (gradients are averaged), so copy 0 speaks for all."""
    out = os.path.join(save_dir, f"step_{step}")
    if info.copy == 0:
        os.makedirs(out, exist_ok=True)
        sd = {canonical_name(n, info.layer_ids): p.detach().float().cpu()
              for n, p in stage.named_parameters() if p.requires_grad}
        torch.save(sd, os.path.join(out, f"lora_stage{info.stage}.pt"))
        if optimizer is not None:
            torch.save({"optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "step": step},
                       os.path.join(out, f"opt_stage{info.stage}.pt"))
    dist.barrier()
    if rank == 0:
        open(os.path.join(out, ".complete"), "w").close()
        logger.info(f"saved LoRA weights for {info.stages} stages to {out}")


def _latest_checkpoint(save_dir: str) -> tuple[int, str] | None:
    if not os.path.isdir(save_dir):
        return None
    best = None
    for d in os.listdir(save_dir):
        m = re.fullmatch(r"step_(\d+)", d)
        if m and os.path.exists(os.path.join(save_dir, d, ".complete")):
            if best is None or int(m.group(1)) > best[0]:
                best = (int(m.group(1)), os.path.join(save_dir, d))
    return best


def _resume(stage: Stage, info: PipelineInfo, optimizer, scheduler, save_dir: str,
            rank: int, device) -> int:
    """Load the newest complete checkpoint into every rank's stage. Returns its step (0 = none).
    A checkpoint without optimizer state (older runs) restores the weights only."""
    found = _latest_checkpoint(save_dir)
    if found is None:
        return 0
    step, path = found
    sd = torch.load(os.path.join(path, f"lora_stage{info.stage}.pt"), map_location="cpu")
    params = {canonical_name(n, info.layer_ids): p for n, p in stage.named_parameters() if p.requires_grad}
    if set(params) != set(sd):
        raise RuntimeError(f"pipeline resume: checkpoint {path} does not match this stage layout "
                           f"({len(sd)} saved tensors vs {len(params)}); same stages/model/LoRA needed")
    with torch.no_grad():
        for k, p in params.items():
            p.copy_(sd[k].to(device=p.device, dtype=p.dtype))
    opt_path = os.path.join(path, f"opt_stage{info.stage}.pt")
    if os.path.exists(opt_path):
        st = torch.load(opt_path, map_location=device)
        optimizer.load_state_dict(st["optimizer"])
        if st.get("scheduler") is not None:
            scheduler.load_state_dict(st["scheduler"])
    elif rank == 0:
        logger.warning(f"pipeline resume: no optimizer state in {path}; weights only")
    if rank == 0:
        logger.info(f"pipeline: resumed from {path} (step {step})")
    return step


@torch.no_grad()
def _run_eval_pipeline(stage: Stage, info: PipelineInfo, val_dataset, cfg, device, rank: int,
                       dtype: torch.dtype) -> float | None:
    """Forward-only pass of up to eval_micro_batches micro-batches per pipeline copy.

    Micro-batches go through the stages one at a time (send/recv between neighbours): eval is
    short, so the pipeline bubble does not matter and no schedule/backward machinery is used.
    Every rank pulls from its validation shard in lockstep (as in training, so the shared
    stream never waits on a stopped consumer); the first stage's batch is the pipeline's and
    its "more data" flag is broadcast. Returns the mean loss per micro-batch over all copies
    (same value on every rank), or None if nothing was evaluated.
    """
    rows_mb = int(cfg.training.batch_size)
    seq_len = int(cfg.data.seq_len)
    limit = int(cfg.training.distributed.pipeline.eval_micro_batches)
    s0 = info.stage0_rank
    prev_r = rank - 1 if not info.first else None
    next_r = rank + 1 if not info.last else None
    loss_fn = _make_loss(stage, 1)
    was_training = stage.training
    stage.eval()
    pos_row = torch.arange(seq_len, device=device).expand(rows_mb, -1).reshape(1, -1).contiguous()
    loader = iter(val_dataset.iter_torch_batches(
        batch_size=rows_mb, drop_last=True, device=device,
        dtypes={"input_ids": torch.long, "attention_mask": torch.long, "labels": torch.long}))
    total, n = 0.0, 0
    while n < limit:
        try:
            batch = next(loader)
            flag = torch.ones(1, device=device, dtype=torch.long)
        except StopIteration:
            batch, flag = None, torch.zeros(1, device=device, dtype=torch.long)
        dist.broadcast(flag, src=s0, group=info.pp_group)
        if not flag.item():
            break
        ids = torch.empty(rows_mb, seq_len, device=device, dtype=torch.long)
        lab = torch.empty_like(ids)
        if info.first:
            ids.copy_(batch["input_ids"]); lab.copy_(batch["labels"])
        dist.broadcast(ids, src=s0, group=info.pp_group)
        dist.broadcast(lab, src=s0, group=info.pp_group)
        if info.first:
            x = ids
        else:
            x = torch.empty(1, rows_mb * seq_len, info.hidden_size, device=device, dtype=dtype)
            dist.recv(x, src=prev_r, group=info.pp_group)
        h = stage(x, pos_row)
        if info.last:
            total += float(loss_fn(h, lab))
        else:
            dist.send(h.contiguous(), dst=next_r, group=info.pp_group)
        n += 1
    stage.train(was_training)
    agg = torch.tensor([total, float(n) if info.last else 0.0], device=device)
    dist.all_reduce(agg)
    return None if agg[1].item() == 0 else float(agg[0] / agg[1])


def _sync_lora_init(lora: list, info: PipelineInfo) -> None:
    """LoRA-A matrices start random, and each rank seeds differently: without this the pipeline
    copies would train different adapters while averaging their gradients."""
    if info.copies > 1:
        for p in lora:
            dist.broadcast(p.data, src=info.stage, group=info.dp_group)


def train_pipeline(*, stage: Stage, info: PipelineInfo, dataset, cfg, val_dataset=None, rank: int, local_rank: int,
                   tracker=None, world_size: int = 1, startup_watchdog=None) -> None:
    device = torch.device(f"cuda:{local_rank}")
    train_cfg, pcfg = cfg.training, cfg.training.distributed.pipeline
    m, rows_mb = int(pcfg.microbatches), int(train_cfg.batch_size)
    rows = m * rows_mb
    seq_len = int(cfg.data.seq_len)
    packed = bool(getattr(cfg.data, "runtime_packing", False))
    if rows_mb > 1 and (cfg.model.attn_impl or "flash_attention_2") != "flash_attention_2":
        raise ValueError("pipeline with batch_size > 1 flattens each micro-batch into one row and "
                         "needs flash_attention_2 to keep its sequences apart")
    eval_steps = int(getattr(train_cfg, "eval_steps", 0) or 0)
    if eval_steps > 0 and val_dataset is None and rank == 0:
        logger.warning("pipeline: eval_steps set but data.val_tokenized_path is not; eval skipped")
    act_dtype = _DTYPES.get(getattr(cfg.model, "dtype", "bfloat16"), torch.bfloat16)

    optimizer = build_optimizer(stage, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    lora = [p for p in stage.parameters() if p.requires_grad]
    _sync_lora_init(lora, info)
    n_train = torch.tensor(sum(p.numel() for p in lora), device=device)
    dist.all_reduce(n_train, group=info.pp_group)

    metrics_cfg = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    throughput = ThroughputTracker(
        params_for_flops=info.total_params,
        world_size=world_size,
        peak_flops_per_gpu=float(getattr(metrics_cfg, "peak_tflops_per_gpu", 312.0)) * 1e12,
        window=int(getattr(metrics_cfg, "throughput_window", 50)),
        warmup_steps=int(getattr(metrics_cfg, "throughput_warmup", 5)),
        trainable_params=int(n_train.item()),
        num_layers=info.num_layers,
        hidden_size=info.hidden_size,
        recompute=bool(pcfg.activation_checkpointing),
    )

    pstage = PipelineStage(stage, stage_index=info.stage, num_stages=info.stages,
                           device=device, group=info.pp_group)
    if pcfg.schedule not in _SCHEDULES:
        raise ValueError(f"pipeline.schedule={pcfg.schedule!r}; supported: {sorted(_SCHEDULES)}")
    sched = _SCHEDULES[pcfg.schedule](pstage, n_microbatches=m, loss_fn=_make_loss(stage, m))

    dtypes = {"input_ids": torch.long, "attention_mask": torch.long, "labels": torch.long,
              **({"position_ids": torch.long} if packed else {})}
    grad_clip = getattr(train_cfg, "grad_clip", None) or getattr(train_cfg, "max_grad_norm", 1.0)
    ckpt = cfg.checkpoint
    step, epoch = 0, 0
    if ckpt.enabled:
        step = _resume(stage, info, optimizer, scheduler, ckpt.save_dir, rank, device)
    if startup_watchdog is not None:
        startup_watchdog.phase("first batch + first step")
    throughput.reset_step_clock()

    while step < train_cfg.max_steps:
        loader = dataset.iter_torch_batches(
            batch_size=rows, drop_last=True, dtypes=dtypes, device=device,
            prefetch_batches=cfg.data.prefetch_batches,
            local_shuffle_buffer_size=max(rows * cfg.data.local_shuffle_buffer_multiplier, 1000),
            local_shuffle_seed=train_cfg.seed + rank,
        )
        for batch in loader:
            t0 = time.perf_counter()
            # 1. the first stage's batch is the pipeline's batch
            ids, labels = batch["input_ids"].contiguous(), batch["labels"].contiguous()
            if packed:
                pos = batch["position_ids"].contiguous()
            else:
                pos = torch.arange(seq_len, device=device).expand(rows, -1).contiguous()
            mask_tokens = batch["attention_mask"].sum().to(torch.long).reshape(1)
            for t in (ids, labels, pos, mask_tokens):
                dist.broadcast(t, src=info.stage0_rank, group=info.pp_group)
            # 2. per-micro-batch positions for every stage, flattened like the packed DDP path
            stage.set_positions([c.reshape(1, -1) for c in pos.chunk(m)])
            # 3. pipeline forward/backward
            losses: list = []
            if info.first:
                sched.step(ids)
            elif info.last:
                sched.step(target=labels, losses=losses)
            else:
                sched.step()
            t1 = time.perf_counter()
            # 4. average LoRA grads across copies, clip over all stages, step
            if info.copies > 1:
                flat = torch._utils._flatten_dense_tensors([p.grad for p in lora])
                dist.all_reduce(flat, group=info.dp_group)
                flat /= info.copies
                for p, g in zip(lora, torch._utils._unflatten_dense_tensors(flat, [p.grad for p in lora])):
                    p.grad.copy_(g)
            t2 = time.perf_counter()
            sq = torch.stack([p.grad.float().pow(2).sum() for p in lora]).sum()
            dist.all_reduce(sq, group=info.pp_group)
            grad_norm = float(sq.sqrt().item())
            if math.isfinite(grad_norm):
                if grad_norm > grad_clip:
                    torch._foreach_mul_([p.grad for p in lora], grad_clip / (grad_norm + 1e-6))
                optimizer.step()
            elif rank == 0:
                logger.warning(f"[step {step}] non-finite grad norm; update skipped")
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            loss = torch.tensor(sum(float(v) for v in losses) if info.last else 0.0, device=device)
            dist.all_reduce(loss, group=info.pp_group)
            dist.all_reduce(loss, group=info.dp_group)
            loss_val = loss.item() / info.copies
            step += 1
            # Throughput counts per GPU: a pipeline's tokens are shared by its `stages` GPUs.
            tm = throughput.step(
                tokens_in_step=float(mask_tokens.item()) / info.stages,
                samples_in_step=rows / info.stages,
                seq_len=seq_len,
                processed_tokens_in_step=rows * seq_len / info.stages,
            )
            if startup_watchdog is not None and step == 1:
                startup_watchdog.done()
            if rank == 0:
                lr = scheduler.get_last_lr()[0]
                msg = f"[epoch {epoch}][step {step}] loss={loss_val:.4f} lr={lr:.2e} grad_norm={grad_norm:.4f}"
                for key, label, fmt in (("throughput/tokens_per_sec", "tok/s", ".0f"),
                                        ("throughput/mfu_pct", "MFU", ".1f"),
                                        ("throughput/hfu_pct", "HFU", ".1f"),
                                        ("throughput/padding_pct", "pad", ".0f"),
                                        ("throughput/step_time_s", "step_s", ".2f"),
                                        ("throughput/mfu_step_pct", "MFU_step", ".1f")):
                    if key in tm:
                        unit = "%" if label in ("MFU", "HFU", "pad", "MFU_step") else ""
                        msg += f" {label}={tm[key]:{fmt}}{unit}"
                msg += f" | pipeline={t1 - t0:.2f}s dp_sync={t2 - t1:.2f}s"
                logger.info(msg)
                if tracker:
                    tracker.log({"loss": loss_val, "lr": lr, "grad_norm": grad_norm, "epoch": float(epoch),
                                 "time/pipeline_s": t1 - t0, "time/dp_sync_s": t2 - t1,
                                 "cuda/max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                 **tm}, step=step)
            if val_dataset is not None and eval_steps > 0 and step % eval_steps == 0:
                te = time.perf_counter()
                eval_loss = _run_eval_pipeline(stage, info, val_dataset, cfg, device, rank, act_dtype)
                if rank == 0:
                    if eval_loss is None:
                        logger.warning(f"[epoch {epoch}][step {step}] eval skipped: no validation batches")
                    else:
                        logger.info(f"[epoch {epoch}][step {step}] eval_loss={eval_loss:.4f} "
                                    f"({time.perf_counter() - te:.1f}s)")
                        if tracker:
                            tracker.log({"eval/loss": eval_loss}, step=step)
                throughput.reset_step_clock()
            if ckpt.enabled and ckpt.save_interval > 0 and step % ckpt.save_interval == 0:
                _save_lora(stage, info, ckpt.save_dir, step, rank, optimizer, scheduler)
                throughput.reset_step_clock()
            if step >= train_cfg.max_steps:
                break
        epoch += 1
    if ckpt.enabled:
        _save_lora(stage, info, ckpt.save_dir, step, rank, optimizer, scheduler)
    dist.barrier()
