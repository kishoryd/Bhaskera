"""
bhaskera.trainer.loop
=====================
Pure training loop.


"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import pickle
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from bhaskera.introspect import ModelProfile
from bhaskera.utils import ThroughputTracker
from bhaskera.utils.system_stats import system_stats, cuda_memory_stats

from .checkpointing import maybe_resume, save_and_prune
from .moe import compute_expert_utilization, extract_aux_loss
from .optim import build_optimizer, build_scheduler
from .precision import resolve_autocast_dtype

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FSDP2 + DDP gradient-sync helper
# ---------------------------------------------------------------------------

def _set_grad_sync(model: torch.nn.Module, enabled: bool) -> None:
    """
    Toggle gradient all-reduce for the wrapped model.

    Dispatches by wrapper type:

        FSDP2  → set_requires_gradient_sync(model, enabled)
                 Walks every sharded submodule regardless of any PEFT /
                 wrapper layers sitting on top.  set_requires_gradient_sync
                 bypasses wrapper __getattr__ chains and operates directly
                 on the FSDP2 submodule state.

        DDP    → model.require_backward_grad_sync = enabled
                 This is the underlying flag that DDP.no_sync() toggles.
                 Setting it directly avoids the no_sync() context-manager
                 dance (and works even when DDP is wrapping a PEFT model
                 that doesn't delegate the no_sync attribute).
                 SKIPPED when static_graph=True — DDP's static_graph mode
                 caches the reduction order on the first iteration and is
                 incompatible with mid-run sync toggling.

        Other  → no-op (single-GPU / CPU / mixed contexts).

    Call with enabled=False before every micro-step except the last,
    and enabled=True on the last micro-step so the all-reduce fires
    exactly once per optimizer step.
    """
    # ── FSDP2 path ──────────────────────────────────────────────────
    try:
        from torch.distributed._composable.fsdp import (
            FSDPModule,
            set_requires_gradient_sync,
        )
        if isinstance(model, FSDPModule) or any(
            isinstance(m, FSDPModule) for m in model.modules()
        ):
            set_requires_gradient_sync(model, enabled)
            return
    except ImportError:
        # torch < 2.4 — FSDP2 unavailable, fall through to DDP / no-op.
        pass

    # ── DDP path ────────────────────────────────────────────────────
    if isinstance(model, DDP):
        # static_graph=True bakes the reduction graph into DDP after the
        # first iteration; toggling require_backward_grad_sync after that
        # raises "Your training graph has changed in this iteration".
        if getattr(model, "_bhaskera_static_graph", False) or getattr(
            model, "static_graph", False
        ):
            return
        model.require_backward_grad_sync = enabled
        return

    # ── Non-distributed: nothing to do ──────────────────────────────


def _flops_params(model: torch.nn.Module, profile: ModelProfile, total_params: int) -> int:
    """
    Effective parameter count for ThroughputTracker's 6*N*D FLOPs estimate.

    For a dense model every parameter runs on every token, so N = total_params.
    For MoE that overstates FLOPs — and understates MFU — by roughly
    num_experts / (experts_per_token + num_shared_experts), since only a
    subset of routed experts (plus any always-on shared experts) actually
    execute per token. Observed on Qwen3-30B-A3B (128 experts, top-8): using
    total_params reports MFU ~10x too low, exactly the same class of error as
    the earlier /world_size double-division bug — a meter that silently
    reads wrong rather than raising, so it looks trustworthy until compared
    against a hand computation.

    profile.expert_modules (from introspect.py's _find_moe_components) is a
    flat list mixing routed *and* shared expert instances — the two aren't
    tracked separately at the module level — so both the active and total
    instance counts below include num_shared_experts on top of the routed
    figures, keeping numerator and denominator on the same basis.

    Falls back to total_params (the old, dense-shaped behaviour) whenever
    introspection couldn't determine the routing shape, rather than guessing
    a fraction from incomplete data.
    """
    if not profile.is_moe or not profile.expert_modules:
        return total_params

    num_experts   = int(profile.num_experts)       # routed experts/layer (config-derived)
    active_routed = int(profile.experts_per_token)  # top-k routed experts/token
    num_shared    = max(0, int(profile.num_shared_experts))

    if num_experts <= 0 or active_routed <= 0:
        return total_params

    active_routed = min(active_routed, num_experts)
    total_instances  = num_experts + num_shared
    active_instances = active_routed + num_shared

    expert_params = sum(p.numel() for m in profile.expert_modules for p in m.parameters())
    non_expert_params = total_params - expert_params
    if non_expert_params < 0:
        # expert_modules double-counted something (e.g. a module reachable
        # under two names) — don't propagate a bad estimate.
        return total_params

    return non_expert_params + expert_params * active_instances // total_instances


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train(
    *,
    model: torch.nn.Module,
    dataset,
    cfg,
    profile: ModelProfile,
    rank: int,
    local_rank: int,
    tracker=None,
    world_size: int = 1,
    val_dataset=None,
    startup_watchdog=None,
) -> None:
    """
    Run the training loop.

    Args:
        model:       Distributed-wrapped model (FSDP2 or DDP).
        dataset:     Ray Dataset pre-tokenized by bhaskera.data.
        cfg:         Bhaskera Config object.
        profile:     ModelProfile from introspection.
        rank:        Global rank of this worker.
        local_rank:  Local GPU index on this host.
        tracker:     Optional logger (MultiLogger from build_logger).
        world_size:  Total number of training ranks.
        val_dataset: Optional Ray Dataset shard for periodic eval /
                     MLPerf-style convergence early-stop (cfg.training.eval_steps,
                     target_eval_loss). None if data.val_tokenized_path isn't set.
    """
    device    = torch.device(f"cuda:{local_rank}")
    train_cfg = cfg.training
    ckpt_cfg  = cfg.checkpoint

    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    model.train()
    step = 0
    if ckpt_cfg.enabled:
        if startup_watchdog is not None:
            startup_watchdog.phase("resume checkpoint")
        step = maybe_resume(model, optimizer, ckpt_cfg.save_dir)
        model.train()
    best_ckpts: list[tuple[float, str]] = []
    grad_skip_count = 0  # consecutive non-finite-grad_norm steps; see max_grad_skip_steps

    # ── Throughput / MFU tracker ────────────────────────────────────
    metrics_cfg    = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    throughput_on  = bool(getattr(metrics_cfg, "throughput", True)) if metrics_cfg else True
    peak_tflops    = float(getattr(metrics_cfg, "peak_tflops_per_gpu", 312.0)) if metrics_cfg else 312.0

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_for_flops = _flops_params(model, profile, total_params) or trainable_params

    throughput = ThroughputTracker(
        params_for_flops=params_for_flops,
        world_size=max(1, int(world_size)),
        peak_flops_per_gpu=peak_tflops * 1e12,
        window=int(getattr(metrics_cfg, "throughput_window", 50)) if metrics_cfg else 50,
        warmup_steps=int(getattr(metrics_cfg, "throughput_warmup", 5)) if metrics_cfg else 5,
    ) if throughput_on else None

    if tracker:
        tracker.log({
            "model/total_params":     float(total_params),
            "model/trainable_params": float(trainable_params),
            "model/active_params":    float(params_for_flops),
            "model/world_size":       float(world_size),
        }, step=0)

    for epoch in range(train_cfg.num_epochs):
        step, best_ckpts, stop_early, grad_skip_count = _run_epoch(
            model=model, dataset=dataset, val_dataset=val_dataset,
            optimizer=optimizer,
            scheduler=scheduler, cfg=cfg, profile=profile,
            rank=rank, local_rank=local_rank, device=device,
            epoch=epoch, step=step, tracker=tracker,
            best_ckpts=best_ckpts, throughput=throughput,
            world_size=world_size, grad_skip_count=grad_skip_count,
            startup_watchdog=startup_watchdog,
        )
        if step >= train_cfg.max_steps or stop_early:
            break

    if tracker:
        tracker.finish()
    if rank == 0:
        logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Validation (periodic eval + MLPerf-style convergence early-stop)
# ---------------------------------------------------------------------------

def _run_eval(
    *, model, val_dataset, cfg, device, autocast_dtype, use_autocast,
) -> Optional[float]:
    """
    One full pass over this rank's local validation shard, no backward.

    Returns the cross-rank sample-weighted mean loss (identical value on
    every rank, via all_reduce) so every rank compares against
    target_eval_loss with the same number and either all stop or all
    continue — a per-rank-only decision would desync the collectives on
    the next training step. Returns None if no rank saw any eval batches
    (e.g. an empty val shard), rather than lying with a 0.0 loss that
    would trivially — and falsely — satisfy any early-stop target.
    """
    train_cfg = cfg.training
    eval_bs   = train_cfg.eval_batch_size or train_cfg.batch_size

    was_training = model.training
    model.eval()

    val_loader = val_dataset.iter_torch_batches(
        batch_size=eval_bs,
        drop_last=False,
        dtypes={
            "input_ids":      torch.long,
            "attention_mask": torch.long,
            "labels":         torch.long,
        },
        device=device,
    )

    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in val_loader:
            autocast_ctx = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if use_autocast
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
            total_loss += out.loss.item()
            n_batches  += 1

    if was_training:
        model.train()

    if dist.is_available() and dist.is_initialized():
        sum_and_count = torch.tensor(
            [total_loss, float(n_batches)], device=device, dtype=torch.float32,
        )
        dist.all_reduce(sum_and_count, op=dist.ReduceOp.SUM)
        total_loss_all, n_batches_all = sum_and_count.tolist()
    else:
        total_loss_all, n_batches_all = total_loss, float(n_batches)

    if n_batches_all == 0:
        return None

    return total_loss_all / n_batches_all


def _log_oom_breakdown(snapshot: dict, dump_path: str) -> None:
    """TEMPORARY diagnostic (see BHASKERA_PROFILE_OOM above). Aggregates
    live allocated blocks by first non-torch-internal frame in their
    allocation call stack, largest first -- answers "what line of code is
    holding the memory", not just "how much is used"."""
    by_stack: dict = {}
    total_active = 0
    for seg in snapshot.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            size = blk.get("size", 0)
            total_active += size
            site = "unknown"
            for f in blk.get("frames", []):
                fname = f.get("filename", "")
                if any(k in fname for k in ("bhaskera", "transformers", "peft", "liger_kernel")):
                    site = f"{fname.split('/')[-1]}:{f.get('line', '?')} in {f.get('name', '?')}"
                    break
            by_stack.setdefault(site, [0, 0])
            by_stack[site][0] += size
            by_stack[site][1] += 1

    ranked = sorted(by_stack.items(), key=lambda kv: -kv[1][0])
    logger.error(f"OOM allocation breakdown (snapshot: {dump_path}, "
                 f"total active: {total_active/1e9:.2f}GB):")
    for site, (size, count) in ranked[:25]:
        logger.error(f"  {size/1e9:7.3f}GB  ({count:5d} allocs)  {site}")


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    *, model, dataset, val_dataset, optimizer, scheduler, cfg, profile,
    rank, local_rank, device, epoch, step, tracker, best_ckpts,
    throughput: Optional[ThroughputTracker], world_size: int,
    grad_skip_count: int = 0,
    startup_watchdog=None,
):
    train_cfg  = cfg.training
    ckpt_cfg   = cfg.checkpoint
    grad_accum = train_cfg.grad_accum

    strategy       = cfg.training.distributed.strategy.lower()
    autocast_dtype = resolve_autocast_dtype(cfg, profile)
    use_autocast   = (strategy == "ddp" and device.type == "cuda")

    moe_cfg         = getattr(cfg, "moe", None)
    aux_loss_weight = getattr(moe_cfg, "aux_loss_weight", 0.01) if moe_cfg else 0.01
    log_expert_util = (
        profile.is_moe and moe_cfg is not None
        and getattr(moe_cfg, "log_expert_utilization", True)
    )
    expert_log_every = getattr(moe_cfg, "log_every_n_steps", 10) if moe_cfg else 10

    metrics_cfg = getattr(getattr(cfg, "monitoring", None), "metrics", None)
    sys_every   = int(getattr(metrics_cfg, "system_every_n_steps", 10)) if metrics_cfg else 10
    cuda_every  = int(getattr(metrics_cfg, "cuda_every_n_steps", 10))   if metrics_cfg else 10
    sys_on      = bool(getattr(metrics_cfg, "enabled", True))           if metrics_cfg else True

    if startup_watchdog is not None:
        startup_watchdog.phase("first batch + first step")

    # ── Data loader ─────────────────────────────────────────────────
    # fix #16: use prefetch_batches from config
    # fix loop: drop_last=True prevents shape mismatches in the last batch
    # fix loop: explicit dtypes — int64 is required for CUDA embedding lookup
    loader = dataset.iter_torch_batches(
        batch_size=train_cfg.batch_size,
        local_shuffle_buffer_size=max(
            train_cfg.batch_size * cfg.data.local_shuffle_buffer_multiplier, 1000
        ),
        local_shuffle_seed=train_cfg.seed + rank,
        prefetch_batches=cfg.data.prefetch_batches,   # fix #16
        drop_last=True,                               # fix loop: no shape mismatch
        dtypes={                                      # fix loop: int64 for embeddings
            "input_ids":      torch.long,
            "attention_mask": torch.long,
            "labels":         torch.long,
        },
        device=device,
    )

    # Per-step accumulators
    epoch_loss     = 0.0
    epoch_aux_loss = 0.0
    epoch_steps    = 0

    # Loss EMA (for spike detection)
    loss_ema: Optional[float] = None
    loss_ema_alpha = 0.05

    # Tokens/samples per accum window — for throughput
    window_tokens  = 0
    window_samples = 0
    window_seq_len = 0

    optimizer.zero_grad(set_to_none=True)

    if throughput is not None:
        throughput.reset_step_clock()

    # ── Training loop ───────────────────────────────────────────────
    # Gradient-sync invariant (non-negotiable):
    #   All-reduce must fire exactly ONCE per optimizer step — on the last
    #   micro-step of every accumulation window.  For FSDP2 we use
    #   set_requires_gradient_sync(); for DDP we toggle
    #   require_backward_grad_sync directly.  Both are routed through
    #   _set_grad_sync() above, which dispatches by wrapper type and
    #   safely falls back when neither applies (single-GPU, static_graph DDP).
    loader_iter = iter(loader)
    stop_early  = False

    while step < train_cfg.max_steps and not stop_early:
        micro_losses:     list[torch.Tensor] = []
        micro_aux_losses: list[torch.Tensor] = []
        window_tokens  = 0
        window_samples = 0

        # ── Gradient accumulation loop ───────────────────────────────
        for micro_step in range(grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                # Epoch exhausted mid-accumulation window — stop cleanly
                loader_iter = None  # type: ignore[assignment]
                break

            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels         = batch["labels"]

            try:
                window_tokens  += int(attention_mask.sum().item())
                window_samples += int(input_ids.size(0))
                window_seq_len  = int(input_ids.size(1))
            except Exception:
                pass

            is_last = (micro_step == grad_accum - 1)

            # Gradient-sync toggle:
            #   False  → suppress all-reduce on this micro-step
            #   True   → allow all-reduce on the final micro-step
            # Dispatches to FSDP2's set_requires_gradient_sync,
            # DDP's require_backward_grad_sync, or no-op as appropriate.
            _set_grad_sync(model, enabled=is_last)

            autocast_ctx = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if use_autocast
                else contextlib.nullcontext()
            )

            with autocast_ctx:
                forward_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                if profile.is_moe and profile.has_aux_loss:
                    forward_kwargs["output_router_logits"] = True

                # TEMPORARY diagnostic for the Qwen3-235B-A22B forward+backward
                # OOM (~78GB, invariant across world_size/expert-sharding/
                # seq_len -- see engineering report §7.3). Env-gated, no-op
                # unless BHASKERA_PROFILE_OOM=1; dumps an allocation-site
                # breakdown on OOM instead of just crashing. Safe to remove
                # once that issue is resolved.
                _profile_oom = os.environ.get("BHASKERA_PROFILE_OOM") == "1" and step == 0 and micro_step == 0
                if _profile_oom:
                    torch.cuda.memory._record_memory_history(max_entries=200_000)

                try:
                    out       = model(**forward_kwargs)
                    main_loss = out.loss
                    aux_loss  = extract_aux_loss(out, profile)

                    if aux_loss is not None:
                        total_loss = (main_loss + aux_loss_weight * aux_loss) / grad_accum
                    else:
                        total_loss = main_loss / grad_accum

                    total_loss.backward()
                except torch.OutOfMemoryError:
                    if _profile_oom:
                        snap = torch.cuda.memory._snapshot()
                        path = f"/tmp/oom_snapshot_rank{dist.get_rank() if dist.is_initialized() else 0}.pickle"
                        with open(path, "wb") as f:
                            pickle.dump(snap, f)
                        _log_oom_breakdown(snap, path)
                    raise
                finally:
                    if _profile_oom:
                        torch.cuda.memory._record_memory_history(enabled=None)

            micro_losses.append(main_loss.detach())
            if aux_loss is not None:
                micro_aux_losses.append(aux_loss.detach())

        # If loader exhausted mid-window, break the outer loop too
        if loader_iter is None:
            break

        # Ensure sync is re-enabled after the accumulation window in case
        # something above exited early (e.g. StopIteration on last micro-step).
        _set_grad_sync(model, enabled=True)

        # A full forward/backward window completed: startup is over even if
        # this step's grads turn out non-finite and the update is skipped.
        if startup_watchdog is not None:
            startup_watchdog.done()

        # ── Optimizer step ──────────────────────────────────────────
        # Use model.clip_grad_norm_ for FSDP2 if available (handles sharded grads).
        # DDP doesn't expose clip_grad_norm_; falls through to the manual clip.
        grad_clip = getattr(train_cfg, "grad_clip", None) or getattr(train_cfg, "max_grad_norm", 1.0)
        if hasattr(model, "clip_grad_norm_"):
            grad_norm = model.clip_grad_norm_(grad_clip).item()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                grad_clip,
            ).item()

        if not math.isfinite(grad_norm):
            grad_skip_count += 1
            max_skips = getattr(train_cfg, "max_grad_skip_steps", 100)
            logger.warning(
                f"[rank {rank}][epoch {epoch}][step {step}] "
                f"Non-finite grad_norm={grad_norm} — skipping optimizer step "
                f"({grad_skip_count}/{max_skips} consecutive skips)"
            )
            optimizer.zero_grad(set_to_none=True)
            if tracker:
                tracker.log({"train/non_finite_grad": 1.0}, step=step)
            if max_skips > 0 and grad_skip_count > max_skips:
                # Previously this field was parsed from config and never
                # consulted anywhere — a run could skip every single step
                # forever, burning the full node allocation while silently
                # training on nothing. max_grad_skip_steps now bounds how
                # many *consecutive* non-finite steps are tolerated before
                # surfacing loudly instead of quietly wasting the rest of
                # the wall-clock budget.
                raise RuntimeError(
                    f"[rank {rank}] {grad_skip_count} consecutive non-finite "
                    f"grad_norm steps, exceeding max_grad_skip_steps={max_skips}. "
                    "Training is not making progress — check for a bad LR, a "
                    "corrupt batch, or numerical instability before resuming."
                )
            continue
        grad_skip_count = 0  # any finite step resets the consecutive counter

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # ── Loss aggregation ────────────────────────────────────────
        window_loss = torch.stack(micro_losses).mean().item()
        window_aux  = (
            torch.stack(micro_aux_losses).mean().item()
            if micro_aux_losses else 0.0
        )

        # Loss EMA + spike ratio
        if loss_ema is None:
            loss_ema = window_loss
        else:
            loss_ema = (1 - loss_ema_alpha) * loss_ema + loss_ema_alpha * window_loss
        loss_spike = (window_loss / loss_ema) if loss_ema > 0 else 1.0

        lr = scheduler.get_last_lr()[0]
        epoch_loss     += window_loss
        epoch_aux_loss += window_aux
        epoch_steps    += 1
        step           += 1

        # ── Throughput ──────────────────────────────────────────────
        throughput_metrics: dict[str, float] = {}
        if throughput is not None:
            throughput_metrics = throughput.step(
                tokens_in_step=window_tokens,
                samples_in_step=window_samples,
                seq_len=window_seq_len,
            )

        # ── Logging ────────────────────────────────────────────────
        if rank == 0:
            msg = (
                f"[epoch {epoch}][step {step}] loss={window_loss:.4f} "
                f"lr={lr:.2e} grad_norm={grad_norm:.4f}"
            )
            if "throughput/tokens_per_sec" in throughput_metrics:
                msg += f" tok/s={throughput_metrics['throughput/tokens_per_sec']:.0f}"
            if "throughput/mfu_pct" in throughput_metrics:
                msg += f" MFU={throughput_metrics['throughput/mfu_pct']:.1f}%"
            logger.info(msg)

            metrics: dict[str, float] = {
                "loss":             window_loss,
                "lr":               lr,
                "grad_norm":        grad_norm,
                "epoch":            float(epoch),
                "loss_running_avg": loss_ema,
                "loss_spike_ratio": loss_spike,
            }
            if profile.is_moe:
                metrics["aux_loss"]   = window_aux
                metrics["total_loss"] = window_loss + aux_loss_weight * window_aux
            metrics.update(throughput_metrics)

            if tracker:
                if log_expert_util and step % expert_log_every == 0:
                    metrics.update(compute_expert_utilization(out, profile))
                tracker.log(metrics, step=step)

        if tracker and sys_on and sys_every > 0 and step % sys_every == 0:
            sysm: dict[str, float] = {}
            sysm.update(system_stats(
                gpu=bool(getattr(metrics_cfg, "gpu", True)) if metrics_cfg else True,
                cpu=bool(getattr(metrics_cfg, "cpu", True)) if metrics_cfg else True,
            ))
            if cuda_every > 0 and step % cuda_every == 0:
                if not metrics_cfg or getattr(metrics_cfg, "cuda_memory", True):
                    sysm.update(cuda_memory_stats(device))
            if sysm:
                tracker.log(sysm, step=step)

        # ── Periodic eval / MLPerf-style convergence early-stop ──────
        eval_steps = train_cfg.eval_steps
        if val_dataset is not None and eval_steps > 0 and step % eval_steps == 0:
            eval_loss = _run_eval(
                model=model, val_dataset=val_dataset, cfg=cfg,
                device=device, autocast_dtype=autocast_dtype,
                use_autocast=use_autocast,
            )
            if eval_loss is None:
                if rank == 0:
                    logger.warning(
                        f"[epoch {epoch}][step {step}] eval skipped: "
                        "no rank saw any validation batches"
                    )
            else:
                if rank == 0:
                    logger.info(f"[epoch {epoch}][step {step}] eval_loss={eval_loss:.4f}")
                    if tracker:
                        tracker.log({"eval/loss": eval_loss}, step=step)
                target = train_cfg.target_eval_loss
                if target is not None and eval_loss <= target:
                    stop_early = True
                    if rank == 0:
                        logger.info(
                            f"eval_loss {eval_loss:.4f} <= target_eval_loss {target} "
                            "— stopping early (convergence target reached)"
                        )

    if epoch_steps == 0:
        return step, best_ckpts, stop_early, grad_skip_count

    avg_loss = epoch_loss / epoch_steps
    if rank == 0:
        epoch_msg     = f"[epoch {epoch}] avg_loss={avg_loss:.4f}"
        epoch_metrics = {"epoch_avg_loss": avg_loss, "epoch": epoch}
        if profile.is_moe:
            avg_aux = epoch_aux_loss / epoch_steps
            epoch_msg += f" avg_aux_loss={avg_aux:.4f}"
            epoch_metrics["epoch_avg_aux_loss"] = avg_aux
        logger.info(epoch_msg)
        if tracker:
            tracker.log(epoch_metrics, step=step)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if ckpt_cfg.enabled and (epoch + 1) % ckpt_cfg.save_interval == 0:
        best_ckpts = save_and_prune(
            model=model, optimizer=optimizer, step=step,
            avg_loss=avg_loss, ckpt_cfg=ckpt_cfg,
            rank=rank, best_ckpts=best_ckpts,
        )

    return step, best_ckpts, stop_early, grad_skip_count
