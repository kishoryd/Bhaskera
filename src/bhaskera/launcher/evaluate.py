"""
bhaskera.launcher.evaluate
==========================
bhaskera-eval: benchmark a model on the same Ray + FSDP/DDP path as training,
so a model too large for one node is evaluated sharded across many.

    bhaskera-eval --config configs/eval_qwen3_30b_a3b.yaml --num-workers 16
    bhaskera-eval --config <train config> --checkpoint <save_dir>/step_1000 \\
                  --tasks hellaswag,arc_challenge,bfcl_multiple --limit 500

The config's model / training.distributed / lora sections decide how the
model is built and sharded (use the training config to evaluate a training
checkpoint); its evaluation section picks tasks, data and batch size, and
the flags override it. Results: <evaluation.output_dir>/<run_name>.json.
"""
from __future__ import annotations

import argparse
import logging
import os

import ray
import ray.train
import torch
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from bhaskera.config import Config, load_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def eval_worker_fn(cfg_dict: dict) -> None:
    """Per-GPU entry: build and shard the model exactly as training does,
    optionally load a checkpoint, run the tasks, rank 0 writes results."""
    from transformers import AutoTokenizer

    from bhaskera.distributed import load_model_weights, wrap_model
    from bhaskera.distributed.wrap import _verify_all_ranks_live, _wait_for_dist
    from bhaskera.evaluation import evaluate, write_results
    from bhaskera.models import build_model

    cfg = Config.from_dict(cfg_dict)
    ctx = ray.train.get_context()
    local_rank, rank, world = ctx.get_local_rank(), ctx.get_world_rank(), ctx.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    _wait_for_dist()
    _verify_all_ranks_live(rank, world, device)

    strategy = cfg.training.distributed.strategy.lower()
    load_device = torch.device("cpu") if strategy == "fsdp" else device
    model, profile = build_model(cfg, load_device, rank=rank)
    model = wrap_model(model, cfg, local_rank, profile)

    extra = {"moe": profile.is_moe, "strategy": strategy, "world_size": world}
    ckpt = cfg_dict.get("_eval_checkpoint")
    if ckpt:
        extra["checkpoint"] = ckpt
        extra["checkpoint_step"] = load_model_weights(model, ckpt)
        logger.info(f"[rank {rank}] loaded {ckpt} (step {extra['checkpoint_step']})")

    tok = AutoTokenizer.from_pretrained(
        cfg.model.name, trust_remote_code=getattr(cfg.model, "trust_remote_code", False))
    results = evaluate(model, tok, cfg.evaluation, device)
    if rank == 0:
        path = write_results(results, cfg, cfg.evaluation.output_dir, extra)
        logger.info(f"results -> {path}")


def main() -> None:
    from bhaskera.launcher.monitoring import setup_monitoring
    from bhaskera.launcher.train import _count_gpus, _init_ray

    args = _parse_args()
    cfg = load_config(args.config)
    ev = cfg.evaluation
    if args.tasks:
        ev.tasks = [t for t in args.tasks.split(",") if t]
    if args.limit is not None:
        ev.limit = args.limit
    if args.data_dir:
        ev.data_dir = args.data_dir
    if args.output_dir:
        ev.output_dir = args.output_dir
    if args.batch_size:
        ev.batch_size = args.batch_size
    if not ev.tasks:
        raise SystemExit("no tasks: set evaluation.tasks or pass --tasks")
    ev.output_dir = os.path.abspath(ev.output_dir)
    ev.data_dir = os.path.abspath(ev.data_dir) if ev.data_dir else ev.data_dir
    if args.no_dashboard:
        cfg.monitoring.dashboard = False

    _init_ray(setup_monitoring(cfg))
    num_workers = args.num_workers or _count_gpus()
    cfg_dict = cfg.as_dict()
    if args.checkpoint:
        cfg_dict["_eval_checkpoint"] = os.path.abspath(args.checkpoint)
    logger.info(f"Evaluating {cfg.model.name} on {ev.tasks} with {num_workers} GPU worker(s)")
    TorchTrainer(
        train_loop_per_worker=eval_worker_fn,
        train_loop_config=cfg_dict,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True,
                                     resources_per_worker={"GPU": 1}),
        run_config=RunConfig(name=f"eval-{cfg.logging.run_name}",
                             storage_path=os.path.abspath(cfg.checkpoint.save_dir),
                             failure_config=ray.train.FailureConfig(max_failures=0)),
    ).fit()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bhaskera benchmark evaluation")
    p.add_argument("--config", required=True, help="YAML config (model, sharding, evaluation)")
    p.add_argument("--num-workers", type=int, default=None, help="GPU workers (default: all)")
    p.add_argument("--tasks", default=None, help="comma-separated; overrides evaluation.tasks")
    p.add_argument("--limit", type=int, default=None, help="examples per task; 0 = all")
    p.add_argument("--data-dir", default=None, help="overrides evaluation.data_dir")
    p.add_argument("--output-dir", default=None, help="overrides evaluation.output_dir")
    p.add_argument("--batch-size", type=int, default=None, help="overrides evaluation.batch_size")
    p.add_argument("--checkpoint", default=None, help="a checkpoint dir written by training")
    p.add_argument("--no-dashboard", action="store_true", help="disable the Ray Dashboard")
    return p.parse_args()


if __name__ == "__main__":
    main()
