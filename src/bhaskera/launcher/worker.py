"""
bhaskera.launcher.worker
========================
Per-GPU entry point.  Called by Ray Train's TorchTrainer for each
actor and also directly by raw SLURM workers.


"""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import ray.train
import torch

from bhaskera.config import Config
from bhaskera.distributed import wrap_model
from bhaskera.distributed.wrap import _verify_all_ranks_live, _wait_for_dist
from bhaskera.models import build_model
from bhaskera.trainer import train
from bhaskera.utils import build_logger
from bhaskera.utils.startup_timer import mark as startup_mark
from bhaskera.utils.watchdog import StartupWatchdog

logger = logging.getLogger(__name__)


def worker_fn(cfg_dict: dict) -> None:
    """Entry point for a single GPU worker."""
    cfg = Config.from_dict(cfg_dict)

    ray_ctx    = ray.train.get_context()
    local_rank = ray_ctx.get_local_rank()
    rank       = ray_ctx.get_world_rank()
    world_size = ray_ctx.get_world_size()

    watchdog = StartupWatchdog(
        getattr(cfg.training, "startup_timeout_s", 0), rank
    ).start()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    watchdog.phase("distributed init")
    _wait_for_dist()
    _verify_all_ranks_live(rank, world_size, device)

    _seed_everything(cfg.training.seed, rank, cfg.training.deterministic)

    logger.info(f"[rank {rank}/{world_size}] GPU {local_rank} ready")
    startup_mark("dist_ready", rank)

    dataset = ray.train.get_dataset_shard("train")
    # "val" is only present in the TorchTrainer datasets dict when
    # data.val_tokenized_path was configured (see launcher/train.py) —
    # gate on the same condition here rather than probing get_dataset_shard
    # for a KeyError.
    val_dataset = (
        ray.train.get_dataset_shard("val")
        if getattr(cfg.data, "val_tokenized_path", None)
        else None
    )

    strategy    = cfg.training.distributed.strategy.lower()
    load_device = torch.device("cpu") if strategy == "fsdp" else device

    watchdog.phase("build_model")
    model, profile = build_model(cfg, load_device, rank=rank)
    startup_mark("model_loaded", rank)

    if rank == 0:
        logger.info(
            f"Model profile: moe={profile.is_moe} | "
            f"decoder={profile.decoder_layer_cls.__name__ if profile.decoder_layer_cls else 'None'} | "
            f"experts={len(profile.expert_modules)} | "
            f"dtype={profile.model_dtype} | "
            f"has_aux_loss={profile.has_aux_loss}"
        )

    watchdog.phase("wrap_model")
    model = wrap_model(model, cfg, local_rank, profile)
    startup_mark("model_wrapped", rank)

    # Every rank builds a logger.  build_logger() decides internally
    # which backends to enable for non-rank-0 (Ray-only).
    tracker = build_logger(cfg, rank=rank, world_size=world_size)

    train(
        model=model,
        dataset=dataset,
        val_dataset=val_dataset,
        cfg=cfg,
        profile=profile,
        rank=rank,
        local_rank=local_rank,
        tracker=tracker,
        world_size=world_size,
        startup_watchdog=watchdog,
    )


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _seed_everything(base_seed: int, rank: int, deterministic: bool) -> None:
    """
    Seed PyTorch, NumPy and Python RNGs with a rank-offset seed so each
    rank shuffles its data differently but every run is reproducible.
    """
    seed = int(base_seed) + int(rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not enable deterministic mode: {e}")
