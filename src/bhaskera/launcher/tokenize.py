"""
bhaskera.launcher.tokenize
==========================
One-shot CLI that tokenizes a dataset, writes to persistent storage, and
prints the config snippet to paste into your training config.

Usage examples
--------------
HF dataset (unchanged):
    bhaskera-tokenize --config configs/tokenize.yaml

Local JSONL with ChatML (your case):
    bhaskera-tokenize --config configs/local_chatml.yaml --split both

Override paths from the CLI:
    bhaskera-tokenize --config configs/local_chatml.yaml \\
        --train-path /data/train.jsonl --val-path /data/val.jsonl \\
        --format chatml --split both

After running, paste the printed YAML snippet into your training config:

    data:
      name: local
      tokenized_path:     "/scratch/cache/local_train_<hash>"
      val_tokenized_path: "/scratch/cache/local_val_<hash>"


"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_BOX_WIDTH = 78


# ---------------------------------------------------------------------------
# Tokenizer prefetch (fix #29 — unchanged)
# ---------------------------------------------------------------------------

def _prefetch_tokenizer(cfg) -> None:
    """Download the tokenizer once in the driver before Ray spawns workers."""
    from transformers import AutoTokenizer

    model_name        = cfg.model.name
    trust_remote_code = getattr(cfg.model, "trust_remote_code", False)

    logger.info(
        f"Prefetching tokenizer for '{model_name}' "
        f"(trust_remote_code={trust_remote_code}) …"
    )

    try:
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, local_files_only=True,
        )
        logger.info(f"Tokenizer already cached locally. vocab_size={tok.vocab_size}")
        return
    except OSError:
        pass
    except Exception as e:
        logger.warning(f"Local tokenizer check raised: {e}. Will attempt download.")

    try:
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, local_files_only=False,
        )
        logger.info(f"Tokenizer downloaded. vocab_size={tok.vocab_size}")
    except Exception as e:
        logger.warning(
            f"Tokenizer prefetch failed: {e}. "
            "Ray workers will retry on their own."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _tokenize_one_split(cfg, dataset_name: str, split: str | None) -> str:
    """
    Tokenize one split and persist to disk. Returns the cache path.

    The persisted cache directory is named ``<dataset><_split>_<hash>`` so
    train and val always end up in different folders.
    """
    from bhaskera.data.registry import call_raw_builder, TEXT_COL
    from bhaskera.data.tokenize import persist_tokenized

    if dataset_name not in TEXT_COL:
        raise ValueError(
            f"No text_col registered for dataset '{dataset_name}'. "
            "Use @register_raw('name', text_col='column_name')."
        )

    logger.info(f"Building raw dataset '{dataset_name}' (split={split!r})…")
    raw_ds   = call_raw_builder(dataset_name, cfg, split=split)
    text_col = TEXT_COL[dataset_name]

    # Suffix the dataset_name with the split so caches don't collide.
    persist_name = f"{dataset_name}_{split}" if split else dataset_name

    cache_path = persist_tokenized(
        ds=raw_ds,
        cfg=cfg,
        text_col=text_col,
        dataset_name=persist_name,
    )
    return cache_path


def main() -> None:
    args = _parse_args()

    # ── Load config ─────────────────────────────────────────────────────
    from bhaskera.config import load_config
    cfg = load_config(args.config)

    # CLI overrides
    if args.dataset:
        cfg.data.name = args.dataset
    if args.storage_path:
        cfg.data.cache_dir = args.storage_path
    if args.overwrite:
        cfg.data.overwrite_cache = True
    if args.num_workers:
        cfg.data.num_workers = args.num_workers
    if args.format:
        cfg.data.format = args.format
    if args.train_path:
        cfg.data.train_path = args.train_path
    if args.val_path:
        cfg.data.val_path = args.val_path

    if not cfg.data.cache_dir:
        logger.error(
            "cfg.data.cache_dir is not set. "
            "Pass --storage-path <dir> or set data.cache_dir in your config."
        )
        sys.exit(1)

    dataset_name = cfg.data.name
    splits       = _resolve_splits(args.split, cfg)

    logger.info(
        f"bhaskera-tokenize | dataset={dataset_name!r} | format={cfg.data.format!r} | "
        f"splits={splits} | config={args.config!r}"
    )

    # ── Prefetch tokenizer BEFORE Ray init (fix #29) ────────────────────
    _prefetch_tokenizer(cfg)

    # ── Init Ray ────────────────────────────────────────────────────────
    # Object-store overflow spills to disk, and Ray's default spill
    # location is the node's local /tmp — far smaller than scratch on this
    # cluster's compute nodes. A real seq_len=16384 tokenize pass produces
    # intermediate blocks large enough to fill it ("OSError: [Errno 28]
    # No space left on device", hit tokenizing Nemotron chat at 2.3M rows).
    #
    # Fixed via object_spilling_config rather than `_temp_dir=`: `_temp_dir`
    # also relocates Ray's own Unix-domain sockets (plasma_store, raylet),
    # and nests a long auto-generated `session_<timestamp>_<pid>/sockets/...`
    # path underneath whatever base you give it — a scratch-rooted path here
    # is long enough on its own to blow AF_UNIX's 107-byte path limit
    # (confirmed: `validate_socket_filename failed: AF_UNIX path length
    # cannot exceed 107 bytes`). Sockets are tiny and fine on local /tmp;
    # only the potentially-large spilled *data* needs to move to scratch,
    # so only object_spilling_config's directory_path is redirected.
    import json
    import ray
    _spill_dir = os.environ.get(
        "RAY_TMPDIR",
        os.path.join(os.path.dirname(cfg.data.cache_dir.rstrip("/")) or ".", ".ray_spill"),
    )
    os.makedirs(_spill_dir, exist_ok=True)
    ray.init(
        num_cpus=os.cpu_count(),
        include_dashboard=False,
        ignore_reinit_error=True,
        _system_config={
            "object_spilling_config": json.dumps({
                "type": "filesystem",
                "params": {"directory_path": _spill_dir},
            }),
        },
    )
    logger.info(f"Ray initialized for tokenization (spill dir: {_spill_dir}).")

    # Side-effect: populate REGISTRY / RAW_REGISTRY with all builders,
    # including local_chat which registers the "local" dataset.
    import bhaskera.data.datasets  # noqa: F401

    from bhaskera.data.registry import RAW_REGISTRY
    if dataset_name not in RAW_REGISTRY:
        logger.error(
            f"Dataset '{dataset_name}' is not registered in RAW_REGISTRY. "
            f"Available: {sorted(RAW_REGISTRY)}."
        )
        ray.shutdown()
        sys.exit(1)

    # ── Tokenize each requested split ───────────────────────────────────
    cache_paths: dict[str, str] = {}
    for split in splits:
        try:
            cache_paths[split or "default"] = _tokenize_one_split(cfg, dataset_name, split)
        except Exception as e:
            logger.error(f"Tokenization failed for split={split!r}: {e}", exc_info=True)
            ray.shutdown()
            sys.exit(2)

    ray.shutdown()

    _print_result_box(cache_paths, dataset_name, cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_splits(arg_split: str, cfg) -> List[str | None]:
    """
    Decide which splits to tokenize.

    Any registered raw builder that declares a `split` parameter (checked the
    same way call_raw_builder does) can be tokenized per-split; builders that
    don't declare it get a single pass with split=None, regardless of dataset
    name — no dataset names are hardcoded here.
    """
    from bhaskera.data.registry import RAW_REGISTRY
    import inspect

    fn = RAW_REGISTRY.get(cfg.data.name)
    supports_split = fn is not None and "split" in inspect.signature(fn).parameters

    if arg_split == "none" or not supports_split:
        if arg_split in ("val", "both"):
            logger.warning(
                f"--split {arg_split!r} requested but builder for "
                f"'{cfg.data.name}' doesn't accept a split argument — "
                "running a single pass instead."
            )
        return [None]

    if arg_split == "train":
        return ["train"]
    if arg_split == "val":
        return ["val"]
    if arg_split == "both":
        return ["train", "val"]

    # Default if --split not given: tokenize train only.
    return ["train"]


def _print_result_box(cache_paths: dict[str, str], dataset_name: str, cfg) -> None:
    """Print a copy-pasteable config snippet inside a box."""
    lines: list[str] = ["Tokenization complete!", ""]

    snippet_lines = [
        "data:",
        f"  name: \"{dataset_name}\"",
        f"  seq_len: {cfg.data.seq_len}",
    ]
    if cfg.data.format:
        snippet_lines.append(f"  format: \"{cfg.data.format}\"")

    # Map split -> YAML field
    field_for_split = {
        "train":   "tokenized_path",
        "default": "tokenized_path",
        "val":     "val_tokenized_path",
    }
    for split, path in cache_paths.items():
        field = field_for_split.get(split, "tokenized_path")
        snippet_lines.append(f"  {field}: \"{path}\"")

    lines += [f"Cache path(s):"]
    for split, path in cache_paths.items():
        lines.append(f"  [{split}] {path}")
    lines += ["", "Paste this into your training config:", ""]
    lines += snippet_lines

    border = "═" * _BOX_WIDTH
    print(f"\n╔{border}╗")
    for line in lines:
        padding = _BOX_WIDTH - len(line) - 1
        print(f"║ {line}{' ' * max(padding, 0)}║")
    print(f"╚{border}╝\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bhaskera-tokenize",
        description=(
            "One-shot tokenization CLI. Tokenizes a dataset once and writes to disk. "
            "Subsequent runs with the same config return immediately (cache hit)."
        ),
    )
    p.add_argument("--config", required=True,
        help="Path to Bhaskera YAML config")
    p.add_argument("--dataset", type=str, default=None,
        help="Dataset name to tokenize (overrides config.data.name)")
    p.add_argument("--storage-path", type=str, default=None,
        help="Root directory for tokenized parquet cache (overrides config.data.cache_dir)")
    p.add_argument("--overwrite", action="store_true",
        help="Force re-tokenization even if a valid cache exists")
    p.add_argument("--num-workers", type=int, default=None,
        help="Number of Ray CPU workers (overrides config.data.num_workers)")

    # Phase 2 additions
    p.add_argument("--split", type=str, default="train",
        choices=["train", "val", "both", "none"],
        help="Which split(s) to tokenize. 'none' disables split handling. "
             "Only meaningful for the 'local' dataset.")
    p.add_argument("--format", type=str, default=None,
        help="Format renderer name: chatml | alpaca | sharegpt | <custom>. "
             "Overrides config.data.format.")
    p.add_argument("--train-path", type=str, default=None,
        help="Path/dir/glob for train data (overrides config.data.train_path)")
    p.add_argument("--val-path", type=str, default=None,
        help="Path/dir/glob for val data (overrides config.data.val_path)")
    return p.parse_args()


if __name__ == "__main__":
    main()
