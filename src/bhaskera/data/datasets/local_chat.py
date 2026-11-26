"""
bhaskera.data.datasets.local_chat
=================================
Generic dataset builder for *local* files: JSONL, JSON (lines or one array), CSV, TSV, Parquet, Arrow/Feather, any of them gzipped
(.gz), and Hugging Face datasets saved to disk.

Reads from a path you specify in YAML and forwards everything else to the
existing tokenizer pipeline. The ``format`` column determines how rows are
rendered into a single text string before tokenisation.

YAML usage (tokenize CLI — needs raw paths):

    data:
      name: local
      format: chatml
      train_path: /data/customer_svc/train.jsonl
      val_path:   /data/customer_svc/val.jsonl
      seq_len:    2048
      cache_dir:  /scratch/cache

YAML usage (training — only needs the cache):

    data:
      name: local
      format: chatml
      tokenized_path:     /scratch/cache/local_train_<hash>
      val_tokenized_path: /scratch/cache/local_val_<hash>


"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import ray.data

from bhaskera.data.registry import register, register_raw
from bhaskera.data.tokenize import load_tokenized, tokenize_dataset

logger = logging.getLogger(__name__)


# File discovery

_JSON_EXTS    = (".jsonl", ".json", ".ndjson")
_PARQUET_EXTS = (".parquet",)
_CSV_EXTS     = (".csv",)
_TSV_EXTS     = (".tsv",)
_ARROW_EXTS   = (".arrow", ".feather")
_ALL_EXTS     = _JSON_EXTS + _PARQUET_EXTS + _CSV_EXTS + _TSV_EXTS + _ARROW_EXTS
_STRUCTURED   = ("messages", "conversations")      # columns that hold a list: a CSV or TSV cell holds it as JSON text


def _base(f: str) -> str:
    """The file name without a compression suffix (.gz): data.csv.gz -> data.csv."""
    f = f.lower()
    return f[:-3] if f.endswith(".gz") else f


def _kind(f: str) -> str:
    """json | parquet | csv | tsv | arrow | hf (a Hugging Face dataset saved to disk) | '' (not supported)."""
    if Path(f).is_dir() and _is_hf_disk(Path(f)):
        return "hf"
    b = _base(f)
    for kind, exts in (("json", _JSON_EXTS), ("parquet", _PARQUET_EXTS), ("csv", _CSV_EXTS), ("tsv", _TSV_EXTS), ("arrow", _ARROW_EXTS)):
        if b.endswith(exts):
            return kind
    return ""


def _is_hf_disk(p: Path) -> bool:
    """A folder written by ``datasets``' save_to_disk (one split, or several)."""
    return (p / "dataset_info.json").is_file() or (p / "state.json").is_file() or (p / "dataset_dict.json").is_file()


def _resolve_files(path: str) -> List[str]:
    """Expand ``path`` (file, directory, or glob) into a sorted list of files. A Hugging Face dataset folder is returned as itself."""
    p = Path(path)

    if p.is_file():
        return [str(p)]

    if p.is_dir():
        if _is_hf_disk(p):
            return [str(p)]
        files = []
        for f in sorted(p.rglob("*")):
            if f.is_file() and _kind(str(f)):
                files.append(f)
        return [str(f) for f in files]

    # Treat as glob (e.g. "/data/shards/*.jsonl")
    parent = p.parent if p.parent.exists() else Path(".")
    matches = sorted(parent.glob(p.name))
    if not matches:
        raise FileNotFoundError(
            f"local dataset path '{path}' matched no files. "
            "Provide a file, a directory, or a glob pattern."
        )
    return [str(m) for m in matches]


def _is_json_array(f: str) -> bool:
    """True when a JSON file is one array of rows ([{...}, {...}]) instead of one row per line."""
    import gzip
    opener = gzip.open if f.lower().endswith(".gz") else open
    with opener(f, "rb") as fh:
        head = fh.read(4096).lstrip()
    return head[:1] == b"["


def _decode_structured(df, cols=_STRUCTURED):
    """Turn a JSON-text cell into the list it holds, for the columns that hold lists (a CSV cannot hold a list)."""
    import json
    for c in cols:
        if c in df.columns and df[c].dtype == object:
            def fix(v):
                if isinstance(v, str) and v.lstrip()[:1] in ("[", "{"):
                    try:
                        return json.loads(v)
                    except ValueError:
                        return v
                return v
            df[c] = df[c].map(fix)
    return df


def _from_pandas(df):
    return ray.data.from_pandas(_decode_structured(df))


def _read_files(files: List[str]) -> ray.data.Dataset:
    """Read the files with Ray Data, whatever their format: JSONL, JSON arrays, CSV, TSV, Parquet, Arrow, a saved Hugging Face dataset."""
    kinds = {f: _kind(f) for f in files}
    bad = [f for f, k in kinds.items() if not k]
    if bad:
        raise ValueError(f"unsupported file(s) {bad[:3]}: expected one of {_ALL_EXTS} (optionally .gz), or a Hugging Face dataset folder")
    found = sorted(set(kinds.values()))
    if len(found) > 1 and not (set(found) <= {"json"}):
        raise ValueError(f"mixed formats in one local dataset path ({', '.join(found)}). Keep one format per path.")
    kind = found[0]
    if kind == "parquet":
        return ray.data.read_parquet(files)
    if kind == "json":
        lines = [f for f in files if not _is_json_array(f)]
        arrays = [f for f in files if _is_json_array(f)]
        parts = []
        if lines:
            # ray.data.read_json handles line-delimited JSON (.jsonl and one-row-per-line .json).
            #
            # override_num_blocks targets one block per input file. Note this is
            # a hint, not a guarantee: Ray Data's streaming executor budgets
            # resources across operators and will backpressure reads
            # ("[backpressured:tasks(ResourceBudget)]") regardless, so read
            # concurrency does not scale indefinitely with file count. At ~200
            # files / 54GB it serialises badly; at a few dozen files it is fine.
            # Keep the *input* sized to what training actually consumes rather
            # than relying on read parallelism to absorb an oversized corpus.
            parts.append(ray.data.read_json(lines, override_num_blocks=len(lines) if len(lines) > 1 else None))
        if arrays:                                    # one array per file is read whole, so prefer one row per line for large data
            import pandas as pd
            for f in arrays:
                parts.append(ray.data.from_pandas(pd.read_json(f, compression="infer")))
        ds = parts[0]
        for extra in parts[1:]:
            ds = ds.union(extra)
        return ds
    if kind in ("csv", "tsv"):
        import pandas as pd
        sep = "\t" if kind == "tsv" else ","
        frames = [pd.read_csv(f, sep=sep, compression="infer", dtype=str, keep_default_na=False) for f in files]
        return _from_pandas(pd.concat(frames, ignore_index=True))
    if kind == "arrow":
        import pandas as pd
        import pyarrow as pa

        def read_arrow(f):                            # .arrow and .feather are the Arrow IPC file format (or its stream form)
            with pa.memory_map(f, "r") as src:
                try:
                    return pa.ipc.open_file(src).read_all()
                except pa.ArrowInvalid:
                    return pa.ipc.open_stream(src).read_all()
        return _from_pandas(pd.concat([read_arrow(f).to_pandas() for f in files], ignore_index=True))
    if kind == "hf":
        from datasets import DatasetDict, load_from_disk
        ds = load_from_disk(files[0])
        if isinstance(ds, DatasetDict):
            names = list(ds.keys())
            raise ValueError(f"{files[0]} holds several splits ({', '.join(names)}); point data.train_path / val_path at one split's folder, e.g. {files[0]}/{names[0]}")
        return ray.data.from_pandas(_decode_structured(ds.to_pandas()))
    raise ValueError(f"no supported files found. Expected one of {_ALL_EXTS}.")


# Path resolution per split

def _path_for_split(cfg, split: Optional[str]) -> str:
    """
    Pick which YAML field to read for a given split.

    Only called on the *raw* path (tokenize CLI, or live tokenisation in
    training). When training from a pre-tokenized cache, we never get here.
    """
    data = cfg.data
    train_path = getattr(data, "train_path", None)
    val_path   = getattr(data, "val_path",   None)
    plain_path = getattr(data, "path",       None)

    if split == "val":
        if not val_path:
            raise ValueError(
                "split='val' requested but data.val_path is not set in config."
            )
        return val_path

    chosen = train_path or plain_path
    if not chosen:
        raise ValueError(
            "data.train_path (or data.path) must be set for the local dataset. "
            "Point it at a data file (JSONL, JSON, CSV, TSV, Parquet, Arrow), a directory, or a glob."
        )
    return chosen


# Registered builders

@register_raw("local", text_col="text")
def _build_raw(cfg, split: Optional[str] = None) -> ray.data.Dataset:
    """Return the raw (un-tokenized) local dataset for a given split."""
    path = _path_for_split(cfg, split)
    files = _resolve_files(path)

    if not files:
        raise FileNotFoundError(f"No files found for path '{path}'.")

    logger.info(
        f"local dataset: split={split!r} path={path!r} "
        f"-> {len(files)} file(s)"
    )
    return _read_files(files)


@register("local")
def build(cfg, world_size: int = 1) -> ray.data.Dataset:
    """
    Build the tokenized local dataset for the trainer.

    Two paths:
      * cfg.data.tokenized_path set  → load pre-tokenized parquet from disk
                                       (the normal training case). We never
                                       touch train_path here.
      * cfg.data.tokenized_path None → live tokenisation from train_path
                                       (slow; usually you want the
                                       bhaskera-tokenize CLI instead).
    """
    
    if cfg.data.tokenized_path:
        logger.info(
            f"local dataset: loading pre-tokenized cache "
            f"{cfg.data.tokenized_path!r} (world_size={world_size})"
        )
        return load_tokenized(cfg.data.tokenized_path, cfg, world_size)

    # ── Live tokenisation path: needs raw inputs. ────────────────────────
    return tokenize_dataset(_build_raw(cfg), cfg, "text", world_size=world_size)
