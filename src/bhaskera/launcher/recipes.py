"""Create a training recipe from a few plain choices, instead of writing the YAML by hand.

The home page's *Create a recipe* card calls this. It asks for what a person knows (the model folder, the method, the tokenised data,
how long to train) and fills in the rest from sensible defaults, writes ``recipes/<name>.yaml`` next to ``submit.sh``, and checks
the result with the same loader the training job uses, so a recipe it writes loads without warnings. It never overwrites a file.

Tokenised data is found, not typed: every folder with a ``metadata.json`` and Parquet files in the cache folders it knows about is
offered with its model, sequence length and number of rows. Raw JSONL data is turned into a tokenising recipe the same way
(``tokenize_recipe``), which the page submits as a CPU job.

Only text models are covered here (QLoRA + DDP, DDP, FSDP2). For other strategies and for multimodal models, start from a file
in ``configs/`` and edit it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,60}$")
METHODS = {
    "qlora": {"label": "QLoRA + DDP", "quantization": "qlora", "strategy": "ddp", "batch_size": 2,
              "about": "Fastest for dense models up to about 123B: the frozen model is stored in 4 bits."},
    "ddp": {"label": "DDP", "quantization": "none", "strategy": "ddp", "batch_size": 1,
            "about": "The whole model on every GPU, in bf16: for models that fit on one GPU (around 14B)."},
    "fsdp": {"label": "FSDP2", "quantization": "none", "strategy": "fsdp", "batch_size": 1,
             "about": "The model is split over all GPUs: when it does not fit on one and nothing else works."},
}
FORMATS = ("chatml", "alpaca", "sharegpt")


class RecipeError(ValueError):
    """A problem with the form, in words for the person who filled it in."""


def _num(value, name: str, low: float, high: float, integer: bool = True, default: Optional[float] = None):
    if value in (None, ""):
        if default is None:
            raise RecipeError(f"{name} is missing")
        return default
    try:
        n = int(value) if integer else float(value)
    except (TypeError, ValueError):
        raise RecipeError(f"{name} must be a number")
    if isinstance(value, bool) or not low <= n <= high:
        raise RecipeError(f"{name} must be between {low:g} and {high:g}")
    return n


def _folder(value, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeError(f"{what} is missing")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise RecipeError(f"{what} must be a full path (starting with /)")
    if not path.is_dir():
        raise RecipeError(f"{what} {path} is not a folder that exists on this machine")
    return str(path.resolve())


def model_folder(value) -> str:
    path = _folder(value, "the model folder")
    if not (Path(path) / "config.json").is_file():
        raise RecipeError(f"{path} has no config.json: it is not a Hugging Face model folder (download the model first: compute nodes have no internet)")
    return path


def cache_info(path: Path) -> Optional[dict]:
    """What a tokenised cache folder holds (from its metadata.json), or None when it is not one."""
    meta = path / "metadata.json"
    if not meta.is_file() or not any(path.glob("*.parquet")):
        return None
    try:
        m = json.loads(meta.read_text())
    except (OSError, ValueError):
        return None
    name = path.name
    side = "val" if "_val_" in name or name.startswith("val") else "train" if "_train_" in name or name.startswith("train") else ""
    return {"path": str(path), "name": name, "side": side, "model": os.path.basename(str(m.get("model_name", "")).rstrip("/")),
            "seq_len": m.get("seq_len"), "rows": m.get("num_rows"), "dataset": m.get("dataset_name", "")}


def cache_folders(kit: Optional[Path], recipe_files: list[Path], extra: Optional[list[str]] = None) -> list[Path]:
    """Folders that may hold tokenised caches: those named by existing recipes, a ``cache`` folder next to submit.sh, and any given."""
    found: list[Path] = []

    def add(p):
        try:
            p = Path(p).expanduser().resolve()
        except OSError:
            return
        if p.is_dir() and p not in found:
            found.append(p)

    for p in extra or []:
        add(p)
    if kit:
        add(kit / "cache")
        add(kit.parent / "cache")
    for f in recipe_files:
        try:
            import yaml
            d = (yaml.safe_load(f.read_text()) or {}).get("data") or {}
        except Exception:
            continue
        for key in ("tokenized_path", "val_tokenized_path"):
            if isinstance(d.get(key), str) and "<" not in d[key]:
                add(Path(d[key]).parent)
        if isinstance(d.get("cache_dir"), str) and "<" not in d["cache_dir"]:
            add(d["cache_dir"])
    return found


def find_caches(folders: list[Path]) -> list[dict]:
    out = []
    for folder in folders:
        try:
            children = sorted(p for p in folder.iterdir() if p.is_dir())
        except OSError:
            continue
        for p in children:
            info = cache_info(p)
            if info:
                out.append(info)
    return out


def suggested_models(recipe_files: list[Path]) -> list[str]:
    """Model folders that existing, ready recipes use, to offer in the form."""
    out = []
    for f in recipe_files:
        try:
            import yaml
            name = ((yaml.safe_load(f.read_text()) or {}).get("model") or {}).get("name")
        except Exception:
            continue
        if isinstance(name, str) and name.startswith("/") and "<" not in name and name not in out and Path(name).is_dir():
            out.append(name)
    return out


def build(form: dict, workdir: Path) -> dict:
    """The recipe as a dict, from the form; raises RecipeError."""
    name = form.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        raise RecipeError("the recipe name is letters, digits, - and _, starting with a letter (at most 61 characters)")
    method = METHODS.get(form.get("method"))
    if method is None:
        raise RecipeError("choose a method: QLoRA + DDP, DDP or FSDP2")
    model = model_folder(form.get("model"))
    train = _folder(form.get("train_cache"), "the training data")
    val_raw = form.get("val_cache")
    val = _folder(val_raw, "the validation data") if val_raw else None
    for p, what in ((train, "training"), (val, "validation")):
        if p and cache_info(Path(p)) is None:
            raise RecipeError(f"the {what} data {p} is not a tokenised cache (it needs metadata.json and Parquet files; "
                              "tokenise your data first with 'Prepare data' or ./submit.sh tokenize)")
    info = cache_info(Path(train))
    seq_len = _num(form.get("seq_len"), "sequence length", 128, 262144, default=int(info["seq_len"] or 4096))
    if info["seq_len"] and int(info["seq_len"]) != seq_len:
        raise RecipeError(f"the training data was tokenised with sequence length {info['seq_len']}, not {seq_len}: tokenise again, or use {info['seq_len']}")
    steps = _num(form.get("steps"), "steps", 1, 10_000_000, default=300)
    lr = _num(form.get("lr"), "learning rate", 1e-7, 1.0, integer=False, default=1e-4)
    rank = _num(form.get("rank"), "LoRA rank", 1, 512, default=16)
    batch = _num(form.get("batch_size"), "batch size", 1, 256, default=method["batch_size"])
    accum = _num(form.get("grad_accum"), "gradient accumulation", 1, 1024, default=1)
    every = _num(form.get("save_every_steps"), "save every (steps)", 1, 10_000_000, default=min(500, max(20, steps // 4)))
    evals = _num(form.get("eval_steps"), "evaluate every (steps)", 0, 10_000_000, default=max(1, steps // 6)) if val else 0
    recipe: dict[str, Any] = {
        "model": {"name": model, "dtype": "bfloat16", "attn_impl": "flash_attention_2", "trust_remote_code": False,
                  "use_liger_kernel": True, "quantization": method["quantization"]},
        "data": {"name": "local", "seq_len": seq_len, "runtime_packing": True, "tokenized_path": train,
                 **({"val_tokenized_path": val} if val else {}), "num_workers": 8, "prefetch_batches": 2},
        "lora": {"enabled": True, "r": rank, "alpha": 2 * rank, "dropout": 0.05, "target_modules": ["auto"]},
        "training": {"batch_size": batch, "grad_accum": accum, "lr": lr, "weight_decay": 0.0, "max_steps": steps, "num_epochs": 100,
                     "warmup_steps": max(1, min(100, steps // 20)), "grad_clip": 1.0, "eval_steps": evals, "eval_batch_size": batch,
                     "seed": 42, "distributed": {"strategy": method["strategy"]}},
        "checkpoint": {"enabled": True, "save_dir": str((workdir / "checkpoints" / name).resolve()), "save_interval": 1,
                       "save_every_steps": every, "keep_last_n": 3},
        "logging": {"tracker": "mlflow", "project": "bhaskera", "run_name": name, "mlflow_log_adapter": True, "mlflow_eval_samples": 5},
        "monitoring": {"dashboard_host": "127.0.0.1"},
    }
    if method["strategy"] == "fsdp":
        recipe["training"]["distributed"]["fsdp"] = {"sharding_strategy": "FULL_SHARD", "activation_checkpointing": True}
    return recipe


def render(recipe: dict, note: str = "") -> str:
    import yaml
    head = ["# Written by the Bhaskera home page (Create a recipe). Edit it freely: the job reads it again every time it starts.",
            "# Run it with:  ./submit.sh recipes/<this file> <nodes> <hours>"]
    if note:
        head.append(f"# {note}")
    return "\n".join(head) + "\n" + yaml.safe_dump(recipe, sort_keys=False, default_flow_style=False)


def validate(text: str) -> None:
    """Load the recipe with the training job's own loader; any problem or ignored option is an error."""
    import tempfile
    from bhaskera.config import load_config_checked
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        _cfg, unknown = load_config_checked(tmp)
    except Exception as e:
        raise RecipeError(f"the recipe does not load: {e}")
    finally:
        os.unlink(tmp)
    if unknown:
        raise RecipeError("the recipe has options that do nothing: " + "; ".join(unknown))


def write(recipes_dir: Path, name: str, text: str) -> Path:
    """Write recipes/<name>.yaml; never over an existing file."""
    recipes_dir.mkdir(parents=True, exist_ok=True)
    path = recipes_dir / f"{name}.yaml"
    try:
        with open(path, "x") as f:
            f.write(text)
    except FileExistsError:
        raise RecipeError(f"recipes/{name}.yaml exists already: pick another name, or edit that file")
    return path


def tokenize_recipe(form: dict, workdir: Path) -> tuple[str, str]:
    """(file name, text) of a recipe that tokenises raw JSONL data; raises RecipeError."""
    name = form.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        raise RecipeError("the name is letters, digits, - and _, starting with a letter")
    model = model_folder(form.get("model"))
    fmt = form.get("format") or "chatml"
    if fmt not in FORMATS:
        raise RecipeError("the data format is chatml, alpaca or sharegpt")

    def raw(value, what, required):
        if not value:
            if required:
                raise RecipeError(f"{what} is missing")
            return None
        p = Path(str(value)).expanduser()
        if not p.is_absolute() or not (p.is_file() or p.is_dir()):
            raise RecipeError(f"{what} {value} must be a full path to a file or folder that exists")
        return str(p.resolve())

    train, val = raw(form.get("train_path"), "the training data file", True), raw(form.get("val_path"), "the validation data file", False)
    seq_len = _num(form.get("seq_len"), "sequence length", 128, 262144, default=4096)
    data: dict[str, Any] = {"name": "local", "format": fmt, "train_path": train, **({"val_path": val} if val else {}), "seq_len": seq_len,
                            "cache_dir": str((workdir / "cache").resolve()), "num_workers": 8, "tokenize_batch_size": 32,
                            "tokenize_compression": "snappy"}
    import yaml
    text = ("# Written by the Bhaskera home page (Prepare data): tokenises your files once. The training recipe then uses the cache it makes.\n"
            + yaml.safe_dump({"model": {"name": model, "trust_remote_code": False}, "data": data}, sort_keys=False))
    return f"tokenize-{name}.yaml", text
