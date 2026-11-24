"""Create a training recipe from a few plain choices, instead of writing the YAML by hand.

The home page's *Create a recipe* card calls this. It asks for what a person knows (the model folder, the method, the tokenised data,
how long to train) and fills in the rest from sensible defaults, writes ``recipes/<name>.yaml`` next to ``submit.sh``, and checks
the result with the same loader the training job uses, so a recipe it writes loads without warnings. It never overwrites a file.

Tokenised data is found, not typed: every folder with a ``metadata.json`` and Parquet files in the cache folders it knows about is
offered with its model, sequence length and number of rows. Raw JSONL data is turned into a tokenising recipe the same way
(``tokenize_recipe``), which the page submits as a CPU job.

Text and multimodal models are covered, with QLoRA + DDP, DDP, FSDP2, HSDP and Pipeline. Every other setting is reachable through
``settings`` (the page's *All settings* list, built from ``bhaskera.config`` by ``schema``), and *Custom* takes a whole recipe as
YAML, for example a starter from ``configs/`` that you then edit.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,60}$")
METHODS = {
    "qlora": {"label": "QLoRA + DDP", "quantization": "qlora", "strategy": "ddp", "batch_size": 2, "kinds": ("text",),
              "about": "The usual starting point. The base model is squeezed to 4 bits so it needs far less memory, and only a small add-on is trained on top. Text models only."},
    "ddp": {"label": "DDP", "quantization": "none", "strategy": "ddp", "batch_size": 1, "kinds": ("text", "multimodal"),
            "about": "Every GPU keeps a full copy of the model. Simple and fast when the model fits on one GPU, up to about 14 billion parameters."},
    "fsdp": {"label": "FSDP2", "quantization": "none", "strategy": "fsdp", "batch_size": 1, "kinds": ("text", "multimodal"),
             "about": "The model is split across all the GPUs, so it can be bigger than one GPU holds. Pick this when DDP runs out of memory."},
    "hsdp": {"label": "HSDP", "quantization": "none", "strategy": "fsdp", "batch_size": 1, "kinds": ("text", "multimodal"),
             "about": "Like FSDP2, but the model is split only inside each machine and copied between machines, which keeps the busiest traffic on the fastest links."},
    "pipeline": {"label": "Pipeline", "quantization": "none", "strategy": "pipeline", "batch_size": 1, "kinds": ("text",),
                 "about": "The model is cut into consecutive pieces, one piece per group of GPUs. For the very largest text models. Text models only."},
    "custom": {"label": "Custom", "quantization": "none", "strategy": "", "batch_size": 1, "kinds": ("text", "multimodal"),
               "about": "For people who already know the settings: write or paste the whole recipe, or start from a starter or one of your recipes and edit it."},
}
KINDS = {"text": "Text model", "multimodal": "Multimodal model"}
MODALITIES = ("image", "video", "audio")
MAX_YAML = 100_000
MAX_SETTINGS = 400
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


def shared_folders(env: dict, kind: str) -> list[Path]:
    """Folders of models or data shared on this cluster: $BHASKERA_MODELS / $BHASKERA_DATA (colon separated), else a ``models`` or ``data``
    folder next to the Bhaskera install (the parent of $BHASKERA_HOME). Only folders that exist are returned."""
    var = {"models": "BHASKERA_MODELS", "data": "BHASKERA_DATA"}[kind]
    found = [Path(p) for p in (env.get(var) or "").split(":") if p]
    if not found and env.get("BHASKERA_HOME"):
        found = [Path(env["BHASKERA_HOME"]).parent / kind]
    return [p for p in found if p.is_dir()]


def _model_info(path: Path) -> Optional[dict]:
    """What the model list shows about a model folder: its name, whether it reads images or sound, and its size on disk."""
    try:
        cfg = json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    arch = str((cfg.get("architectures") or [""])[0])
    multi = any(k in cfg for k in ("vision_config", "audio_config", "thinker_config")) or any(w in arch for w in ("VL", "Omni", "ImageText"))
    size = 0
    try:
        with os.scandir(path) as it:
            size = sum(e.stat().st_size for e in it if e.name.endswith((".safetensors", ".bin")) and e.is_file())
    except OSError:
        pass
    return {"path": str(path), "name": path.name, "kind": "multimodal" if multi else "text", "arch": arch, "size_gb": round(size / 1e9, 1)}


def model_catalog(shared: list[Path], mine: list[str]) -> list[dict]:
    """The models to offer: every model folder inside the shared folders, then those named by your own recipes."""
    found: list[dict] = []
    seen: set[str] = set()
    candidates: list[tuple[str, Path]] = []
    for folder in shared:
        try:
            candidates += [("shared", c) for c in sorted(folder.iterdir()) if c.is_dir()]
        except OSError:
            continue
    candidates += [("yours", Path(m)) for m in mine]
    for group, path in candidates:
        try:
            key = str(path.resolve())
        except OSError:
            continue
        if key in seen:
            continue
        info = _model_info(path)
        if info:
            seen.add(key)
            found.append({**info, "group": group})
    return found


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


def _choice(form: dict) -> tuple[str, dict, str, list[str]]:
    """(method key, method, kind, modalities) from the form, checked against each other."""
    key = form.get("method")
    method = METHODS.get(key)
    if method is None:
        raise RecipeError("choose a method: " + ", ".join(m["label"] for m in METHODS.values()))
    kind = form.get("model_kind") or "text"
    if kind not in KINDS:
        raise RecipeError("the model is a text model or a multimodal model")
    if kind not in method["kinds"]:
        raise RecipeError(f"{method['label']} is not available for {KINDS[kind].lower()}s" + (": QLoRA is not supported for multimodal models yet" if key == "qlora" else ""))
    mods: list[str] = []
    if kind == "multimodal":
        raw = form.get("modalities") or ["image"]
        raw = raw if isinstance(raw, list) else [raw]
        mods = [m for m in MODALITIES if m in raw]
        if not mods or len(mods) != len(set(raw)):
            raise RecipeError("choose what the data holds: image, video and audio (any of them)")
    return key, method, kind, mods


def build(form: dict, workdir: Path) -> dict:
    """The recipe as a dict, from the form; raises RecipeError."""
    name = form.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        raise RecipeError("the recipe name is letters, digits, - and _, starting with a letter (at most 61 characters)")
    key, method, kind, mods = _choice(form)
    if key == "custom":
        return custom(form, workdir)
    multi = kind == "multimodal"
    model = model_folder(form.get("model"))
    train = _folder(form.get("train_cache"), "the training data")
    val_raw = form.get("val_cache")
    val = _folder(val_raw, "the validation data") if val_raw else None
    for p, what in ((train, "training"), (val, "validation")):
        if p and cache_info(Path(p)) is None:
            raise RecipeError(f"the {what} data {p} is not a tokenised cache (it needs metadata.json and Parquet files; "
                              "tokenise your data first with 'Prepare data' or ./submit.sh tokenize)")
    info = cache_info(Path(train))
    seq_len = _num(form.get("seq_len"), "sequence length", 128, 262144, default=int(info["seq_len"] or (1280 if multi else 4096)))
    if info["seq_len"] and int(info["seq_len"]) != seq_len:
        raise RecipeError(f"the training data was tokenised with sequence length {info['seq_len']}, not {seq_len}: tokenise again, or use {info['seq_len']}")
    steps = _num(form.get("steps"), "steps", 1, 10_000_000, default=300)
    lr = _num(form.get("lr"), "learning rate", 1e-7, 1.0, integer=False, default=1e-4)
    rank = _num(form.get("rank"), "LoRA rank", 1, 512, default=16)
    batch = _num(form.get("batch_size"), "batch size", 1, 256, default=2 if multi else method["batch_size"])
    accum = _num(form.get("grad_accum"), "gradient accumulation", 1, 1024, default=1)
    every = _num(form.get("save_every_steps"), "save every (steps)", 1, 10_000_000, default=min(500, max(20, steps // 4)))
    evals = _num(form.get("eval_steps"), "evaluate every (steps)", 0, 10_000_000, default=max(1, steps // 6)) if val else 0
    data: dict[str, Any] = {"name": "local", "seq_len": seq_len, "tokenized_path": train, **({"val_tokenized_path": val} if val else {})}
    if multi:
        data.update({"modalities": ["text"] + mods, "num_workers": 8, "image_workers": 8, "image_prefetch_batches": 4})
    else:
        data.update({"runtime_packing": True, "num_workers": 8, "prefetch_batches": 2})
    recipe: dict[str, Any] = {
        "model": {"name": model, "dtype": "bfloat16", "attn_impl": "flash_attention_2", "trust_remote_code": False,
                  "use_liger_kernel": not multi, "quantization": method["quantization"],
                  **({"kind": "multimodal", "freeze_encoders": ["all"]} if multi else {})},
        "data": data,
        "lora": {"enabled": True, "r": rank, "alpha": 2 * rank, "dropout": 0.05, "target_modules": ["auto"]},
        "training": {"batch_size": batch, "grad_accum": accum, "lr": lr, "weight_decay": 0.0, "max_steps": steps, "num_epochs": 100,
                     "warmup_steps": max(1, min(100, steps // 20)), "grad_clip": 1.0, "eval_steps": evals, "eval_batch_size": batch,
                     "seed": 42, "distributed": {"strategy": method["strategy"]}},
        "checkpoint": {"enabled": True, "save_dir": str((workdir / "checkpoints" / name).resolve()), "save_interval": 1,
                       "save_every_steps": every, "keep_last_n": 3},
        "logging": {"tracker": "mlflow", "project": "bhaskera", "run_name": name, "mlflow_log_adapter": True, "mlflow_eval_samples": 5},
        "monitoring": {"dashboard_host": "127.0.0.1"},
    }
    dist = recipe["training"]["distributed"]
    if method["strategy"] == "ddp":
        dist["ddp"] = {"activation_checkpointing": True}      # off by default under DDP: a 14B model at 4096 tokens then runs out of memory
    if key == "fsdp":
        dist["fsdp"] = {"sharding_strategy": "FULL_SHARD", "activation_checkpointing": True}
    elif key == "hsdp":
        dist["fsdp"] = {"sharding_strategy": "HYBRID_SHARD", "hsdp_shard_size": _num(form.get("hsdp_shard_size"), "GPUs per shard group", 0, 4096, default=0),
                        "activation_checkpointing": True}
    elif key == "pipeline":
        stages = _num(form.get("stages"), "pipeline stages", 2, 1024, default=4)
        micro = _num(form.get("microbatches"), "micro-batches", 1, 4096, default=2 * stages)
        dist["pipeline"] = {"stages": stages, "microbatches": micro, "schedule": "1f1b", "activation_checkpointing": True}
        recipe["checkpoint"]["save_interval"] = every
    return finish(recipe, form, workdir)


def finish(recipe: dict, form: dict, workdir: Path) -> dict:
    """The user's own settings on top, then the one thing no setting may change: the Ray Dashboard stays on this machine."""
    apply_settings(recipe, form.get("settings"))
    recipe.setdefault("monitoring", {})["dashboard_host"] = "127.0.0.1"
    return recipe


def custom(form: dict, workdir: Path) -> dict:
    """A recipe written by hand: the YAML from the form. A copy of an existing recipe gets its own run name and checkpoint folder."""
    import yaml
    text = form.get("yaml")
    if not isinstance(text, str) or not text.strip():
        raise RecipeError("the recipe text is empty: write it, or pick a starter or one of your recipes to start from")
    if len(text) > MAX_YAML:
        raise RecipeError(f"the recipe is longer than {MAX_YAML:,} characters")
    try:
        recipe = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise RecipeError(f"the recipe is not valid YAML: {str(e).splitlines()[0] if str(e) else e}")
    if not isinstance(recipe, dict):
        raise RecipeError("the recipe must be a YAML mapping (model:, data:, training: ...)")
    values = json.dumps(recipe, default=str)
    for needle in ("<project>", "<username>", "REPLACE_", "RUN_NAME", "TOKENIZED_TRAIN", "TOKENIZED_VAL"):
        if needle in values:
            raise RecipeError(f"the recipe still has a placeholder ({needle}): put a real value there")
    name = form["name"]
    if form.get("copy"):
        recipe.setdefault("logging", {})["run_name"] = name
        recipe.setdefault("checkpoint", {})["save_dir"] = str((workdir / "checkpoints" / name).resolve())
    return finish(recipe, form, workdir)


# What every setting is called, its type and what it does, taken from bhaskera.config so the page and the loader cannot drift apart.

def _comments(cls) -> dict:
    """{field: the comment written next to it in the source}."""
    import inspect
    out: dict = {}
    pending: list[str] = []
    for line in inspect.getsource(cls).splitlines():
        t = line.strip()
        if not t:
            pending = []
        elif t.startswith("#"):
            if "──" not in t:
                pending.append(t.lstrip("# ").strip())
        else:
            m = re.match(r"(\w+)\s*:", t)
            if m and not t.startswith(("class ", "def ", '"""')):
                trailing = t.split("#", 1)[1].strip() if "#" in t else ""
                text = " ".join(pending + ([trailing] if trailing else [])).strip()
                if text:
                    cut = text[:240]
                    out[m.group(1)] = cut if len(text) <= 240 else (cut[:cut.rfind(". ") + 1] if ". " in cut else cut.rstrip() + " …")
            pending = []
    return out


def _kind_of(tp) -> str:
    import typing
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return _kind_of(args[0]) if len(args) == 1 else "text-or-list"
    if tp is bool:
        return "bool"
    if tp is int:
        return "int"
    if tp is float:
        return "float"
    if tp is str:
        return "str"
    if origin is list or tp is list:
        return "list"
    if origin is dict or tp is dict:
        return "dict"
    return "str"


def _walk(cls, prefix: str, hints_of, out: dict) -> None:
    import dataclasses
    hints = hints_of(cls)
    notes = _comments(cls)
    for f in dataclasses.fields(cls):
        tp = hints[f.name]
        if dataclasses.is_dataclass(tp):
            _walk(tp, f"{prefix}{f.name}.", hints_of, out)
            continue
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        out[f"{prefix}{f.name}"] = {"key": f"{prefix}{f.name}", "type": _kind_of(tp), "default": default, "optional": type(None) in getattr(tp, "__args__", ()),
                                    "help": notes.get(f.name, "")}


def fields() -> dict:
    """{dotted key: {type, default, help}} for every setting of a recipe."""
    import typing
    from bhaskera.config import Config
    out: dict = {}
    _walk(Config, "", typing.get_type_hints, out)
    return out


def schema() -> list[dict]:
    """The settings grouped by section (``training.distributed.fsdp``), in the order of the config file."""
    groups: dict = {}
    for k, f in fields().items():
        groups.setdefault(k.rsplit(".", 1)[0], []).append({**f, "name": k.rsplit(".", 1)[1]})
    return [{"section": s, "fields": fs} for s, fs in groups.items()]


def _coerce(key: str, f: dict, value):
    kind = f["type"]
    if isinstance(value, str):
        value = value.strip()
    if value in (None, ""):
        return None
    try:
        if kind == "bool":
            if isinstance(value, bool):
                return value
            low = str(value).lower()
            if low in ("true", "yes", "on", "1"):
                return True
            if low in ("false", "no", "off", "0"):
                return False
            raise ValueError
        if kind == "int":
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError
            return int(value)
        if kind == "float":
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if kind == "list":
            return value if isinstance(value, list) else [x.strip() for x in str(value).split(",") if x.strip()]
        if kind == "text-or-list":
            if isinstance(value, list):
                return value
            parts = [x.strip() for x in str(value).split(",") if x.strip()]
            return parts if len(parts) > 1 else (parts[0] if parts else None)
        if kind == "dict":
            if isinstance(value, dict):
                return value
            d = json.loads(str(value))
            if not isinstance(d, dict):
                raise ValueError
            return d
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError
        return str(value)
    except (ValueError, TypeError):
        what = {"bool": "true or false", "int": "a whole number", "float": "a number", "dict": 'a JSON object like {"a": "b"}', "list": "a comma-separated list"}.get(kind, "text")
        raise RecipeError(f"{key} must be {what}")


def apply_settings(recipe: dict, settings) -> None:
    """Set each dotted key of ``settings`` in the recipe; an empty value leaves the recipe's own value. Unknown keys are refused."""
    if not settings:
        return
    if not isinstance(settings, dict) or len(settings) > MAX_SETTINGS:
        raise RecipeError("the settings could not be read")
    known = fields()
    for key, value in settings.items():
        f = known.get(key)
        if f is None:
            import difflib
            near = difflib.get_close_matches(str(key), list(known), n=1, cutoff=0.6)
            raise RecipeError(f"'{key}' is not a recipe setting" + (f" (did you mean '{near[0]}'?)" if near else ""))
        if isinstance(value, str) and len(value) > 4000:
            raise RecipeError(f"{key} is too long")
        v = _coerce(key, f, value)
        if v is None:
            continue
        node = recipe
        parts = key.split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = node[p] = {}
            node = nxt
        node[parts[-1]] = v


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
    try:
        from bhaskera.models.multimodal import check_multimodal_settings
    except Exception:
        return                      # the model libraries are not usable here (the page may run without them); the job checks again
    try:
        check_multimodal_settings(_cfg)
    except ValueError as e:
        raise RecipeError(str(e))


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
    multi = form.get("model_kind") == "multimodal"
    seq_len = _num(form.get("seq_len"), "sequence length", 128, 262144, default=1280 if multi else 4096)
    data: dict[str, Any] = {"name": "local", "format": fmt, "train_path": train, **({"val_path": val} if val else {}), "seq_len": seq_len,
                            "cache_dir": str((workdir / "cache").resolve()), "num_workers": 8, "tokenize_batch_size": 32,
                            "tokenize_compression": "snappy"}
    if multi:
        raw_mods = form.get("modalities") or ["image"]
        mods = [m for m in MODALITIES if m in (raw_mods if isinstance(raw_mods, list) else [raw_mods])]
        if not mods:
            raise RecipeError("choose what the data holds: image, video and audio (any of them)")
        data["modalities"] = ["text"] + mods
        root = raw(form.get("media_root"), "the media folder", False)
        if root:
            data["media_root"] = root
        data["loss_on"] = "assistant"
    import yaml
    text = ("# Written by the Bhaskera home page (Prepare data): tokenises your files once. The training recipe then uses the cache it makes.\n"
            + yaml.safe_dump({"model": {"name": model, "trust_remote_code": False, **({"kind": "multimodal"} if multi else {})}, "data": data}, sort_keys=False))
    return f"tokenize-{name}.yaml", text
