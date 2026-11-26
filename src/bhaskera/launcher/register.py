"""Save a trained model in the MLflow model registry, and find it again by name.

    python -m bhaskera.launcher.register add --run <run id> --name <model> [--alias champion]
    python -m bhaskera.launcher.register list
    python -m bhaskera.launcher.register resolve <model>[@alias | :version]

``add`` registers the adapter a run stored (``logging.mlflow_log_adapter: true``) as a new version of the model ``--name``. Do it after
you have looked at the results: the version carries the base model, the run, the final evaluation and any benchmark results as tags, so
the registry shows what each version is. An alias such as ``champion`` points at one version and can be moved to a better one later.
``list`` shows every model with its versions and aliases. ``resolve`` turns ``name``, ``name:3`` or ``name@champion`` into the training
run behind it: ``./submit.sh serve --run name@champion`` and ``./submit.sh eval --run name:3`` accept the same names.

The registry is part of the MLflow store (``~/mlflow-runs`` by default), so the dashboard's *Models* link shows it.
"""
from __future__ import annotations

from bhaskera import paths

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

RUN_ID = re.compile(r"^[0-9a-f]{32}$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
ALIAS = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")


class RegisterError(Exception):
    """A problem the user can fix, in words."""


def _store_default() -> str:
    return paths.mlflow_store()


def make_client(store: str):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import logging
    logging.getLogger("mlflow").setLevel(logging.WARNING)         # no database-setup chatter on the terminal
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as e:
        raise RegisterError("MLflow is not installed in this Python environment") from e
    uri = store if "://" in store else f"file://{Path(store).resolve()}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def _truncate(v: Any, n: int = 250) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def run_tags(client, run_id: str) -> dict[str, str]:
    """What the registry shows about a version: the run, its base model, its final evaluation and its benchmark results."""
    run = client.get_run(run_id)
    m, p = run.data.metrics, run.data.params
    tags = {"framework": "bhaskera", "format": "peft-lora", "run_name": run.info.run_name or "", "base_model": p.get("model.name", "")}
    for k in ("eval_loss", "eval_accuracy"):
        if k in m:
            tags[k] = f"{m[k]:.4f}"
    for k in sorted(m):
        if k.startswith("lmeval_") and (k.endswith("_tuned") or k.endswith("_delta")) and len(tags) < 24:
            tags[k] = f"{m[k]:.4f}"
    return {k: _truncate(v) for k, v in tags.items() if v != ""}


def model_card(run, run_id: str, note: str = "") -> str:
    """The hand-off note kept with a registered version: what it is, how it was made, how well it did, how to use it."""
    p, m = run.data.params, run.data.metrics
    got = lambda *keys: next((p[k] for k in keys if p.get(k)), "")
    lines = []
    if note.strip():
        lines += [note.strip(), ""]
    lines += [f"Base model: {got('model.name') or 'unknown'}", f"Trained in run: {run.info.run_name or ''} ({run_id})"]
    kind = got("model.quantization")
    how = ", ".join(x for x in (f"LoRA rank {got('lora.r')}, alpha {got('lora.alpha')}" if got("lora.r") else "", "QLoRA (4-bit base)" if kind == "qlora" else "",
                                f"parallelism {got('training.distributed.strategy')}" if got("training.distributed.strategy") else "") if x)
    if how:
        lines.append("Method: " + how)
    train = ", ".join(x for x in (f"{got('training.max_steps')} steps" if got("training.max_steps") else "", f"learning rate {got('training.lr')}" if got("training.lr") else "",
                                  f"batch {got('training.batch_size')} x accumulation {got('training.grad_accum')}" if got("training.batch_size") else "",
                                  f"sequence length {got('data.seq_len')}" if got("data.seq_len") else "") if x)
    if train:
        lines.append("Training: " + train)
    if got("data.tokenized_path"):
        lines.append("Data: " + got("data.tokenized_path"))
    res = [f"{k.replace('_', ' ')} {m[k]:.4f}" for k in ("eval_loss", "eval_accuracy") if k in m]
    res += [f"{k[len('lmeval_'):-len('_tuned')].replace('_', ' ')} {m[k]:.3f}" for k in sorted(m) if k.startswith("lmeval_") and k.endswith("_tuned")][:8]
    lines.append("Results: " + ("; ".join(res) if res else "not evaluated yet"))
    lines.append("Use: the adapter needs the base model above at the same path. ./submit.sh serve --run <name>:<version> to chat, ./submit.sh eval --run <name>:<version> to test.")
    return "\n".join(lines)


def add(client, run_id: str, name: str, alias: Optional[str] = None, note: str = "") -> dict:
    """Register the adapter stored with a run as a new version of ``name``; point ``alias`` at it when given."""
    if not RUN_ID.match(run_id or ""):
        raise RegisterError("give the 32-character MLflow run id (the dashboard lists it, and `evaluate`/`serve` use the same id)")
    if not NAME.match(name or ""):
        raise RegisterError("a model name is letters, digits, _, - and . (at most 64), starting with a letter or digit")
    if alias and not ALIAS.match(alias):
        raise RegisterError("an alias is letters, digits, _ and - (at most 32), starting with a letter")
    try:
        run = client.get_run(run_id)
    except Exception as e:
        raise RegisterError(f"there is no run {run_id} in this MLflow store") from e
    if not any(a.path.endswith("adapter_config.json") for a in client.list_artifacts(run_id, "adapter")):
        raise RegisterError(f"run {run_id} has no stored adapter; train with logging.mlflow_log_adapter: true")
    try:
        client.create_registered_model(name, description="LoRA adapters trained with Bhaskera")
    except Exception:
        pass                                              # it exists already: this adds a version
    version = client.create_model_version(name, source=f"{run.info.artifact_uri}/adapter", run_id=run_id, tags=run_tags(client, run_id),
                                          description=model_card(run, run_id, note))
    v = str(version.version)
    if alias:
        client.set_registered_model_alias(name, alias, v)
    return {"name": name, "version": v, "alias": alias or "", "run_id": run_id}


def listing(client) -> list[dict]:
    """Every registered model with its versions, newest first."""
    out = []
    for rm in client.search_registered_models():
        versions = client.search_model_versions(f"name='{rm.name}'")
        rows = [{"version": str(v.version), "run_id": v.run_id, "aliases": sorted(getattr(v, "aliases", None) or []),
                 "tags": dict(v.tags or {}), "description": getattr(v, "description", "") or ""} for v in versions]
        rows.sort(key=lambda r: int(r["version"]), reverse=True)
        out.append({"name": rm.name, "versions": rows})
    return sorted(out, key=lambda m: m["name"])


def resolve(client, ref: str) -> str:
    """The training run behind ``name``, ``name:version`` or ``name@alias`` (a run id is returned as it is)."""
    if RUN_ID.match(ref or ""):
        return ref
    name, alias, ver = ref, None, None
    if "@" in ref:
        name, alias = ref.split("@", 1)
    elif ":" in ref:
        name, ver = ref.split(":", 1)
    try:
        if alias:
            mv = client.get_model_version_by_alias(name, alias)
        elif ver:
            mv = client.get_model_version(name, ver)
        else:
            versions = client.search_model_versions(f"name='{name}'")
            if not versions:
                raise RegisterError(f"there is no registered model '{name}' (`register list` shows the ones there are)")
            mv = max(versions, key=lambda v: int(v.version))
    except RegisterError:
        raise
    except Exception as e:
        raise RegisterError(f"there is no '{ref}' in the model registry (`register list` shows what there is)") from e
    if not mv.run_id:
        raise RegisterError(f"'{ref}' was not made from a training run, so its adapter cannot be found")
    return mv.run_id


def resolve_in_store(store: str, ref: str) -> str:
    """For callers that have a store path: a run id stays a run id (no MLflow needed); a registry name is looked up."""
    return ref if RUN_ID.match(ref or "") else resolve(make_client(store), ref)


def _show(models: list[dict]) -> str:
    if not models:
        return "no models are registered yet: `register add --run <run id> --name <model>`"
    lines = []
    for m in models:
        lines.append(m["name"])
        for v in m["versions"]:
            t = v["tags"]
            bits = [f"run {v['run_id'][:8]}", t.get("run_name", ""), f"eval accuracy {t['eval_accuracy']}" if "eval_accuracy" in t else ""]
            lines.append(f"  version {v['version']:>3}  {', '.join(b for b in bits if b)}" + (f"   <- {', '.join(v['aliases'])}" if v["aliases"] else ""))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="bhaskera-register", description="Save trained models in the MLflow model registry.")
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("add", "list", "resolve"):
        s = sub.add_parser(name)
        s.add_argument("--store", default=_store_default(), help="MLflow store (default ~/mlflow-runs)")
    a = sub.choices["add"]
    a.add_argument("--run", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--alias", help="point this alias (for example champion) at the new version")
    a.add_argument("--note", default="", help="a hand-off note kept with the version: what it is for, the data, known limits")
    sh = sub.add_parser("show", help="print the hand-off note of a version")
    sh.add_argument("ref", help="name, name:version or name@alias")
    sh.add_argument("--store", default=_store_default())
    sub.choices["resolve"].add_argument("ref", help="name, name:version or name@alias")
    args = p.parse_args(argv)
    try:
        client = make_client(args.store)
        if args.action == "add":
            r = add(client, args.run, args.name, args.alias, args.note)
            print(f"registered {r['name']} version {r['version']}" + (f" (alias {r['alias']})" if r["alias"] else ""))
            print(f"use it:  ./submit.sh serve --run {r['name']}:{r['version']}")
        elif args.action == "list":
            print(_show(listing(client)))
        elif args.action == "show":
            name, _, rest = args.ref.partition("@") if "@" in args.ref else (args.ref.partition(":")[0], "", args.ref.partition(":")[2])
            run_id = resolve(client, args.ref)
            hit = [v for m in listing(client) if m["name"] == name for v in m["versions"] if v["run_id"] == run_id and (not rest or rest in v["aliases"] or v["version"] == rest)]
            print(hit[0]["description"] if hit and hit[0]["description"] else "this version has no hand-off note")
        else:
            print(resolve(client, args.ref))
        return 0
    except RegisterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
