"""Hyperparameter sweeps: many runs of one recipe that differ in a few options, compared in MLflow.

    python -m bhaskera.launcher.sweep plan   sweep.yaml     # write one recipe per trial and list them
    python -m bhaskera.launcher.sweep submit sweep.yaml     # plan, then queue every trial with ./submit.sh
    python -m bhaskera.launcher.sweep report <sweep name>   # the finished trials, best first

A sweep file:

    name: lr-and-rank
    recipe: recipes/qwen3-14b-qlora.yaml
    nodes: 8
    hours: 6
    mode: grid                  # grid (every combination) or random
    trials: 8                   # random: how many; grid: all combinations
    seed: 0
    params:                     # option path -> list of values (random mode also takes ranges)
      training.lr: [5.0e-5, 1.0e-4, 2.0e-4]
      lora.r: [8, 16]
      # training.lr: {loguniform: [1.0e-5, 1.0e-3]}     uniform / loguniform / int / choice

Every trial is a normal run: it gets its own run name and checkpoint folder, and the tag ``group`` is the sweep
name, so MLflow shows the trials side by side. A parameter that is not a real recipe option is rejected before
anything is submitted.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

MAX_TRIALS = 64


def set_path(tree: dict, dotted: str, value: Any) -> None:
    """tree['a']['b'] = value for 'a.b', creating the sections on the way."""
    keys = dotted.split(".")
    for key in keys[:-1]:
        tree = tree.setdefault(key, {})
        if not isinstance(tree, dict):
            raise ValueError(f"'{dotted}': '{key}' is not a section")
    tree[keys[-1]] = value


def _draw(rng: random.Random, spec: Any) -> Any:
    """One value from a list, or from a range given as {uniform|loguniform|int|choice: [...]}."""
    if isinstance(spec, list):
        return rng.choice(spec)
    if isinstance(spec, dict) and len(spec) == 1:
        kind, args = next(iter(spec.items()))
        if kind == "uniform":
            return round(rng.uniform(*args), 8)
        if kind == "loguniform":
            return float(f"{math.exp(rng.uniform(math.log(args[0]), math.log(args[1]))):.4g}")
        if kind == "int":
            return rng.randint(*args)
        if kind == "choice":
            return rng.choice(args)
    raise ValueError(f"cannot read the parameter range {spec!r}: use a list, or uniform/loguniform/int/choice")


def combinations(params: dict, mode: str = "grid", trials: int = 0, seed: int = 0) -> list[dict]:
    """The sets of option values to try: every combination (grid) or ``trials`` random draws."""
    if not params:
        raise ValueError("the sweep has no params")
    if mode == "grid":
        for name, values in params.items():
            if not isinstance(values, list):
                raise ValueError(f"grid mode needs a list of values for '{name}'")
        names = list(params)
        return [dict(zip(names, combo)) for combo in itertools.product(*(params[n] for n in names))]
    if mode == "random":
        if trials <= 0:
            raise ValueError("random mode needs trials: <number>")
        rng = random.Random(seed)
        seen, out = set(), []
        for _ in range(trials * 20):                       # skip repeated draws; stop when the space is used up
            draw = {n: _draw(rng, spec) for n, spec in params.items()}
            key = repr(sorted(draw.items()))
            if key not in seen:
                seen.add(key)
                out.append(draw)
            if len(out) == trials:
                break
        return out
    raise ValueError(f"mode must be grid or random, not {mode!r}")


def plan_trials(spec: dict, recipe: dict) -> list[dict]:
    """For each trial: its name, the option values and the complete recipe."""
    name = str(spec["name"])
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError("the sweep name may only contain letters, digits, '.', '_' and '-'")
    sets = combinations(spec["params"], spec.get("mode", "grid"), int(spec.get("trials", 0) or 0), int(spec.get("seed", 0) or 0))
    if len(sets) > MAX_TRIALS:
        raise ValueError(f"{len(sets)} trials is a lot of GPU time; the limit is {MAX_TRIALS} (reduce the values or use random mode)")
    base_dir = (recipe.get("checkpoint") or {}).get("save_dir") or "./checkpoints"
    plans = []
    for i, values in enumerate(sets, start=1):
        trial = f"{name}-{i:02d}"
        cfg = copy.deepcopy(recipe)
        for path, value in values.items():
            set_path(cfg, path, value)
        set_path(cfg, "logging.run_name", trial)
        set_path(cfg, "logging.group", name)
        tags = list((cfg.get("logging") or {}).get("tags") or [])
        set_path(cfg, "logging.tags", tags + [f"sweep:{name}", f"trial:{i:02d}"])
        set_path(cfg, "checkpoint.save_dir", f"{str(base_dir).rstrip('/')}/{name}/{trial}")
        plans.append({"name": trial, "values": values, "recipe": cfg})
    return plans


def check_options(plans: list[dict]) -> list[str]:
    """Options in a trial recipe that the training code would ignore (a mistyped sweep parameter, for example)."""
    from bhaskera.config import _Tracked, _dict_to_config, find_unknown_keys
    problems: list[str] = []
    for plan in plans[:1]:                                 # the trials differ only in values, so one is enough
        tracked = _Tracked(plan["recipe"])
        _dict_to_config(tracked)
        problems += find_unknown_keys(tracked)
    return problems


def write_trials(plans: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for plan in plans:
        path = out_dir / f"{plan['name']}.yaml"
        path.write_text(yaml.safe_dump(plan["recipe"], sort_keys=False))
        paths.append(path)
    return paths


def table(plans: list[dict]) -> str:
    cols = list(plans[0]["values"])
    rows = [["trial"] + cols] + [[p["name"]] + [str(p["values"][c]) for c in cols] for p in plans]
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() for r in rows)


def _load(spec_path: Path) -> tuple[dict, dict]:
    spec = yaml.safe_load(spec_path.read_text()) or {}
    for key in ("name", "recipe", "params"):
        if key not in spec:
            raise SystemExit(f"{spec_path}: '{key}' is missing")
    recipe_path = Path(spec["recipe"])
    if not recipe_path.is_absolute():
        recipe_path = spec_path.parent / recipe_path
    return spec, yaml.safe_load(recipe_path.read_text()) or {}


def cmd_plan(spec_path: Path, submit: bool) -> int:
    spec, recipe = _load(spec_path)
    try:
        plans = plan_trials(spec, recipe)
    except ValueError as e:
        print(f"error: {e}")
        return 1
    problems = check_options(plans)
    if problems:
        print("error: the sweep would change options that do nothing:")
        for text in problems:
            print(f"  {text}")
        return 1
    out_dir = spec_path.parent / "sweeps" / str(spec["name"])
    paths = write_trials(plans, out_dir)
    print(f"{len(plans)} trials of {spec['recipe']} written to {out_dir}\n")
    print(table(plans))
    if not submit:
        print(f"\nQueue them with: python -m bhaskera.launcher.sweep submit {spec_path}")
        return 0
    nodes, hours = int(spec.get("nodes", 8)), spec.get("hours")
    # Slurm lets one user have only so many jobs submitted in a partition (the QoS limit). Queue what fits and
    # remember it, so that running `submit` again later continues with the rest.
    done_file = out_dir / "submitted.json"
    done: dict = json.loads(done_file.read_text()) if done_file.exists() else {}
    todo = [(p, pl) for p, pl in zip(paths, plans) if pl["name"] not in done]
    from . import qos
    try:
        partition, _ = qos.partition_for(nodes)
        limits = qos.limits_for(partition)
        running, submitted = qos.user_jobs(partition)
        left = qos.room(limits, running, submitted)
        print("\n" + qos.summary(partition, limits, running, submitted))
    except qos.NoPartitionTable:
        left = None                                # no partition table for this cluster: queue everything
    except ValueError as e:
        print(f"error: {e}")
        return 1
    if left is not None and left < len(todo):
        print(f"room for {left} more job(s) in {partition}; the other {len(todo) - left} wait until some finish.")
        todo_now, todo_later = todo[:left], todo[left:]
    else:
        todo_now, todo_later = todo, []
    print(f"\nsubmitting {len(todo_now)} of {len(plans)} trials on {nodes} nodes each ({len(done)} were queued before) ...")
    failed = 0
    for path, plan in todo_now:
        cmd = ["./submit.sh", str(path), str(nodes)] + ([str(hours)] if hours else [])
        res = subprocess.run(cmd, cwd=spec_path.parent, capture_output=True, text=True)
        line = next((l for l in res.stdout.splitlines() if l.startswith("submitted job")), "")
        if res.returncode != 0 or not line:
            failed += 1
            print(f"  {plan['name']}: NOT submitted\n    " + (res.stdout + res.stderr).strip().replace("\n", "\n    ")[-400:])
        else:
            done[plan["name"]] = line
            done_file.write_text(json.dumps(done, indent=1))
            print(f"  {plan['name']}: {line}")
    if todo_later:
        print(f"\n{len(todo_later)} trial(s) not queued yet. When some of your jobs finish, run again:"
              f"\n  python -m bhaskera.launcher.sweep submit {spec_path}")
    print(f"\nCompare them in MLflow (experiment bhaskera, tag group = {spec['name']}), or run:"
          f"\n  python -m bhaskera.launcher.sweep report {spec['name']}")
    return 1 if failed else 0


def runs_in_group(uri: str, name: str) -> list[str]:
    """Ids of the runs whose tag ``group`` is ``name``. On a file store this reads one small tag file per run
    instead of loading every run through MLflow, which takes minutes once the store holds hundreds of runs."""
    if uri.startswith("file://"):
        store = Path(uri[len("file://"):])
        found = []
        for tag in store.glob("*/*/tags/group"):
            try:
                if tag.read_text().strip() == name:
                    found.append(tag.parent.parent.name)
            except OSError:
                continue
        return found
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    return [r.info.run_id for e in client.search_experiments()
            for r in client.search_runs([e.experiment_id], filter_string=f"tags.group = '{name}'", max_results=500)]


def cmd_report(name: str, store: str | None, top: int) -> int:
    import os
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    from mlflow.tracking import MlflowClient
    uri = store or os.environ.get("MLFLOW_TRACKING_URI") or f"file://{Path.home() / 'mlflow-runs'}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    rows = []
    for run_id in runs_in_group(uri, name):
        for run in [client.get_run(run_id)]:
            history = client.get_metric_history(run.info.run_id, "eval_loss")
            best = min((m.value for m in history), default=None)
            p = run.data.params
            rows.append({"run": run.data.tags.get("mlflow.runName", run.info.run_id[:8]), "status": run.info.status,
                         "best_eval_loss": best, "eval_accuracy": run.data.metrics.get("eval_accuracy"),
                         "loss": run.data.metrics.get("loss"),
                         "lr": p.get("training.lr"), "lora.r": p.get("lora.r")})
    if not rows:
        print(f"no runs with group '{name}' in {uri}")
        return 1
    rows.sort(key=lambda r: (r["best_eval_loss"] is None, r["best_eval_loss"] or 0))
    fmt = lambda v: "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    cols = ["run", "status", "best_eval_loss", "eval_accuracy", "loss", "lr", "lora.r"]
    table_rows = [cols] + [[fmt(r[c]) for c in cols] for r in rows[:top]]
    widths = [max(len(r[i]) for r in table_rows) for i in range(len(cols))]
    print("\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in table_rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bhaskera-sweep", description="Hyperparameter sweeps.")
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("plan", "submit"):
        sub.add_parser(action).add_argument("spec")
    r = sub.add_parser("report")
    r.add_argument("name")
    r.add_argument("--store", help="MLflow store (default: ~/mlflow-runs)")
    r.add_argument("--top", type=int, default=20)
    args = p.parse_args(argv)
    if args.action == "report":
        return cmd_report(args.name, args.store, args.top)
    return cmd_plan(Path(args.spec).resolve(), submit=args.action == "submit")


if __name__ == "__main__":
    sys.exit(main())
