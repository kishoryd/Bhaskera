"""Judge a fine-tuned model with the standard benchmarks (EleutherAI lm-evaluation-harness), next to the base model.

    python -m bhaskera.launcher.evaluate run --run <mlflow run id> --tasks arc_easy,hellaswag --limit 200
    python -m bhaskera.launcher.evaluate run --model <base model folder> --adapter <adapter folder> --tasks mmlu
    python -m bhaskera.launcher.evaluate prepare --tasks arc_easy,hellaswag      # download the datasets (needs internet)
    python -m bhaskera.launcher.evaluate show results.json                       # print a saved comparison again

``run`` evaluates the base model and the tuned model (base + adapter) on the same tasks, prints a table with the change and
whether it is larger than the noise, saves ``results.json``, and, when a run is named, stores the numbers and the table
with that MLflow run (metrics ``lmeval_<task>_<metric>_base``, ``..._tuned`` and ``..._delta``). With ``--run`` the base model
and the adapter are found from the training run itself (the adapter that ``logging.mlflow_log_adapter`` stored).

It needs the harness (``pip install "lm-eval>=0.4.5"``, installed by ``install-bhaskera.sh``) and the task datasets in the
Hugging Face cache. Compute nodes usually have no internet: run ``prepare`` once on a machine that has it. Use ``slurm/eval.sbatch``
(``./submit.sh eval ...``) to run it on a GPU node.
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_TASKS = "arc_easy,hellaswag"
_PREFERRED = ("acc_norm", "acc", "exact_match", "pass@1", "f1", "bleu", "rouge1", "word_perplexity", "perplexity")
_LOWER_IS_BETTER = ("perplexity", "loss", "bits_per_byte", "wer", "ter")


class EvalError(Exception):
    """A problem the user can fix (shown without a traceback)."""


def primary_metric(entry: dict) -> Optional[tuple[str, float, Optional[float]]]:
    """(metric name, value, standard error) to show for one task's results, or None when it has no number."""
    # metric keys look like "acc,none" (metric, filter); other numbers in an entry, such as sample_len, are not scores
    numbers = {k: v for k, v in entry.items() if "," in k and isinstance(v, (int, float)) and not isinstance(v, bool) and "_stderr" not in k}
    if not numbers:
        return None
    for want in _PREFERRED:
        for key in numbers:
            if key.split(",")[0] == want:
                se = entry.get(key.replace(",", "_stderr,", 1))
                return key.split(",")[0], float(numbers[key]), float(se) if isinstance(se, (int, float)) else None
    key = next(iter(numbers))
    return key.split(",")[0], float(numbers[key]), None


def flatten(results: dict) -> dict[str, dict]:
    """{task: {'metric': name, 'value': v, 'stderr': se}} from the ``results`` of ``simple_evaluate``."""
    out = {}
    for task, entry in (results or {}).items():
        pm = primary_metric(entry)
        if pm:
            out[task] = {"metric": pm[0], "value": pm[1], "stderr": pm[2]}
    return out


def higher_is_better(metric: str) -> bool:
    return not any(word in metric for word in _LOWER_IS_BETTER)


def compare(base: dict[str, dict], tuned: dict[str, dict]) -> list[dict]:
    """One row per task found in both: values, the change, and whether it is bigger than twice its standard error."""
    rows = []
    for task in sorted(set(base) & set(tuned)):
        b, t = base[task], tuned[task]
        delta = t["value"] - b["value"]
        noise = math.sqrt((b["stderr"] or 0) ** 2 + (t["stderr"] or 0) ** 2)
        known = b["stderr"] is not None and t["stderr"] is not None
        significant = known and abs(delta) > 2 * noise
        better = (delta > 0) == higher_is_better(b["metric"])
        rows.append({"task": task, "metric": b["metric"], "base": b["value"], "tuned": t["value"], "delta": delta,
                     "stderr": noise if known else None, "significant": bool(significant),
                     "verdict": ("better" if better else "worse") if significant else "no clear change"})
    return rows


def table(rows: list[dict]) -> str:
    if not rows:
        return "no task has a number in both runs"
    head = ["task", "metric", "base", "tuned", "change", "verdict"]
    body = [[r["task"], r["metric"], f"{r['base']:.4f}", f"{r['tuned']:.4f}", f"{r['delta']:+.4f}", r["verdict"]] for r in rows]
    widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    fmt = lambda row: "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip()
    return "\n".join([fmt(head), fmt(["-" * w for w in widths])] + [fmt(r) for r in body])


def model_args(model: str, adapter: Optional[str] = None, dtype: str = "bfloat16", four_bit: bool = False) -> str:
    """The ``model_args`` string of the harness' Hugging Face model."""
    for value in (model, adapter or ""):
        if "," in value:
            raise EvalError(f"a comma in {value!r} cannot be passed to the harness; use a path without commas")
    parts = [f"pretrained={model}", f"dtype={dtype}"]
    if adapter:
        parts.append(f"peft={adapter}")
    if four_bit:
        parts.append("load_in_4bit=True")
    return ",".join(parts)


def load_harness():
    try:
        from lm_eval import evaluator, tasks
    except ImportError as e:
        raise EvalError('the evaluation harness is not installed: pip install "lm-eval>=0.4.5" '
                        "(install-bhaskera.sh does it; it needs internet)") from e
    return evaluator, tasks


def check_tasks(names: list[str], known: Optional[list[str]] = None) -> None:
    """Refuse a task name that does not exist, with the closest real name."""
    if known is None:
        _ev, tasks = load_harness()
        known = list(tasks.TaskManager().all_tasks)
    unknown = [n for n in names if n not in known]
    if unknown:
        lines = []
        for n in unknown:
            near = difflib.get_close_matches(n, known, n=3, cutoff=0.6)
            lines.append(f"'{n}' is not a task" + (f" (did you mean {', '.join(near)}?)" if near else ""))
        raise EvalError("; ".join(lines))


def evaluate_model(evaluator, args: str, tasks: list[str], limit: Optional[int], num_fewshot: Optional[int],
                   batch_size: str, device: Optional[str] = None) -> dict:
    """One benchmark run; returns the harness' full result dict."""
    return evaluator.simple_evaluate(model="hf", model_args=args, tasks=tasks, num_fewshot=num_fewshot, batch_size=batch_size,
                                     limit=limit, device=device, log_samples=False)


def free_gpu() -> None:
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def evaluate_pair(model: str, adapter: Optional[str], tasks: list[str], *, limit: Optional[int] = None,
                  num_fewshot: Optional[int] = None, batch_size: str = "auto", dtype: str = "bfloat16", four_bit: bool = False,
                  device: Optional[str] = None, skip_base: bool = False, evaluator=None,
                  log: Callable[[str], None] = lambda s: print(s, flush=True)) -> dict:
    """The base model and the tuned model on the same tasks. Returns everything that is saved."""
    if evaluator is None:
        evaluator, _ = load_harness()
    out: dict[str, Any] = {"model": model, "adapter": adapter, "tasks": tasks, "limit": limit, "num_fewshot": num_fewshot}
    if not skip_base:
        log(f"== base model: {model}")
        base = evaluate_model(evaluator, model_args(model, None, dtype, four_bit), tasks, limit, num_fewshot, batch_size, device)
        out["base"] = flatten(base.get("results"))
        free_gpu()
    if adapter:
        log(f"== tuned model: {model} + {adapter}")
        tuned = evaluate_model(evaluator, model_args(model, adapter, dtype, four_bit), tasks, limit, num_fewshot, batch_size, device)
        out["tuned"] = flatten(tuned.get("results"))
        free_gpu()
    if "base" in out and "tuned" in out:
        out["comparison"] = compare(out["base"], out["tuned"])
    return out


def _clean(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def from_run(store: str | Path, run_id: str) -> tuple[str, Optional[str]]:
    """(base model path, adapter folder) of a training run in an MLflow file store."""
    store = Path(store)
    hits = list(store.glob(f"*/{run_id}/meta.yaml"))
    if not hits:
        raise EvalError(f"no run {run_id} in {store}")
    run = hits[0].parent
    model = ""
    try:
        model = (run / "params" / "model.name").read_text().strip()
    except OSError:
        pass
    adapter = run / "artifacts" / "adapter"
    if not model:
        raise EvalError(f"run {run_id} has no model.name parameter; give --model")
    if not (adapter / "adapter_config.json").exists():
        raise EvalError(f"run {run_id} has no stored adapter; train with logging.mlflow_log_adapter: true, or give --adapter")
    return model, str(adapter)


def log_to_mlflow(result: dict, store: str, run_id: Optional[str], experiment: str = "bhaskera-eval", name: str = "lm-eval") -> str:
    """Store metrics, the table and the full result with a run (an existing training run, or a new one). Returns the run id."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    from mlflow.tracking import MlflowClient
    uri = store if "://" in store else f"file://{Path(store).resolve()}"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    if run_id is None:
        exp = client.get_experiment_by_name(experiment)
        exp_id = exp.experiment_id if exp else client.create_experiment(experiment)
        run_id = client.create_run(exp_id, run_name=name).info.run_id
    for side in ("base", "tuned"):
        for task, r in (result.get(side) or {}).items():
            client.log_metric(run_id, f"lmeval_{_clean(task)}_{_clean(r['metric'])}_{side}", r["value"])
    for row in result.get("comparison", []):
        client.log_metric(run_id, f"lmeval_{_clean(row['task'])}_{_clean(row['metric'])}_delta", row["delta"])
    if result.get("comparison"):
        rows = result["comparison"]
        client.log_table(run_id, data={k: [r[k] for r in rows] for k in ("task", "metric", "base", "tuned", "delta", "verdict")},
                         artifact_file="lm_eval/comparison.json")
    client.log_dict(run_id, result, "lm_eval/results.json")
    return run_id


def _store_default() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "").replace("file://", "") or str(Path.home() / "mlflow-runs")


def cmd_run(args) -> int:
    tasks = [t for t in args.tasks.split(",") if t]
    if not tasks:
        raise EvalError("give at least one task with --tasks")
    model, adapter = args.model, args.adapter
    if args.run:
        m, a = from_run(args.store, args.run)
        model, adapter = model or m, adapter or a
    if not model:
        raise EvalError("give --model <base model folder>, or --run <mlflow run id>")
    if not adapter and not args.skip_base:
        print("no adapter given: only the base model is evaluated")
    check_tasks(tasks)
    result = evaluate_pair(model, adapter, tasks, limit=args.limit, num_fewshot=args.num_fewshot, batch_size=args.batch_size,
                           dtype=args.dtype, four_bit=args.four_bit, device=args.device, skip_base=args.skip_base)
    print()
    print(table(result.get("comparison", [])) if "comparison" in result else
          "\n".join(f"{t}: {r['metric']} = {r['value']:.4f}" for side in ("base", "tuned") for t, r in (result.get(side) or {}).items()))
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"\nsaved {args.out}")
    if args.run or args.mlflow:
        rid = log_to_mlflow(result, args.store, args.run)
        print(f"stored with MLflow run {rid}")
    return 0


def cmd_prepare(args) -> int:
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    tasks = [t for t in args.tasks.split(",") if t]
    _ev, tasks_mod = load_harness()
    check_tasks(tasks)
    manager = tasks_mod.TaskManager()
    for name in tasks:
        print(f"downloading {name} ...", flush=True)
        tasks_mod.get_task_dict([name], manager)            # building the task downloads its dataset into the cache
    print("done: the datasets are in the Hugging Face cache")
    return 0


def cmd_show(args) -> int:
    result = json.loads(Path(args.file).read_text())
    print(table(result.get("comparison", [])))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="bhaskera-evaluate", description="Benchmark a tuned model against its base model.")
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", help="MLflow run id of the training run: finds the base model and the adapter, stores the result there")
    r.add_argument("--model", help="base model folder")
    r.add_argument("--adapter", help="adapter folder (adapter_config.json, adapter_model.safetensors)")
    r.add_argument("--tasks", default=DEFAULT_TASKS, help=f"comma-separated harness tasks (default {DEFAULT_TASKS})")
    r.add_argument("--limit", type=int, help="use only this many examples per task (quick check)")
    r.add_argument("--num-fewshot", type=int, dest="num_fewshot")
    r.add_argument("--batch-size", default="auto", dest="batch_size")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--4bit", action="store_true", dest="four_bit", help="load the base model in 4 bit (as QLoRA trained it)")
    r.add_argument("--device")
    r.add_argument("--skip-base", action="store_true", dest="skip_base", help="evaluate only the tuned model")
    r.add_argument("--out", default="lm_eval_results.json")
    r.add_argument("--store", default=_store_default(), help="MLflow store (default ~/mlflow-runs)")
    r.add_argument("--mlflow", action="store_true", help="store the result in a new MLflow run when --run is not given")
    pr = sub.add_parser("prepare")
    pr.add_argument("--tasks", default=DEFAULT_TASKS)
    sh = sub.add_parser("show")
    sh.add_argument("file")
    args = p.parse_args(argv)
    try:
        return {"run": cmd_run, "prepare": cmd_prepare, "show": cmd_show}[args.action](args)
    except EvalError as e:
        print(f"error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
