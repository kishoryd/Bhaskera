"""Judge a fine-tuned model with the standard benchmarks (EleutherAI lm-evaluation-harness), next to the base model.

    python -m bhaskera.launcher.evaluate run --run <mlflow run id or registered name> --tasks arc_easy,hellaswag --limit 200
    python -m bhaskera.launcher.evaluate run --model <base model folder> --adapter <adapter folder> --tasks mmlu
    python -m bhaskera.launcher.evaluate prepare --tasks arc_easy,hellaswag      # download the datasets (needs internet)
    python -m bhaskera.launcher.evaluate show results.json                       # print a saved comparison again
    python -m bhaskera.launcher.evaluate init-task --name my_test --data my.jsonl  # a test from your own question/answer file
    python -m bhaskera.launcher.evaluate run ... --tasks my_test --save-samples    # keep every example's answer
    python -m bhaskera.launcher.evaluate review lm_eval_results_samples            # what got fixed, broken, or stays wrong

``run`` evaluates the base model and the tuned model (base + adapter) on the same tasks, prints a table with the change and
whether it is larger than the noise, saves ``results.json``, and, when a run is named, stores the numbers and the table
with that MLflow run (metrics ``lmeval_<task>_<metric>_base``, ``..._tuned`` and ``..._delta``). With ``--run`` the base model
and the adapter are found from the training run itself (the adapter that ``logging.mlflow_log_adapter`` stored).

Your own tests: ``init-task`` writes a task definition for a JSONL file of questions and answers into a ``tasks`` folder. A ``tasks``
folder in the current directory, ``$BHASKERA_TASKS_DIR`` or ``--tasks-dir`` is added to the harness's own tasks, so ``--tasks my_test`` works.
``--save-samples`` keeps each example's prompt, expected answer and response for the base and the tuned model, and ``review`` sorts them.

It needs the harness (``pip install "lm-eval>=0.4.5"``, installed by ``install-bhaskera.sh``) and the task datasets in the
Hugging Face cache. Compute nodes usually have no internet: run ``prepare`` once on a machine that has it. Use ``slurm/eval.sbatch``
(``./submit.sh eval ...``) to run it on a GPU node.
"""
from __future__ import annotations

from bhaskera import paths

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


def tasks_dir(given: Optional[str] = None) -> Optional[str]:
    """The folder with the user's own task definitions: --tasks-dir, $BHASKERA_TASKS_DIR, or ./tasks when it exists."""
    if given:
        if not Path(given).is_dir():
            raise EvalError(f"the tasks folder {given} does not exist")
        return str(Path(given).resolve())
    for c in (os.environ.get("BHASKERA_TASKS_DIR"), "tasks"):
        if c and Path(c).is_dir():
            return str(Path(c).resolve())
    return None


def make_manager(tasks_mod, include: Optional[str]):
    return tasks_mod.TaskManager(include_path=include) if include else tasks_mod.TaskManager()


def check_tasks(names: list[str], known: Optional[list[str]] = None, include: Optional[str] = None) -> None:
    """Refuse a task name that does not exist, with the closest real name."""
    if known is None:
        _ev, tasks = load_harness()
        known = list(make_manager(tasks, include).all_tasks)
    unknown = [n for n in names if n not in known]
    if unknown:
        lines = []
        for n in unknown:
            near = difflib.get_close_matches(n, known, n=3, cutoff=0.6)
            lines.append(f"'{n}' is not a task" + (f" (did you mean {', '.join(near)}?)" if near else ""))
        raise EvalError("; ".join(lines))


def evaluate_model(evaluator, args: str, tasks: list[str], limit: Optional[int], num_fewshot: Optional[int],
                   batch_size: str, device: Optional[str] = None, task_manager=None, log_samples: bool = False) -> dict:
    """One benchmark run; returns the harness' full result dict."""
    extra = {"task_manager": task_manager} if task_manager is not None else {}
    return evaluator.simple_evaluate(model="hf", model_args=args, tasks=tasks, num_fewshot=num_fewshot, batch_size=batch_size,
                                     limit=limit, device=device, log_samples=log_samples, **extra)


def sample_records(task: str, samples: list[dict], metric: str) -> list[dict]:
    """One compact line per example: id, prompt, expected answer, the model's response and its score."""
    out = []
    for i, smp in enumerate(samples or []):
        arg = (smp.get("arguments") or [[None]])
        arg0 = arg[0] if isinstance(arg, list) and arg else arg
        prompt = arg0[0] if isinstance(arg0, (list, tuple)) and arg0 else smp.get("doc")
        resp = smp.get("filtered_resps") or smp.get("resps") or [""]
        while isinstance(resp, (list, tuple)) and resp:
            resp = resp[0]
        score = smp.get(metric)
        out.append({"task": task, "id": smp.get("doc_id", i), "prompt": prompt if isinstance(prompt, str) else json.dumps(prompt, default=str),
                    "target": smp.get("target"), "response": resp if isinstance(resp, str) else json.dumps(resp, default=str),
                    "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None})
    return out


def save_samples(side: str, result: dict, flat: dict, folder: str | Path) -> int:
    """Write samples_<task>_<side>.jsonl for every task of a harness result; returns the number of examples."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for task, smp in (result.get("samples") or {}).items():
        metric = (flat.get(task) or {}).get("metric", "")
        rows = sample_records(task, smp, metric)
        with open(folder / f"samples_{_clean(task)}_{side}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n += len(rows)
    return n


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
                  device: Optional[str] = None, skip_base: bool = False, evaluator=None, task_manager=None,
                  samples_dir: Optional[str] = None, log: Callable[[str], None] = lambda s: print(s, flush=True)) -> dict:
    """The base model and the tuned model on the same tasks. Returns everything that is saved."""
    if evaluator is None:
        evaluator, _ = load_harness()
    out: dict[str, Any] = {"model": model, "adapter": adapter, "tasks": tasks, "limit": limit, "num_fewshot": num_fewshot}
    if not skip_base:
        log(f"== base model: {model}")
        base = evaluate_model(evaluator, model_args(model, None, dtype, four_bit), tasks, limit, num_fewshot, batch_size, device,
                              task_manager, bool(samples_dir))
        out["base"] = flatten(base.get("results"))
        if samples_dir:
            save_samples("base", base, out["base"], samples_dir)
        free_gpu()
    if adapter:
        log(f"== tuned model: {model} + {adapter}")
        tuned = evaluate_model(evaluator, model_args(model, adapter, dtype, four_bit), tasks, limit, num_fewshot, batch_size, device,
                               task_manager, bool(samples_dir))
        out["tuned"] = flatten(tuned.get("results"))
        if samples_dir:
            save_samples("tuned", tuned, out["tuned"], samples_dir)
        free_gpu()
    if "base" in out and "tuned" in out:
        out["comparison"] = compare(out["base"], out["tuned"])
    return out


def load_samples(folder: str | Path, side: str) -> dict[str, dict]:
    """{task: {id: record}} from the samples_*_<side>.jsonl files of a folder."""
    out: dict[str, dict] = {}
    for f in sorted(Path(folder).glob(f"samples_*_{side}.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["task"], {})[r["id"]] = r
    return out


def review(folder: str | Path, threshold: float = 0.5) -> dict[str, dict]:
    """Per task: how many examples tuning fixed, broke, left wrong, and left right, and the records of the last three groups'
    interesting cases ('broken' and 'still_wrong')."""
    base, tuned = load_samples(folder, "base"), load_samples(folder, "tuned")
    if not tuned:
        raise EvalError(f"no samples_*_tuned.jsonl in {folder}: run with --save-samples and an adapter")
    out = {}
    for task, rows in tuned.items():
        groups = {"fixed": [], "broken": [], "still_wrong": [], "still_right": []}
        for i, t in rows.items():
            b = (base.get(task) or {}).get(i)
            t_ok = t["score"] is not None and t["score"] >= threshold
            b_ok = None if b is None or b["score"] is None else b["score"] >= threshold
            if t_ok:
                groups["still_right" if b_ok in (True, None) else "fixed"].append({**t, "base_response": b and b["response"]})
            else:
                groups["broken" if b_ok else "still_wrong"].append({**t, "base_response": b and b["response"]})
        out[task] = groups
    return out


def write_review(summary: dict[str, dict], folder: str | Path) -> list[str]:
    """review_<task>.jsonl with the broken and still-wrong examples: what to look at, and what to correct in the data."""
    written = []
    for task, g in summary.items():
        rows = [{"group": name, **r} for name in ("broken", "still_wrong") for r in g[name]]
        if rows:
            path = Path(folder) / f"review_{_clean(task)}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            written.append(str(path))
    return written


def cmd_review(args) -> int:
    summary = review(args.folder, args.threshold)
    for task, g in summary.items():
        n = sum(len(v) for v in g.values())
        print(f"\n{task}: {n} examples | right now {len(g['still_right']) + len(g['fixed'])} | fixed by tuning {len(g['fixed'])}"
              f" | broken by tuning {len(g['broken'])} | still wrong {len(g['still_wrong'])}")
        for name, label in (("broken", "BROKEN (the base model had it right)"), ("still_wrong", "STILL WRONG")):
            for r in g[name][:args.show]:
                print(f"  {label}\n    prompt:   {r['prompt'][:200]!r}\n    expected: {str(r['target'])[:120]!r}\n    answered: {r['response'][:160]!r}")
    for path in write_review(summary, args.folder):
        print(f"wrote {path}")
    return 0


_TASK_TEMPLATE = """# Your own test, written by: python -m bhaskera.launcher.evaluate init-task
# Each line of the data file has a question and an answer. The model is asked "Question: ... Answer:" and passes an example when the
# first line of its reply contains the expected answer (ignoring case and punctuation). The check is in {name}_utils.py next to this
# file: change it, or `metric_list`, for other kinds of tests (the harness's documentation describes them). Run it with:
#   ./submit.sh eval --run <run id> --tasks {name} --save-samples
task: {name}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: generate_until
doc_to_text: "Question: {{{{{question}}}}}\\nAnswer:"
doc_to_target: "{{{{{answer}}}}}"
process_results: !function {name}_utils.process_results
generation_kwargs:
  until: ["\\n"]
  do_sample: false
  max_gen_toks: 64
metric_list:
  - metric: contains
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

_UTILS_TEMPLATE = """import re


def _words(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def process_results(doc, results):
    got, want = _words(results[0]), _words(doc["{answer}"])
    return {{"contains": float(bool(want) and want in got)}}
"""


def cmd_init_task(args) -> int:
    import re
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,60}", args.name):
        raise EvalError("the test name is letters, digits and _, starting with a letter")
    data = Path(args.data).resolve()
    if not data.is_file():
        raise EvalError(f"the data file {args.data} does not exist")
    first = json.loads(next(l for l in data.read_text().splitlines() if l.strip()))
    for field in (args.question_field, args.answer_field):
        if field not in first:
            raise EvalError(f"the first line of {args.data} has no '{field}' (it has: {', '.join(first)}); use --question-field / --answer-field")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.yaml"
    path.write_text(_TASK_TEMPLATE.format(name=args.name, data=data, question=args.question_field, answer=args.answer_field))
    (out / f"{args.name}_utils.py").write_text(_UTILS_TEMPLATE.format(answer=args.answer_field))
    print(f"wrote {path}\nrun it with:  ./submit.sh eval --run <run id> --tasks {args.name} --save-samples")
    return 0


def _clean(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def resolve_run(store: str | Path, ref: str) -> str:
    """A run id as it is, or the run behind a registry name (name, name:3, name@champion)."""
    from .register import RUN_ID, RegisterError, resolve_in_store
    if RUN_ID.match(ref or "") or list(Path(store).glob(f"*/{ref}/meta.yaml")):
        return ref
    try:
        return resolve_in_store(str(store), ref)
    except RegisterError as e:
        raise EvalError(f"no run {ref} in {store}, and {e}")


def from_run(store: str | Path, run_id: str) -> tuple[str, Optional[str]]:
    """(base model path, adapter folder) of a training run in an MLflow file store."""
    store = Path(store)
    hits = list(store.glob(f"*/{run_id}/meta.yaml"))
    if not hits:                                    # not a run id: maybe a name in the model registry (name, name:3, name@champion)
        from .register import RegisterError, resolve_in_store
        try:
            hits = list(store.glob(f"*/{resolve_in_store(str(store), run_id)}/meta.yaml"))
        except RegisterError as e:
            raise EvalError(f"no run {run_id} in {store}, and {e}")
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


def log_to_mlflow(result: dict, store: str, run_id: Optional[str], experiment: str = "bhaskera-eval", name: str = "lm-eval",
                  samples_dir: Optional[str] = None) -> str:
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
    if samples_dir and Path(samples_dir).is_dir():
        client.log_artifacts(run_id, str(samples_dir), "lm_eval/samples")
    return run_id


def _store_default() -> str:
    return paths.mlflow_store()


def cmd_run(args) -> int:
    tasks = [t for t in args.tasks.split(",") if t]
    if not tasks:
        raise EvalError("give at least one task with --tasks")
    model, adapter = args.model, args.adapter
    if args.run:
        args.run = resolve_run(args.store, args.run)        # the result is stored with this run, so it must be an id, not a registry name
        m, a = from_run(args.store, args.run)
        model, adapter = model or m, adapter or a
    if not model:
        raise EvalError("give --model <base model folder>, or --run <mlflow run id>")
    if not adapter and not args.skip_base:
        print("no adapter given: only the base model is evaluated")
    include = tasks_dir(args.tasks_dir)
    check_tasks(tasks, include=include)
    evaluator, tasks_mod = load_harness()
    samples_dir = (args.samples_dir or str(Path(args.out).with_suffix("")) + "_samples") if args.save_samples else None
    result = evaluate_pair(model, adapter, tasks, limit=args.limit, num_fewshot=args.num_fewshot, batch_size=args.batch_size,
                           dtype=args.dtype, four_bit=args.four_bit, device=args.device, skip_base=args.skip_base, evaluator=evaluator,
                           task_manager=make_manager(tasks_mod, include) if include else None, samples_dir=samples_dir)
    print()
    print(table(result.get("comparison", [])) if "comparison" in result else
          "\n".join(f"{t}: {r['metric']} = {r['value']:.4f}" for side in ("base", "tuned") for t, r in (result.get(side) or {}).items()))
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"\nsaved {args.out}")
    if samples_dir:
        print(f"every example's answer is in {samples_dir}; see what changed with:  python -m bhaskera.launcher.evaluate review {samples_dir}")
    if args.run or args.mlflow:
        rid = log_to_mlflow(result, args.store, args.run, samples_dir=samples_dir)
        print(f"stored with MLflow run {rid}")
    return 0


def cmd_prepare(args) -> int:
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    tasks = [t for t in args.tasks.split(",") if t]
    _ev, tasks_mod = load_harness()
    include = tasks_dir(args.tasks_dir)
    check_tasks(tasks, include=include)
    manager = make_manager(tasks_mod, include)
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
    paths.apply_cache_env()                      # benchmark data lives in the run folder, unless the user keeps it at home already
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
    r.add_argument("--tasks-dir", dest="tasks_dir", help="folder with your own task definitions (default: ./tasks or $BHASKERA_TASKS_DIR)")
    r.add_argument("--save-samples", action="store_true", dest="save_samples", help="keep every example's prompt, answer and response, to review mistakes")
    r.add_argument("--samples-dir", dest="samples_dir", help="where --save-samples writes (default: <out>_samples)")
    r.add_argument("--out", default="lm_eval_results.json")
    r.add_argument("--store", default=_store_default(), help="MLflow store (default ~/mlflow-runs)")
    r.add_argument("--mlflow", action="store_true", help="store the result in a new MLflow run when --run is not given")
    pr = sub.add_parser("prepare")
    pr.add_argument("--tasks", default=DEFAULT_TASKS)
    pr.add_argument("--tasks-dir", dest="tasks_dir")
    it = sub.add_parser("init-task", help="write a test definition for your own question/answer JSONL file")
    it.add_argument("--name", required=True)
    it.add_argument("--data", required=True, help="JSONL file, one example per line")
    it.add_argument("--question-field", default="question", dest="question_field")
    it.add_argument("--answer-field", default="answer", dest="answer_field")
    it.add_argument("--out", default="tasks", help="folder for the task definition (default ./tasks)")
    rv = sub.add_parser("review", help="what tuning fixed, broke, or left wrong, from the saved samples")
    rv.add_argument("folder", help="the folder --save-samples wrote")
    rv.add_argument("--show", type=int, default=3, help="examples to print per group (default 3)")
    rv.add_argument("--threshold", type=float, default=0.5, help="a score at or above this counts as right")
    sh = sub.add_parser("show")
    sh.add_argument("file")
    args = p.parse_args(argv)
    try:
        return {"run": cmd_run, "prepare": cmd_prepare, "show": cmd_show, "init-task": cmd_init_task, "review": cmd_review}[args.action](args)
    except EvalError as e:
        print(f"error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
