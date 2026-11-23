"""Benchmarking a tuned model against its base: result handling, the comparison, MLflow and the command line.
The harness itself is replaced by a fake with the same result shape, so no GPU, model or dataset is needed."""
from __future__ import annotations

import json

import pytest

from bhaskera.launcher import evaluate as ev

BASE = {"results": {
    "arc_easy": {"alias": "arc_easy", "acc,none": 0.700, "acc_stderr,none": 0.010, "acc_norm,none": 0.650, "acc_norm_stderr,none": 0.010},
    "hellaswag": {"alias": "hellaswag", "acc,none": 0.600, "acc_stderr,none": 0.010, "acc_norm,none": 0.780, "acc_norm_stderr,none": 0.010},
    "wikitext": {"alias": "wikitext", "word_perplexity,none": 12.0, "word_perplexity_stderr,none": "N/A"},
}}
TUNED = {"results": {
    "arc_easy": {"alias": "arc_easy", "acc,none": 0.760, "acc_stderr,none": 0.010, "acc_norm,none": 0.700, "acc_norm_stderr,none": 0.010},
    "hellaswag": {"alias": "hellaswag", "acc,none": 0.602, "acc_stderr,none": 0.010, "acc_norm,none": 0.781, "acc_norm_stderr,none": 0.010},
    "wikitext": {"alias": "wikitext", "word_perplexity,none": 9.0, "word_perplexity_stderr,none": "N/A"},
}}


class FakeHarness:
    def __init__(self):
        self.calls = []

    def simple_evaluate(self, model, model_args, tasks, num_fewshot, batch_size, limit, device, log_samples):
        self.calls.append((model, model_args, tasks, limit, batch_size))
        return TUNED if "peft=" in model_args else BASE


def test_the_metric_shown_per_task_is_the_usual_one():
    flat = ev.flatten(BASE["results"])
    assert flat["arc_easy"] == {"metric": "acc_norm", "value": 0.65, "stderr": 0.01}          # acc_norm is preferred over acc
    assert flat["wikitext"]["metric"] == "word_perplexity" and flat["wikitext"]["stderr"] is None   # "N/A" is not a number
    assert ev.primary_metric({"alias": "x"}) is None and ev.primary_metric({"foo,none": 1.0})[0] == "foo"
    assert ev.primary_metric({"sample_len": 5, "alias": "x"}) is None          # a count in the entry is not a score
    real = {"name": "arc_easy", "alias": "arc_easy", "sample_len": 5, "acc,none": 0.2, "acc_stderr,none": 0.2}
    assert ev.primary_metric(real) == ("acc", 0.2, 0.2)


def test_comparison_marks_only_changes_bigger_than_the_noise():
    rows = {r["task"]: r for r in ev.compare(ev.flatten(BASE["results"]), ev.flatten(TUNED["results"]))}
    assert rows["arc_easy"]["verdict"] == "better" and rows["arc_easy"]["significant"] and round(rows["arc_easy"]["delta"], 3) == 0.05
    assert rows["hellaswag"]["verdict"] == "no clear change" and not rows["hellaswag"]["significant"]
    assert rows["wikitext"]["verdict"] == "no clear change"                      # no standard error: nothing can be called clear
    text = ev.table(list(rows.values()))
    assert "arc_easy" in text and "+0.0500" in text and "better" in text


def test_lower_perplexity_counts_as_better_and_a_drop_in_accuracy_as_worse():
    b = {"t": {"metric": "word_perplexity", "value": 12.0, "stderr": 0.1}}
    t = {"t": {"metric": "word_perplexity", "value": 9.0, "stderr": 0.1}}
    assert ev.compare(b, t)[0]["verdict"] == "better"
    b = {"t": {"metric": "acc", "value": 0.70, "stderr": 0.01}}
    t = {"t": {"metric": "acc", "value": 0.60, "stderr": 0.01}}
    assert ev.compare(b, t)[0]["verdict"] == "worse"


def test_model_args_and_their_limits():
    assert ev.model_args("/m/base") == "pretrained=/m/base,dtype=bfloat16"
    assert ev.model_args("/m/base", "/a/ad", "float16", True) == "pretrained=/m/base,dtype=float16,peft=/a/ad,load_in_4bit=True"
    with pytest.raises(ev.EvalError, match="comma"):
        ev.model_args("/m/ba,se")


def test_both_models_are_evaluated_on_the_same_tasks():
    fake, lines = FakeHarness(), []
    out = ev.evaluate_pair("/m/base", "/a/ad", ["arc_easy", "wikitext"], limit=50, evaluator=fake, log=lines.append)
    assert [c[1].count("peft=") for c in fake.calls] == [0, 1] and all(c[2] == ["arc_easy", "wikitext"] and c[3] == 50 for c in fake.calls)
    assert set(out) >= {"base", "tuned", "comparison"} and lines[0].startswith("== base model")
    only = ev.evaluate_pair("/m/base", "/a/ad", ["arc_easy"], skip_base=True, evaluator=FakeHarness(), log=lambda s: None)
    assert "base" not in only and "comparison" not in only and "tuned" in only


def test_unknown_task_names_are_refused_with_a_suggestion():
    ev.check_tasks(["arc_easy"], known=["arc_easy", "arc_challenge", "hellaswag"])
    with pytest.raises(ev.EvalError, match=r"'arc_eazy' is not a task \(did you mean arc_easy"):
        ev.check_tasks(["arc_eazy"], known=["arc_easy", "arc_challenge", "hellaswag"])


def _store(tmp_path, with_adapter=True, with_model=True):
    run = tmp_path / "store" / "1" / "abc123"
    (run / "params").mkdir(parents=True)
    (run / "meta.yaml").write_text("run_name: train\nstatus: 3\n")
    if with_model:
        (run / "params" / "model.name").write_text("/models/Qwen3-14B")
    if with_adapter:
        (run / "artifacts" / "adapter").mkdir(parents=True)
        (run / "artifacts" / "adapter" / "adapter_config.json").write_text("{}")
    return tmp_path / "store"


def test_a_training_run_tells_the_base_model_and_the_adapter(tmp_path):
    store = _store(tmp_path)
    assert ev.from_run(store, "abc123") == ("/models/Qwen3-14B", str(store / "1" / "abc123" / "artifacts" / "adapter"))
    with pytest.raises(ev.EvalError, match="no run"):
        ev.from_run(store, "nope")
    with pytest.raises(ev.EvalError, match="no stored adapter"):
        ev.from_run(_store(tmp_path / "x", with_adapter=False), "abc123")
    with pytest.raises(ev.EvalError, match="model.name"):
        ev.from_run(_store(tmp_path / "y", with_model=False), "abc123")


def test_results_are_stored_with_the_training_run(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    store = tmp_path / "mlruns"
    uri = f"file://{store}"
    client = mlflow.tracking.MlflowClient(uri, uri)
    exp = client.create_experiment("bhaskera")
    run_id = client.create_run(exp, run_name="train").info.run_id
    result = ev.evaluate_pair("/m/base", "/a/ad", ["arc_easy", "wikitext"], evaluator=FakeHarness(), log=lambda s: None)
    assert ev.log_to_mlflow(result, str(store), run_id) == run_id
    metrics = client.get_run(run_id).data.metrics
    assert metrics["lmeval_arc_easy_acc_norm_base"] == 0.65 and metrics["lmeval_arc_easy_acc_norm_tuned"] == 0.70
    assert round(metrics["lmeval_arc_easy_acc_norm_delta"], 3) == 0.05 and "lmeval_wikitext_word_perplexity_delta" in metrics
    files = {a.path for a in client.list_artifacts(run_id, "lm_eval")}
    assert files == {"lm_eval/comparison.json", "lm_eval/results.json"}
    new_id = ev.log_to_mlflow(result, str(store), None, experiment="bhaskera-eval")          # without a run: a new one
    assert new_id != run_id and client.get_experiment_by_name("bhaskera-eval") is not None


def test_command_line_end_to_end(tmp_path, monkeypatch, capsys):
    store = _store(tmp_path)
    fake = FakeHarness()
    monkeypatch.setattr(ev, "load_harness", lambda: (fake, None))
    monkeypatch.setattr(ev, "check_tasks", lambda names, known=None, include=None: None)
    monkeypatch.setattr(ev, "free_gpu", lambda: None)
    monkeypatch.setattr(ev, "log_to_mlflow", lambda result, store, run_id, **k: run_id or "new")
    out = tmp_path / "r.json"
    code = ev.main(["run", "--run", "abc123", "--store", str(store), "--tasks", "arc_easy,wikitext", "--limit", "20", "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and "arc_easy" in text and "better" in text and "stored with MLflow run abc123" in text
    saved = json.loads(out.read_text())
    assert saved["model"] == "/models/Qwen3-14B" and saved["limit"] == 20 and saved["comparison"]
    assert ev.main(["show", str(out)]) == 0 and "arc_easy" in capsys.readouterr().out


def test_command_line_problems_are_explained_not_dumped(capsys, monkeypatch):
    monkeypatch.setattr(ev, "load_harness", lambda: (FakeHarness(), None))
    assert ev.main(["run", "--tasks", "arc_easy"]) == 1 and "--model" in capsys.readouterr().out
    monkeypatch.setattr(ev, "load_harness", lambda: (_ for _ in ()).throw(ev.EvalError("the evaluation harness is not installed")))
    assert ev.main(["prepare", "--tasks", "arc_easy"]) == 1 and "not installed" in capsys.readouterr().out


# ---- your own tests, saved examples, and the review ------------------------------------------------------------------
import yaml  # noqa: E402


def test_init_task_writes_a_valid_definition(tmp_path, capsys):
    data = tmp_path / "mine.jsonl"
    data.write_text('{"question": "2+2?", "answer": "4"}\n{"question": "3+3?", "answer": "6"}\n')
    assert ev.main(["init-task", "--name", "my_test", "--data", str(data), "--out", str(tmp_path / "tasks")]) == 0
    spec = yaml.safe_load((tmp_path / "tasks" / "my_test.yaml").read_text().replace("!function", "!!str"))
    assert spec["task"] == "my_test" and spec["doc_to_text"] == "Question: {{question}}\nAnswer:" and spec["doc_to_target"] == "{{answer}}"
    assert spec["dataset_kwargs"]["data_files"]["test"] == str(data) and spec["generation_kwargs"]["until"] == ["\n"]
    assert spec["metric_list"][0]["metric"] == "contains" and spec["process_results"] == "my_test_utils.process_results"
    import importlib.util
    sp = importlib.util.spec_from_file_location("my_test_utils", tmp_path / "tasks" / "my_test_utils.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    assert mod.process_results({"answer": "4"}, [" The answer is 4."]) == {"contains": 1.0} and mod.process_results({"answer": "4"}, [" What is 5+5?"]) == {"contains": 0.0}
    assert mod.process_results({"answer": "High performance computing"}, [" High-performance computing."]) == {"contains": 1.0}      # a hyphen is a space, not nothing
    assert "my_test" in capsys.readouterr().out


def test_init_task_names_the_problem(tmp_path, capsys):
    data = tmp_path / "x.jsonl"
    data.write_text('{"q": "a", "a": "b"}\n')
    assert ev.main(["init-task", "--name", "t", "--data", str(data), "--out", str(tmp_path)]) == 1
    assert "no 'question'" in capsys.readouterr().out
    assert ev.main(["init-task", "--name", "bad name", "--data", str(data), "--out", str(tmp_path)]) == 1
    assert ev.main(["init-task", "--name", "t", "--data", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path)]) == 1


def test_tasks_dir_is_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BHASKERA_TASKS_DIR", raising=False)
    assert ev.tasks_dir() is None
    (tmp_path / "tasks").mkdir()
    assert ev.tasks_dir() == str((tmp_path / "tasks").resolve())
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.setenv("BHASKERA_TASKS_DIR", str(other))
    assert ev.tasks_dir() == str(other.resolve())
    with pytest.raises(ev.EvalError):
        ev.tasks_dir(str(tmp_path / "nope"))


def _sample(i, ok, resp):
    return {"doc_id": i, "arguments": [[f"question {i}", {}]], "target": f"answer {i}", "filtered_resps": [resp], "exact_match": 1.0 if ok else 0.0}


class FakeSamples:
    """A harness that returns the same examples every time, right or wrong depending on the model."""
    def simple_evaluate(self, model, model_args, tasks, log_samples, **kw):
        tuned = "peft" in model_args
        right = {0: True, 1: False, 2: False, 3: True} if not tuned else {0: True, 1: True, 2: False, 3: False}
        samples = [_sample(i, ok, "x") for i, ok in right.items()]
        return {"results": {"my_test": {"exact_match,none": sum(right.values()) / 4, "alias": "my_test"}},
                "samples": {"my_test": samples} if log_samples else {}}


def test_samples_are_saved_and_reviewed(tmp_path, capsys):
    out = ev.evaluate_pair("/m", "/a", ["my_test"], evaluator=FakeSamples(), samples_dir=str(tmp_path), log=lambda s: None)
    assert out["comparison"][0]["task"] == "my_test"
    assert (tmp_path / "samples_my_test_base.jsonl").exists() and (tmp_path / "samples_my_test_tuned.jsonl").exists()
    summary = ev.review(tmp_path)["my_test"]
    assert [r["id"] for r in summary["fixed"]] == [1] and [r["id"] for r in summary["broken"]] == [3]
    assert [r["id"] for r in summary["still_wrong"]] == [2] and [r["id"] for r in summary["still_right"]] == [0]
    assert summary["broken"][0]["prompt"] == "question 3" and summary["broken"][0]["base_response"] == "x"
    assert ev.main(["review", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "fixed by tuning 1" in text and "broken by tuning 1" in text and "still wrong 1" in text
    rows = [json.loads(l) for l in (tmp_path / "review_my_test.jsonl").read_text().splitlines()]
    assert {(r["group"], r["id"]) for r in rows} == {("broken", 3), ("still_wrong", 2)}


def test_review_needs_tuned_samples(tmp_path):
    with pytest.raises(ev.EvalError):
        ev.review(tmp_path)
