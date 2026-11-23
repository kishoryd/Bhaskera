"""Hyperparameter sweeps: trial recipes, validation, submission and the report. No cluster needed."""
from __future__ import annotations

import subprocess

import pytest
import yaml

from bhaskera.launcher import sweep

RECIPE = {"model": {"name": "m"}, "training": {"lr": 1e-4, "max_steps": 10},
          "lora": {"enabled": True, "r": 16}, "checkpoint": {"save_dir": "/scratch/u/ckpt"},
          "logging": {"tracker": "mlflow", "run_name": "base", "tags": ["a:b"]}}


def test_grid_is_every_combination():
    sets = sweep.combinations({"training.lr": [1, 2, 3], "lora.r": [8, 16]})
    assert len(sets) == 6 and {"training.lr": 3, "lora.r": 16} in sets


def test_random_is_repeatable_and_has_no_repeats():
    params = {"training.lr": {"loguniform": [1e-5, 1e-3]}, "lora.r": [8, 16, 32]}
    a = sweep.combinations(params, "random", trials=5, seed=7)
    assert a == sweep.combinations(params, "random", trials=5, seed=7) and len(a) == 5
    assert len({repr(sorted(x.items())) for x in a}) == 5
    assert all(1e-5 <= x["training.lr"] <= 1e-3 for x in a)


def test_random_stops_when_the_space_is_used_up():
    assert len(sweep.combinations({"lora.r": [8, 16]}, "random", trials=10, seed=1)) == 2


def test_bad_specs_are_explained():
    with pytest.raises(ValueError, match="no params"):
        sweep.combinations({})
    with pytest.raises(ValueError, match="list of values"):
        sweep.combinations({"training.lr": {"uniform": [0, 1]}}, "grid")
    with pytest.raises(ValueError, match="trials"):
        sweep.combinations({"lora.r": [8]}, "random")
    with pytest.raises(ValueError, match="grid or random"):
        sweep.combinations({"lora.r": [8]}, "bayes")


def test_trials_get_their_own_name_group_tags_and_checkpoint_folder():
    plans = sweep.plan_trials({"name": "lr-search", "params": {"training.lr": [1e-4, 2e-4], "lora.r": [8]}}, RECIPE)
    assert [p["name"] for p in plans] == ["lr-search-01", "lr-search-02"]
    r = plans[1]["recipe"]
    assert r["training"]["lr"] == 2e-4 and r["lora"]["r"] == 8 and r["training"]["max_steps"] == 10
    assert r["logging"]["run_name"] == "lr-search-02" and r["logging"]["group"] == "lr-search"
    assert r["logging"]["tags"] == ["a:b", "sweep:lr-search", "trial:02"]
    assert r["checkpoint"]["save_dir"] == "/scratch/u/ckpt/lr-search/lr-search-02"
    assert RECIPE["training"]["lr"] == 1e-4 and RECIPE["logging"]["run_name"] == "base"      # the base recipe is untouched


def test_too_many_trials_and_bad_names_are_refused():
    with pytest.raises(ValueError, match="limit"):
        sweep.plan_trials({"name": "x", "params": {"training.lr": list(range(9)), "lora.r": list(range(9))}}, RECIPE)
    with pytest.raises(ValueError, match="sweep name"):
        sweep.plan_trials({"name": "bad name/", "params": {"lora.r": [8]}}, RECIPE)


def test_a_mistyped_parameter_is_caught_before_anything_is_submitted():
    ok = sweep.plan_trials({"name": "s", "params": {"training.lr": [1e-4]}}, RECIPE)
    assert sweep.check_options(ok) == []
    bad = sweep.plan_trials({"name": "s", "params": {"training.lrr": [1e-4]}}, RECIPE)
    problems = sweep.check_options(bad)
    assert problems and "training.lrr" in problems[0] and "did you mean 'lr'" in problems[0]


def _write(tmp_path, params, **extra):
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(RECIPE))
    spec = {"name": "demo", "recipe": "recipe.yaml", "nodes": 8, "hours": 4, "params": params, **extra}
    (tmp_path / "sweep.yaml").write_text(yaml.safe_dump(spec))
    return tmp_path / "sweep.yaml"


def test_plan_writes_one_recipe_per_trial(tmp_path, capsys):
    spec = _write(tmp_path, {"training.lr": [1e-4, 2e-4]})
    assert sweep.main(["plan", str(spec)]) == 0
    files = sorted((tmp_path / "sweeps" / "demo").glob("*.yaml"))
    assert [f.name for f in files] == ["demo-01.yaml", "demo-02.yaml"]
    assert yaml.safe_load(files[1].read_text())["training"]["lr"] == 2e-4
    assert "demo-02" in capsys.readouterr().out


def test_plan_stops_on_an_option_that_does_nothing(tmp_path, capsys):
    spec = _write(tmp_path, {"training.lrr": [1e-4]})
    assert sweep.main(["plan", str(spec)]) == 1
    assert "do nothing" in capsys.readouterr().out and not (tmp_path / "sweeps").exists()


def test_submit_queues_every_trial_and_reports_failures(tmp_path, monkeypatch, capsys, param_rudra):
    spec = _write(tmp_path, {"training.lr": [1e-4, 2e-4, 3e-4]})
    calls = []

    def fake_run(cmd, cwd, capture_output, text):
        calls.append((cmd, cwd))
        if "demo-02" in cmd[1]:
            return subprocess.CompletedProcess(cmd, 1, "preflight FAIL\n", "")
        return subprocess.CompletedProcess(cmd, 0, f"submitted job {100 + len(calls)}: x\n", "")

    from bhaskera.launcher import qos
    monkeypatch.setattr(qos, "limits_for", lambda part: qos.QosLimits(part, 24, 4, 8))
    monkeypatch.setattr(qos, "user_jobs", lambda part: (0, 0))
    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    assert sweep.main(["submit", str(spec)]) == 1                       # one trial failed
    assert [c[0][0] for c in calls] == ["./submit.sh"] * 3
    assert calls[0][0][2:] == ["8", "4"] and calls[0][1] == tmp_path
    out = capsys.readouterr().out
    assert "demo-01: submitted job 101" in out and "demo-02: NOT submitted" in out and "preflight FAIL" in out


def test_report_ranks_trials_by_best_eval_loss(tmp_path, monkeypatch, capsys):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = f"file://{tmp_path / 'store'}"
    client = mlflow.tracking.MlflowClient(uri, uri)
    exp = client.create_experiment("bhaskera")
    for name, losses, lr in (("demo-01", [2.0, 1.4], "0.0001"), ("demo-02", [2.0, 1.1], "0.0002"), ("other-01", [0.5], "1")):
        run = client.create_run(exp, tags={"group": "other" if name.startswith("other") else "demo", "mlflow.runName": name})
        client.log_param(run.info.run_id, "training.lr", lr)
        for step, v in enumerate(losses):
            client.log_metric(run.info.run_id, "eval_loss", v, step=step)
        client.set_terminated(run.info.run_id)
    assert sweep.main(["report", "demo", "--store", uri]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split()[:3] == ["run", "status", "best_eval_loss"]
    assert lines[1].startswith("demo-02") and "1.1000" in lines[1] and lines[2].startswith("demo-01")
    assert len(lines) == 3                                                # the other sweep is not mixed in
    assert sweep.main(["report", "nothing", "--store", uri]) == 1


def test_submit_queues_only_what_the_qos_allows_and_continues_later(tmp_path, monkeypatch, capsys, param_rudra):
    from bhaskera.launcher import qos
    spec = _write(tmp_path, {"training.lr": [1e-4, 2e-4, 3e-4]})
    queued = []

    def fake_run(cmd, cwd, capture_output, text):
        queued.append(cmd[1].rsplit("/", 1)[-1])
        return subprocess.CompletedProcess(cmd, 0, f"submitted job {200 + len(queued)}: x\n", "")

    monkeypatch.setattr(qos, "limits_for", lambda part: qos.QosLimits(part, 24, 4, 8))
    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    monkeypatch.setattr(qos, "user_jobs", lambda part: (4, 6))               # 6 of the 8 allowed jobs are used
    assert sweep.main(["submit", str(spec)]) == 0
    out = capsys.readouterr().out
    assert queued == ["demo-01.yaml", "demo-02.yaml"] and "room for 2 more" in out and "1 trial(s) not queued yet" in out
    monkeypatch.setattr(qos, "user_jobs", lambda part: (0, 0))               # later: the queue is empty
    assert sweep.main(["submit", str(spec)]) == 0
    assert queued == ["demo-01.yaml", "demo-02.yaml", "demo-03.yaml"]        # only the missing trial is queued
    assert "(2 were queued before)" in capsys.readouterr().out
