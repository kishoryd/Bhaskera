"""Launcher behaviour added for the page: GPU-count checks, benchmark options, your own tests, registered model references."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace as NS

import pytest
import yaml

from bhaskera.launcher import cluster_profile, launch

RUN = "c" * 32


class Fake:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, "wrote done" if "init-task" in argv else "submitted job 9: x", "")


@pytest.fixture
def world(tmp_path, monkeypatch):
    (tmp_path / "recipes").mkdir()
    (tmp_path / "submit.sh").write_text("")
    monkeypatch.setattr(cluster_profile, "current", lambda: NS(gpus_per_node=4, partitions=[]))
    fake = Fake()
    L = launch.Launcher(tmp_path, runs=lambda: [{"run_id": RUN, "name": "r", "status": "FINISHED", "has_adapter": True}],
                        env={"BHASKERA_DIR": "/x", "SLURM_ACCOUNT": "a"}, run=fake)
    return tmp_path, L, fake


def recipe(path, name, training):
    (path / "recipes" / f"{name}.yaml").write_text(yaml.safe_dump({"model": {"name": "/m/X"}, "training": {"distributed": training}}))


def test_hsdp_and_pipeline_must_divide_the_gpus(world):
    work, L, _ = world
    recipe(work, "hs", {"strategy": "fsdp", "fsdp": {"sharding_strategy": "HYBRID_SHARD", "hsdp_shard_size": 8}})
    recipe(work, "pp", {"strategy": "pipeline", "pipeline": {"stages": 3}})
    recipe(work, "ok", {"strategy": "pipeline", "pipeline": {"stages": 4}})
    with pytest.raises(launch.LaunchError, match="does not divide the 12 GPUs of 3 nodes"):
        L.build({"kind": "train", "recipe": "hs.yaml", "nodes": 3})
    with pytest.raises(launch.LaunchError, match="3 stages, which does not divide the 8 GPUs"):
        L.build({"kind": "train", "recipe": "pp.yaml", "nodes": 2})
    assert L.build({"kind": "train", "recipe": "ok.yaml", "nodes": 2})[0] == [str(work / "recipes" / "ok.yaml"), "2"]
    assert L.build({"kind": "train", "recipe": "hs.yaml", "nodes": 2})[0][1] == "2"          # 8 GPUs, groups of 8


def test_benchmark_options_reach_the_command(world):
    work, L, _ = world
    args, extra = L.build({"kind": "eval", "run": RUN, "tasks": "arc_easy", "num_fewshot": "5", "skip_base": True})
    assert args[:3] == ["eval", "--run", RUN] and ["--num-fewshot", "5"] == args[args.index("--num-fewshot"):][:2] and "--skip-base" in args
    with pytest.raises(launch.LaunchError, match="between 0 and 50"):
        L.build({"kind": "eval", "run": RUN, "num_fewshot": "80"})


def test_own_tests_are_added_listed_and_handed_to_the_job(world):
    work, L, fake = world
    data = work / "q.jsonl"
    data.write_text('{"question": "a", "answer": "b"}\n')
    r = L.launch({"kind": "evaltask", "action": "start", "name": "my_test", "data": str(data), "question_field": "question", "answer_field": "answer"})
    assert r["ok"] and fake.calls[-1][0][1:4] == ["-m", "bhaskera.launcher.evaluate", "init-task"] and str(work / "tasks") in fake.calls[-1][0]
    (work / "tasks").mkdir()
    (work / "tasks" / "my_test.yaml").write_text("task: my_test\n")
    assert L.options()["own_tests"] == ["my_test"]
    with pytest.raises(launch.LaunchError, match="exists already"):
        L.launch({"kind": "evaltask", "action": "start", "name": "my_test", "data": str(data)})
    for bad, word in (({"name": "1x"}, "test name"), ({"name": "ok", "data": "relative.jsonl"}, "full path"), ({"name": "ok", "data": str(data), "answer_field": "a b"}, "field name")):
        with pytest.raises(launch.LaunchError, match=word):
            L.launch({"kind": "evaltask", "action": "start", **{"data": str(data), **bad}})
    assert L.build({"kind": "eval", "run": RUN})[1]["BHASKERA_TASKS_DIR"] == str((work / "tasks").resolve())


def test_registered_versions_are_offered_with_their_run_and_note(world, monkeypatch):
    work, L, _ = world
    from bhaskera.launcher import register
    monkeypatch.setattr(register, "make_client", lambda store: object())
    monkeypatch.setattr(register, "listing", lambda c: [{"name": "m", "versions": [{"version": "2", "run_id": RUN, "aliases": ["live"],
                                                                                    "tags": {"run_name": "r", "eval_loss": "1.0"}, "description": "note text"}]}])
    ref = L.options()["models"][0]["refs"][0]
    assert ref["ref"] == "m:2" and ref["run_id"] == RUN and ref["aliases"] == ["live"] and ref["note"] == "note text" and ref["eval_loss"] == "1.0"
