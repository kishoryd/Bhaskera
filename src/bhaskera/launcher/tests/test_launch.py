"""The Start buttons: what the form may ask for, and that nothing else reaches submit.sh."""
import json
import subprocess
from pathlib import Path

import pytest

from bhaskera.launcher import launch

RUN = "a" * 32
GOOD = "model:\n  name: /m/Qwen3-14B\n  quantization: qlora\nlora:\n  enabled: true\ndistributed:\n  strategy: ddp\n"


@pytest.fixture
def kit(tmp_path):
    (tmp_path / "recipes").mkdir()
    (tmp_path / "submit.sh").write_text("#!/bin/bash\n")
    (tmp_path / "recipes" / "ok.yaml").write_text(GOOD)
    (tmp_path / "recipes" / "template.yaml").write_text(GOOD.replace("/m/", "/scratch/<project>/<username>/"))
    (tmp_path / "recipes" / "tokenize-chatml.yaml").write_text("data:\n  name: local\n")
    return tmp_path


class Fake:
    def __init__(self, out="submitted job 42: x", code=0):
        self.calls, self.out, self.code = [], out, code

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, self.code, self.out, "")


def make(kit, fake=None, env=None, now=lambda: 1000.0):
    env = {"BHASKERA_DIR": "/x", "SLURM_ACCOUNT": "acct", **(env or {})}
    runs = lambda: [{"run_id": RUN, "name": "r", "status": "FINISHED", "has_adapter": True},
                    {"run_id": "b" * 32, "name": "n", "status": "FINISHED", "has_adapter": False}]
    return launch.Launcher(kit, runs=runs, env=env, run=fake or Fake(), clock=now)


def test_recipes_are_described_and_placeholders_blocked(kit):
    by = {r["file"]: r for r in launch.list_recipes(kit, {})}
    assert by["ok.yaml"]["ready"] and by["ok.yaml"]["method"] == "QLoRA" and by["ok.yaml"]["model"] == "Qwen3-14B"
    assert not by["template.yaml"]["ready"] and "placeholders" in by["template.yaml"]["why"]
    assert by["tokenize-chatml.yaml"]["tokenize"] and not by["tokenize-chatml.yaml"]["ready"]


def test_placeholders_in_comments_do_not_block(kit):
    (kit / "recipes" / "commented.yaml").write_text("# replace every <project>/<username> and REPLACE_ lines\n" + GOOD)
    assert {r["file"]: r for r in launch.list_recipes(kit, {})}["commented.yaml"]["ready"]


def test_train_arguments_and_clean_environment(kit):
    fake = Fake()
    out = make(kit, fake, {"SKIP_PREFLIGHT": "1", "SKIP_QOS_CHECK": "1"}).launch({"action": "start", "kind": "train", "recipe": "ok.yaml", "nodes": "2", "hours": "1"})
    argv, kw = fake.calls[0]
    assert argv == ["bash", str(kit / "submit.sh"), str(kit / "recipes" / "ok.yaml"), "2", "1"]
    assert "SKIP_PREFLIGHT" not in kw["env"] and "SKIP_QOS_CHECK" not in kw["env"] and "DRY_RUN" not in kw["env"]
    assert out["ok"] and out["job"] == "42"


def test_check_is_a_dry_run(kit):
    fake = Fake(out="dry run: every check passed; nothing was submitted")
    out = make(kit, fake).launch({"action": "check", "kind": "train", "recipe": "ok.yaml", "nodes": 1})
    assert fake.calls[0][1]["env"]["DRY_RUN"] == "1" and out["ok"] and out["dry"] and out["job"] is None


def test_serve_and_eval(kit):
    fake = Fake()
    m = make(kit, fake)
    m.launch({"action": "start", "kind": "serve", "run": RUN, "hours": 3, "four_bit": True})
    argv, kw = fake.calls[-1]
    assert argv[2:] == ["serve", "--run", RUN, "--4bit"] and kw["env"]["SERVE_HOURS"] == "3"
    m._last = 0
    m.launch({"action": "start", "kind": "eval", "run": RUN, "tasks": "arc_easy,gsm8k", "limit": "50"})
    argv, kw = fake.calls[-1]
    assert argv[2:] == ["eval", "--run", RUN, "--tasks", "arc_easy,gsm8k", "--limit", "50"] and kw["env"]["EVAL_HOURS"] == "2"
    m._last = 0
    m.launch({"action": "start", "kind": "eval", "run": RUN, "samples": True})
    assert "--save-samples" in fake.calls[-1][0]
    m._last = 0
    m.launch({"action": "start", "kind": "eval", "run": RUN, "samples": "yes"})
    assert "--save-samples" not in fake.calls[-1][0]


@pytest.mark.parametrize("form", [
    {"kind": "shell"}, {"kind": "train", "recipe": "../../etc/passwd", "nodes": 1}, {"kind": "train", "recipe": "ok.yaml;rm -rf", "nodes": 1},
    {"kind": "train", "recipe": "missing.yaml", "nodes": 1}, {"kind": "train", "recipe": "template.yaml", "nodes": 1},
    {"kind": "train", "recipe": "ok.yaml", "nodes": "2 --x"}, {"kind": "train", "recipe": "ok.yaml", "nodes": 0},
    {"kind": "train", "recipe": "ok.yaml", "nodes": 5000}, {"kind": "train", "recipe": "ok.yaml", "nodes": True},
    {"kind": "train", "recipe": "ok.yaml", "nodes": 1, "hours": -3},
    {"kind": "serve", "run": "nope"}, {"kind": "serve", "run": "b" * 32}, {"kind": "serve", "run": RUN + "; ls"},
    {"kind": "eval", "run": RUN, "tasks": "a b; rm"}, {"kind": "eval", "run": RUN, "limit": "1e9"},
])
def test_bad_forms_never_reach_the_script(kit, form):
    fake = Fake()
    with pytest.raises(launch.LaunchError):
        make(kit, fake).launch({"action": "start", **form})
    assert fake.calls == []


def test_disabled_without_account_or_kit_or_switch(kit):
    for env, word in (({"SLURM_ACCOUNT": ""}, "SLURM_ACCOUNT"), ({"BHASKERA_HUB_LAUNCH": "0"}, "turned off")):
        m = make(kit, env=env)
        assert word in m.options()["reason"] and not m.options()["enabled"]
        with pytest.raises(launch.LaunchError):
            m.launch({"action": "start", "kind": "train", "recipe": "ok.yaml", "nodes": 1})
    assert not launch.Launcher(None, runs=lambda: [], env={}).options()["enabled"]


def test_runaway_guard(kit):
    t = [1000.0]
    m = make(kit, now=lambda: t[0])
    form = {"action": "start", "kind": "train", "recipe": "ok.yaml", "nodes": 1}
    m.launch(form)
    with pytest.raises(launch.LaunchError, match="just now"):
        m.launch(form)
    for _ in range(launch.STARTS_PER_HOUR - 1):
        t[0] += 5
        m.launch(form)
    t[0] += 5
    with pytest.raises(launch.LaunchError, match="last hour"):
        m.launch(form)
    t[0] += 3600
    assert m.launch(form)["ok"]


def test_failure_output_is_returned(kit):
    out = make(kit, Fake(out="preflight reported FAIL", code=1)).launch({"action": "start", "kind": "train", "recipe": "ok.yaml", "nodes": 1})
    assert not out["ok"] and "FAIL" in out["output"]


def test_find_kit(kit, tmp_path_factory):
    assert launch.find_kit({"BHASKERA_WORKDIR": str(kit)}, tmp_path_factory.mktemp("c")) == kit.resolve()
    assert launch.find_kit({}, tmp_path_factory.mktemp("d")) is None
    assert launch.find_kit({"BHASKERA_KIT": str(kit)}, tmp_path_factory.mktemp("e")) == kit.resolve()


STOPPED = "e" * 32


def make_with_stopped(kit, fake=None, describe=None):
    runs = lambda: [{"run_id": STOPPED, "name": "crashed", "status": "FAILED", "job": "55"},
                    {"run_id": "f" * 32, "name": "slow", "status": "RUNNING", "stale": True, "job": "56"},
                    {"run_id": "1" * 32, "name": "fine", "status": "FINISHED", "job": "57"},
                    {"run_id": "2" * 32, "name": "unknown-job", "status": "FAILED", "job": ""}]
    return launch.Launcher(kit, runs=runs, env={"BHASKERA_DIR": "/x", "SLURM_ACCOUNT": "a"}, run=fake or Fake(), describe=describe)


def test_stopped_runs_are_listed_and_resumed(kit):
    fake = Fake()
    m = make_with_stopped(kit, fake)
    assert [(r["name"], r["status"]) for r in m.options()["stopped"]] == [("crashed", "FAILED"), ("slow", "STALLED")]
    m.launch({"action": "start", "kind": "resume", "run": STOPPED})
    assert fake.calls[-1][0][2:] == ["resume", "--run", STOPPED]
    m._last = 0
    m.launch({"action": "start", "kind": "resume", "run": STOPPED, "step": "200", "nodes": 8, "hours": "6"})
    assert fake.calls[-1][0][2:] == ["resume", "--run", STOPPED, "--step", "200", "--nodes", "8", "--hours", "6"]


@pytest.mark.parametrize("form", [
    {"kind": "resume", "run": "1" * 32}, {"kind": "resume", "run": "nope"}, {"kind": "resume"}, {"kind": "resume", "run": STOPPED, "step": "-1"},
    {"kind": "resume", "run": STOPPED, "nodes": "9999"}, {"kind": "resume", "run": STOPPED + "; ls"},
])
def test_only_stopped_runs_can_be_resumed(kit, form):
    fake = Fake()
    with pytest.raises(launch.LaunchError):
        make_with_stopped(kit, fake).launch({"action": "start", **form})
    assert fake.calls == []


def test_resume_info_reports_what_the_run_left(kit):
    info = {"run_id": STOPPED, "name": "crashed", "status": "FAILED", "recipe": "r.yaml", "nodes": 4, "hours": 12, "steps": [100, 200],
            "latest": 200, "max_steps": "400", "resumable": True, "why": "", "recipe_path": "/secret/path", "save_dir": "/secret/ckpt"}
    out = make_with_stopped(kit, describe=lambda store, run: info).resume_info(STOPPED)
    assert out["steps"] == [100, 200] and out["resumable"] and "recipe_path" not in out and "save_dir" not in out, "paths stay on the server"
    with pytest.raises(launch.LaunchError):
        make_with_stopped(kit, describe=lambda s, r: info).resume_info("1" * 32)
    from bhaskera.launcher import resume

    def broken(store, run):
        raise resume.ResumeError("Slurm no longer knows job 55")
    out = make_with_stopped(kit, describe=broken).resume_info(STOPPED)
    assert not out["resumable"] and "no longer knows" in out["why"]
