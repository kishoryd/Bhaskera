"""Resuming a run that stopped: what is read from the run and from Slurm, and what is refused."""
from __future__ import annotations

import subprocess

import pytest

from bhaskera.launcher import resume

RUN = "d" * 32


def _store(tmp_path, status="4", job="777", steps=(100, 200), incomplete=(300,), recipe=True, save_rel=False):
    work = tmp_path / "work"
    (work / "recipes").mkdir(parents=True)
    if recipe:
        (work / "recipes" / "my-run.yaml").write_text("model: {}\n")
    ckpt = work / "ckpt"
    for s in steps:
        (ckpt / f"step_{s:07d}").mkdir(parents=True)
        (ckpt / f"step_{s:07d}" / ".complete").touch()
    for s in incomplete:
        (ckpt / f"step_{s:07d}").mkdir(parents=True)
    run = tmp_path / "mlflow" / "1" / RUN
    (run / "tags").mkdir(parents=True)
    (run / "params").mkdir()
    (run / "meta.yaml").write_text(f"run_name: my-run\nstatus: {status}\n")
    if job:
        (run / "tags" / "slurm_job").write_text(job)
    (run / "params" / "checkpoint.save_dir").write_text("ckpt" if save_rel else str(ckpt))
    (run / "params" / "training.max_steps").write_text("400")
    return tmp_path / "mlflow", work


def _sacct(work, state="FAILED", line=None, nodes="4", limit="12:00:00"):
    line = line or "sbatch --parsable -A acct --partition=gpu-small --nodes=4 --time=12:00:00 train.sbatch recipes/my-run.yaml"

    def runner(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, f"{line}|{nodes}|{limit}|{work}|{state}\n", "")
    return runner


def test_a_failed_run_can_be_resumed_with_its_own_recipe_nodes_and_time(tmp_path):
    store, work = _store(tmp_path)
    info = resume.describe(store, RUN, _sacct(work))
    assert info["resumable"] and info["steps"] == [100, 200] and info["latest"] == 200, "an unfinished checkpoint is not counted"
    assert info["recipe"] == "my-run.yaml" and info["recipe_path"] == str((work / "recipes" / "my-run.yaml").resolve())
    assert (info["nodes"], info["hours"]) == (4, 12) and info["name"] == "my-run" and info["status"] == "FAILED"
    todo = resume.plan(info)
    assert todo == {"recipe": info["recipe_path"], "nodes": 4, "hours": 12, "step": 0, "from_step": 200, "run": RUN}


def test_nodes_hours_and_step_can_be_changed(tmp_path):
    store, work = _store(tmp_path)
    info = resume.describe(store, RUN, _sacct(work))
    todo = resume.plan(info, step=100, nodes=8, hours=6)
    assert (todo["nodes"], todo["hours"], todo["step"], todo["from_step"]) == (8, 6, 100, 100)
    with pytest.raises(resume.ResumeError, match="steps 100, 200"):
        resume.plan(info, step=150)


def test_a_relative_checkpoint_folder_is_found_from_the_job_folder(tmp_path):
    store, work = _store(tmp_path, save_rel=True)
    assert resume.describe(store, RUN, _sacct(work))["steps"] == [100, 200]


@pytest.mark.parametrize("kwargs,state,word", [
    ({"steps": (), "incomplete": ()}, "FAILED", "no checkpoint was saved before"),
    ({"recipe": False}, "FAILED", "not there any more"),
    ({"job": ""}, "FAILED", "does not record"),
    ({}, "RUNNING", "still running"),
])
def test_what_cannot_be_resumed_says_why(tmp_path, kwargs, state, word):
    store, work = _store(tmp_path, **kwargs)
    info = resume.describe(store, RUN, _sacct(work, state=state))
    assert not info["resumable"] and word in info["why"]
    with pytest.raises(resume.ResumeError, match=word):
        resume.plan(info)


def test_a_job_slurm_has_forgotten_is_explained(tmp_path):
    store, work = _store(tmp_path)
    with pytest.raises(resume.ResumeError, match="no longer knows"):
        resume.describe(store, RUN, lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))


def test_unknown_run(tmp_path):
    with pytest.raises(resume.ResumeError, match="no run"):
        resume.run_record(tmp_path, RUN)


@pytest.mark.parametrize("text,hours", [("1:00:00", 1), ("12:00:00", 12), ("1-00:00:00", 24), ("00:30:00", 1), ("90", 2),
                                         ("30:00", 1), ("2-12:00:00", 60)])
def test_slurm_times(text, hours):
    assert resume.parse_time(text) == hours


def test_command_line(tmp_path, capsys, monkeypatch):
    store, work = _store(tmp_path)
    monkeypatch.setattr(resume.subprocess, "run", _sacct(work))
    assert resume.main(["show", "--run", RUN, "--store", str(store)]) == 0
    assert "can be resumed" in capsys.readouterr().out
    assert resume.main(["plan", "--run", RUN, "--store", str(store), "--step", "100", "--hours", "6"]) == 0
    out = capsys.readouterr()
    line = out.out.strip().split()
    assert line[1:] == ["4", "6", "100", RUN] and "step 100" in out.err
    assert resume.main(["plan", "--run", RUN, "--store", str(store), "--step", "5"]) == 1


def test_a_relative_checkpoint_folder_must_be_found_from_the_same_place(tmp_path):
    store, work = _store(tmp_path)                                    # the run saved to <work>/ckpt
    (work / "recipes" / "my-run.yaml").write_text("checkpoint:\n  save_dir: ckpt\n")
    info = resume.describe(store, RUN, _sacct(work))
    assert resume.plan(info, submit_dir=str(work))["nodes"] == 4      # from the job's own folder: fine
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(resume.ResumeError, match="relative"):
        resume.plan(info, submit_dir=str(elsewhere))                  # from another folder it would start from step 0
    (work / "recipes" / "my-run.yaml").write_text(f"checkpoint:\n  save_dir: {work / 'ckpt'}\n")
    info = resume.describe(store, RUN, _sacct(work))
    assert resume.plan(info, submit_dir=str(elsewhere))["nodes"] == 4  # an absolute folder is found from anywhere


def test_checkpointing_switched_off_is_named_as_the_reason(tmp_path):
    store, work = _store(tmp_path, steps=(), incomplete=())
    (work / "recipes" / "my-run.yaml").write_text("checkpoint:\n  enabled: false\n")
    info = resume.describe(store, RUN, _sacct(work))
    assert not info["resumable"] and "checkpoint.enabled: false" in info["why"]
