"""Where a user's files go: the run folder, with an existing home-folder copy kept until a copy or a link exists in the run folder."""
from __future__ import annotations

from pathlib import Path

import pytest

from bhaskera import paths


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    run = tmp_path / "runs"
    run.mkdir()
    (run / "submit.sh").write_text("")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for var in ("BHASKERA_WORKDIR", "MLFLOW_TRACKING_URI", "HF_HOME", "HF_DATASETS_CACHE"):
        monkeypatch.delenv(var, raising=False)
    return home, run


def test_the_run_folder_is_the_workdir_or_the_current_folder_with_submit_sh(world, tmp_path):
    home, run = world
    assert paths.run_folder({"BHASKERA_WORKDIR": str(run)}, tmp_path) == run.resolve()
    assert paths.run_folder({}, run) == run.resolve()
    assert paths.run_folder({}, tmp_path) is None and paths.run_folder({"BHASKERA_WORKDIR": str(tmp_path)}, tmp_path) is None


def test_a_new_user_gets_everything_inside_the_run_folder(world):
    home, run = world
    assert paths.mlflow_store({}, run) == str(run.resolve() / "mlflow-runs")
    assert paths.mlflow_uri({}, run) == f"file://{run.resolve() / 'mlflow-runs'}"
    assert paths.state_dir({}, run) == run.resolve() / ".bhaskera"
    assert paths.hf_home({}, run) == run.resolve() / ".cache" / "huggingface"


def test_what_a_user_already_keeps_at_home_is_used_until_a_copy_or_link_exists_in_the_run_folder(world):
    home, run = world
    (home / "mlflow-runs").mkdir()
    (home / ".bhaskera").mkdir()
    (home / ".cache" / "huggingface").mkdir(parents=True)
    assert paths.mlflow_store({}, run) == str(home / "mlflow-runs") and paths.state_dir({}, run) == home / ".bhaskera"
    assert paths.hf_home({}, run) is None                                          # leave Hugging Face's own default: the data is there
    (run / "mlflow-runs").symlink_to(home / "mlflow-runs")                          # a link in the run folder brings it inside, with no move
    (run / ".bhaskera").mkdir()
    assert paths.mlflow_store({}, run) == str(run.resolve() / "mlflow-runs") and paths.state_dir({}, run) == run.resolve() / ".bhaskera"
    (run / ".cache" / "huggingface").mkdir(parents=True)
    assert paths.hf_home({}, run) == run.resolve() / ".cache" / "huggingface"


def test_without_a_run_folder_the_home_folder_is_used_as_before(world, tmp_path):
    home, run = world
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert paths.mlflow_store({}, elsewhere) == str(home / "mlflow-runs") and paths.state_dir({}, elsewhere) == home / ".bhaskera"
    assert paths.hf_home({}, elsewhere) is None


def test_the_tracking_uri_and_hugging_face_settings_always_win(world):
    home, run = world
    assert paths.mlflow_store({"MLFLOW_TRACKING_URI": "file:///data/store"}, run) == "/data/store"
    assert paths.mlflow_uri({"MLFLOW_TRACKING_URI": "sqlite:////data/m.db"}, run) == "sqlite:////data/m.db"
    assert paths.hf_home({"HF_HOME": "/somewhere"}, run) is None and paths.hf_home({"HF_DATASETS_CACHE": "/somewhere"}, run) is None


def test_the_commands_take_their_defaults_from_it(world, monkeypatch):
    home, run = world
    monkeypatch.chdir(run)
    from bhaskera.launcher import evaluate, register, resume
    for module in (register, resume, evaluate):
        fn = getattr(module, "_store_default", None) or getattr(module, "_store", None)
        assert fn is not None and fn() == str(run.resolve() / "mlflow-runs"), module.__name__


def test_without_a_run_folder_the_default_one_in_home_is_made_from_the_shared_kit(world, tmp_path):
    home, run = world
    (run / "submit.sh").unlink()
    kit = tmp_path / "sharedkit"
    (kit / "recipes").mkdir(parents=True)
    (kit / "submit.sh").write_text("#!/bin/bash\n")
    (kit / "recipes" / "a.yaml").write_text("x: 1\n")
    kit.chmod(0o555); (kit / "submit.sh").chmod(0o555)                       # a shared kit is read-only
    env = {"BHASKERA_KIT": str(kit)}
    assert paths.run_folder(env, tmp_path) is None                              # nothing yet
    made = paths.ensure_default(env)
    assert made == (home / "bhaskera-runs").resolve() and (made / "recipes" / "a.yaml").read_text() == "x: 1\n"
    assert os_access_write(made / "submit.sh") and os_access_write(made / "recipes")   # the copy belongs to the user
    assert paths.run_folder(env, tmp_path) == made                               # now it is the run folder
    assert paths.mlflow_store(env, tmp_path) == str(made / "mlflow-runs")
    (made / "recipes" / "a.yaml").write_text("mine: 2\n")
    assert paths.ensure_default(env) == made and (made / "recipes" / "a.yaml").read_text() == "mine: 2\n"      # never overwritten
    kit.chmod(0o755)


def os_access_write(p):
    import os
    return os.access(p, os.W_OK)


def test_a_folder_the_user_already_has_at_the_default_place_is_left_alone(world, tmp_path):
    home, run = world
    (run / "submit.sh").unlink()
    (home / "bhaskera-runs").mkdir()
    (home / "bhaskera-runs" / "notes.txt").write_text("keep")
    kit = tmp_path / "sharedkit"
    kit.mkdir()
    (kit / "submit.sh").write_text("")
    assert paths.ensure_default({"BHASKERA_KIT": str(kit)}) is None
    assert (home / "bhaskera-runs" / "notes.txt").read_text() == "keep" and not (home / "bhaskera-runs" / "submit.sh").exists()
    assert paths.default_folder({"BHASKERA_RUNS": "/x/y"}) == Path("/x/y")
