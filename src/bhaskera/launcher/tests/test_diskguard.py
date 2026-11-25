"""Disk safety: every action that writes a lot checks the room first, and says so plainly instead of failing half way."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from bhaskera import diskguard as dg
from bhaskera.launcher import cluster_profile, hub_state, launch, recipes
from bhaskera.launcher.tests.test_recipes import form, world  # noqa: F401  (a model folder, a cache folder and a run folder)

GB = 2 ** 30


def disk(monkeypatch, free_gb):
    monkeypatch.setattr("bhaskera.diskguard.shutil.disk_usage", lambda p: NS(free=int(free_gb * GB), total=int(500 * GB), used=0))


def test_require_refuses_a_disk_without_room_and_says_where_and_how_much(monkeypatch, tmp_path):
    disk(monkeypatch, 10)
    assert dg.require(tmp_path / "x" / "y", 5 * GB, "the checkpoints") == 10 * GB          # a folder that does not exist yet: its disk is checked
    with pytest.raises(dg.DiskError) as e:
        dg.require(tmp_path / "x", 12 * GB, "the checkpoints")
    msg = str(e.value)
    assert "not enough space for the checkpoints" in msg and str(tmp_path / "x") in msg and "12.0 GB is needed and 10.0 GB is free" in msg and "quota" in msg
    with pytest.raises(dg.DiskError):
        dg.require(tmp_path, 10 * GB, "x")                  # exactly what is free is not enough: there is a margin
    assert dg.require(tmp_path, 0, "nothing") == 10 * GB    # nothing to write, nothing to refuse


def test_a_disk_that_cannot_be_read_never_blocks(monkeypatch, tmp_path):
    def boom(p):
        raise OSError("no such device")
    monkeypatch.setattr("bhaskera.diskguard.shutil.disk_usage", boom)
    assert dg.free_bytes(tmp_path) == -1 and dg.require(tmp_path, 99 * GB, "x") == -1 and dg.floor(tmp_path) == -1


def test_the_floor_stops_a_nearly_full_disk(monkeypatch, tmp_path):
    disk(monkeypatch, 0.4)
    with pytest.raises(dg.DiskError, match="only 0.4 GB free"):
        dg.floor(tmp_path, "the logs")
    disk(monkeypatch, 3)
    assert dg.floor(tmp_path) == 3 * GB


def test_a_checkpoint_is_about_the_size_of_the_weights(tmp_path):
    m = tmp_path / "m"
    m.mkdir()
    (m / "a.safetensors").write_bytes(b"x" * 1000)
    (m / "b.bin").write_bytes(b"x" * 500)
    (m / "notes.txt").write_bytes(b"x" * 9999)
    assert dg.model_bytes(m) == 1500 and dg.model_bytes(tmp_path / "none") == 0
    assert dg.checkpoint_need(m, 3) == (1500, 4500) and dg.checkpoint_need(m, 0) == (1500, 4500)       # keep-all is counted as 3


def test_the_note_for_a_preview_says_whether_there_is_room(monkeypatch, tmp_path):
    disk(monkeypatch, 100)
    assert "enough room" in dg.note(tmp_path, 10 * GB, 30 * GB, 3)
    disk(monkeypatch, 20)
    assert "room for one, not for all of them" in dg.note(tmp_path, 10 * GB, 30 * GB, 3)
    disk(monkeypatch, 5)
    assert "NOT ENOUGH ROOM" in dg.note(tmp_path, 10 * GB, 30 * GB, 3)
    assert dg.note(tmp_path, 0, 0, 3) == ""


def test_the_run_folder_status_flags_a_low_disk(monkeypatch, tmp_path):
    disk(monkeypatch, 8)
    assert dg.run_folder_status(tmp_path) == {"path": str(tmp_path), "free_gb": 8.0, "total_gb": 500.0, "low": True}
    disk(monkeypatch, 80)
    assert dg.run_folder_status(tmp_path)["low"] is False and dg.run_folder_status(None) is None


# ---- a new recipe: where the checkpoints go, and room for them
def heavy_model(model):
    (model / "w.safetensors").write_bytes(b"x" * 1024)
    return model


def test_a_recipe_saves_checkpoints_in_the_folder_you_choose(world, tmp_path):
    work, model, cache = world
    big = tmp_path / "bigdisk" / "ckpts"
    r = recipes.build(form(model, cache, checkpoint_dir=str(big)), work)
    assert r["checkpoint"]["save_dir"] == str((big / "my-run").resolve())          # the recipe's name is added, so recipes never share a folder
    assert recipes.build(form(model, cache, name="other"), work)["checkpoint"]["save_dir"] == str((work / "checkpoints" / "other").resolve())
    for bad in ("relative/dir", "/etc/ckpt", "/proc/x"):
        with pytest.raises(recipes.RecipeError, match="(full path|system folder)"):
            recipes.build(form(model, cache, name="bad", checkpoint_dir=bad), work)


def test_a_recipe_is_refused_when_the_checkpoints_would_not_fit(world, monkeypatch):
    work, model, cache = world
    heavy_model(model)
    monkeypatch.setattr("bhaskera.diskguard.model_bytes", lambda f: 60 * GB)
    disk(monkeypatch, 30)
    with pytest.raises(recipes.RecipeError, match="not enough space for the checkpoints.*60.0 GB is needed and 30.0 GB is free"):
        recipes.build(form(model, cache), work)
    r = recipes.build(form(model, cache, checkpoints=False, name="nockpt"), work)        # with checkpoints off there is nothing to fit
    assert r["checkpoint"]["enabled"] is False


# ---- starting a job
@pytest.fixture
def launcher(world, tmp_path, monkeypatch):
    work, model, cache = world
    monkeypatch.setattr(cluster_profile, "current", lambda: NS(gpus_per_node=4, partitions=[], gpu_mem_gb=None))
    (work / "recipes").mkdir(exist_ok=True)
    (work / "submit.sh").write_text("")
    calls = []
    L = launch.Launcher(work, runs=lambda: [{"run_id": "a" * 32, "name": "r", "status": "FINISHED", "has_adapter": True}], env={"BHASKERA_DIR": "/x"}, run=lambda *a, **k: calls.append(a) or NS(returncode=0, stdout="Submitted batch job 7", stderr=""))
    return work, model, cache, L, calls


def write_recipe(work, model, save_dir, **ck):
    p = work / "recipes" / "r.yaml"
    p.write_text(yaml.safe_dump({"model": {"name": str(model)}, "training": {"distributed": {"strategy": "ddp"}},
                                 "checkpoint": {"enabled": True, "save_dir": str(save_dir), "keep_last_n": 2, **ck}}))
    return p


def test_training_is_not_started_when_its_checkpoints_would_not_fit(launcher, monkeypatch, tmp_path):
    work, model, cache, L, calls = launcher
    write_recipe(work, model, tmp_path / "ck")
    monkeypatch.setattr("bhaskera.diskguard.model_bytes", lambda f: 60 * GB)
    disk(monkeypatch, 30)
    with pytest.raises(launch.LaunchError, match="not enough space for the checkpoints of this run.*checkpoint: save_dir"):
        L.build({"kind": "train", "recipe": "r.yaml", "nodes": 1})
    disk(monkeypatch, 500)
    assert L.build({"kind": "train", "recipe": "r.yaml", "nodes": 1})[0][0].endswith("r.yaml")
    write_recipe(work, model, tmp_path / "ck", enabled=False)
    disk(monkeypatch, 30)
    L.build({"kind": "train", "recipe": "r.yaml", "nodes": 1})                       # checkpoints off: nothing to fit


def test_no_job_is_submitted_on_a_practically_full_disk(launcher, monkeypatch):
    work, model, cache, L, calls = launcher
    disk(monkeypatch, 0.2)
    for form_ in ({"kind": "serve", "run": "a" * 32}, {"kind": "eval", "run": "a" * 32}):
        with pytest.raises(launch.LaunchError, match="only 0.2 GB free"):
            L.launch({"action": "start", **form_})
    assert calls == []
    with pytest.raises(launch.LaunchError, match="only 0.2 GB free"):
        L.launch({"action": "start", "kind": "tokenize", "train": "/x", "name": "d", "model": "/m"})


def test_preparing_data_needs_room_for_the_result(launcher, monkeypatch, tmp_path):
    work, model, cache, L, calls = launcher
    raw = tmp_path / "train.jsonl"
    raw.write_bytes(b"x" * 1000)
    monkeypatch.setattr("bhaskera.diskguard.free_bytes", lambda p: 1500)           # past the floor's check by mocking only the room
    monkeypatch.setattr(dg, "FLOOR", 0)
    with pytest.raises(launch.LaunchError, match="not enough space for the prepared data"):
        L._tokenize({"train": str(raw), "name": "d", "model": str(model)}, dry=False)


def test_a_preview_of_a_recipe_says_how_much_room_the_checkpoints_have(launcher, monkeypatch):
    work, model, cache, L, calls = launcher
    monkeypatch.setattr("bhaskera.diskguard.model_bytes", lambda f: 10 * GB)
    disk(monkeypatch, 100)
    out = L.launch({"kind": "recipe", "action": "check", **form(model, cache)})
    assert out["dry"] and out["output"].startswith("# Disk: 100.0 GB free where the checkpoints go") and "enough room" in out["output"].splitlines()[0]
    yaml.safe_load(out["output"])                                                   # the note is a comment: still a valid recipe


def test_the_page_state_carries_the_disk_of_the_run_folder(monkeypatch, tmp_path):
    disk(monkeypatch, 8)
    (tmp_path / "submit.sh").write_text("")
    monkeypatch.setenv("BHASKERA_WORKDIR", str(tmp_path))
    b = hub_state.StateBuilder(NS(), str(tmp_path / "store"), "u")
    assert b._cached("disk", 30.0, lambda: dg.run_folder_status(__import__("bhaskera.paths", fromlist=["x"]).run_folder()))["low"] is True
