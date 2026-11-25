"""Dataset versions: the same content keeps its version, changed content becomes the next number, and changes are noticed."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from bhaskera import dataversions as dv


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    run = tmp_path / "runs"
    run.mkdir()
    (run / "submit.sh").write_text("")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for var in ("MLFLOW_TRACKING_URI", "HF_HOME", "HF_DATASETS_CACHE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BHASKERA_WORKDIR", str(run))
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"q": "a", "a": "b"}\n{"q": "c", "a": "d"}\n')
    return run, data


def test_the_same_content_is_the_same_version_and_changed_content_is_the_next(world):
    run, data = world
    a = dv.add("support", str(data), "first")
    assert a["version"] == 1 and a["new"] is True and a["rows"] == 2 and a["kind"] == "raw"
    again = dv.add("support", str(data))
    assert again["version"] == 1 and again["new"] is False
    (data / "train.jsonl").write_text('{"q": "a", "a": "b"}\n{"q": "c", "a": "d"}\n{"q": "e", "a": "f"}\n')
    b = dv.add("support", str(data), "with one more example")
    assert b["version"] == 2 and b["new"] is True and b["rows"] == 3 and b["hash"] != a["hash"]
    assert (run / ".bhaskera" / "datasets.json").is_file()


def test_changes_are_noticed_without_reading_the_files(world):
    run, data = world
    dv.add("support", str(data))
    assert dv.listing()[0]["versions"][0]["status"] == "ok"
    f = data / "train.jsonl"
    f.write_text(f.read_text() + '{"q": "x", "a": "y"}\n')
    assert dv.listing()[0]["versions"][0]["status"] == "changed"
    found = dv.for_path(str(data))
    assert found["name"] == "support" and found["version"] == 1 and found["status"] == "changed"
    assert dv.verify("support")["status"] == "changed"
    nxt = dv.add("support", str(data))
    assert nxt["version"] == 2
    assert dv.for_path(str(data))["version"] == 2 and dv.for_path(str(data))["status"] == "ok"


def test_a_folder_that_is_gone_is_missing_and_unknown_data_has_no_version(world, tmp_path):
    run, data = world
    dv.add("support", str(data))
    (data / "train.jsonl").unlink()
    data.rmdir()
    assert dv.listing()[0]["versions"][0]["status"] == "missing"
    assert dv.verify("support")["status"] == "missing"
    assert dv.for_path(str(tmp_path / "nowhere")) is None


def test_names_paths_and_versions_are_checked(world, tmp_path):
    run, data = world
    for bad in ("", "-x", "a b", "x" * 80, None):
        with pytest.raises(dv.DataError):
            dv.add(bad, str(data))
    with pytest.raises(dv.DataError, match="full path"):
        dv.add("ok", "relative/dir")
    with pytest.raises(dv.DataError, match="does not exist"):
        dv.add("ok", str(tmp_path / "none"))
    dv.add("ok", str(data))
    with pytest.raises(dv.DataError, match="no version 5"):
        dv.get("ok", 5)
    with pytest.raises(dv.DataError, match="no dataset"):
        dv.get("nope")


def test_a_runs_tags_name_the_version_of_its_data(world):
    run, data = world
    v = dv.add("support", str(data))
    cfg = NS(data=NS(tokenized_path=str(data), val_tokenized_path=None))
    assert dv.run_tags(cfg) == {"dataset.name": "support", "dataset.version": "1", "dataset.hash": v["hash"][:12], "dataset.status": "ok"}
    assert dv.run_tags(NS(data=NS(tokenized_path="/not/registered", val_tokenized_path=None))) == {}
    assert dv.run_tags(NS()) == {}


def test_forgetting_a_version_keeps_the_files(world):
    run, data = world
    dv.add("support", str(data))
    assert dv.remove("support", 1)["removed"] is True
    assert dv.listing() == [] and (data / "train.jsonl").is_file()


def test_the_fingerprint_depends_on_content_and_names(world, tmp_path):
    run, data = world
    h1 = dv.fingerprint(data)[0]
    other = tmp_path / "copy"
    other.mkdir()
    (other / "train.jsonl").write_text((data / "train.jsonl").read_text())
    assert dv.fingerprint(other)[0] == h1                      # the same files in another folder
    (other / "train.jsonl").rename(other / "renamed.jsonl")
    assert dv.fingerprint(other)[0] != h1


def test_the_command_line_registers_lists_and_verifies(world, capsys):
    run, data = world
    assert dv.main(["add", "--name", "support", "--path", str(data)]) == 0
    assert "support v1 registered" in capsys.readouterr().out
    assert dv.main(["list"]) == 0 and "support:1" in capsys.readouterr().out
    assert dv.main(["verify", "support"]) == 0
    os.utime(data / "train.jsonl", (1, 1))
    assert dv.main(["add", "--name", "support", "--path", str(data)]) == 0
    assert dv.main(["show", "nope"]) == 2
