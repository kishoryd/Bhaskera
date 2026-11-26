"""A recipe that names a Hugging Face model or dataset: it is downloaded before the job, and the job gets a resolved copy of the recipe."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from bhaskera import downloads as dl
from bhaskera.launcher import fetch


@pytest.fixture
def run(tmp_path, monkeypatch):
    (tmp_path / "submit.sh").write_text("")
    monkeypatch.setenv("BHASKERA_WORKDIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def recipe(run, **parts):
    p = run / "my.yaml"
    p.write_text(yaml.safe_dump(parts))
    return str(p)


def test_a_name_is_a_repository_only_when_it_is_owner_slash_name_and_not_a_path(run):
    (run / "local" / "dir").mkdir(parents=True)
    for yes in ("Qwen/Qwen3-1.7B", "owner/data-set_v2", "a/b"):
        assert fetch.looks_like_repo(yes), yes
    for no in ("/abs/path/m", "./rel/m", "~/m", "gpt2", "a/b/c", "local/dir", "", None, "hf://a/b", "a b/c", 5):
        assert not fetch.looks_like_repo(no), no


def test_a_reference_has_an_optional_revision():
    assert fetch.parse_ref("hf://owner/name") == ("owner/name", "main")
    assert fetch.parse_ref("hf://owner/name@v2") == ("owner/name", "v2")
    assert fetch.parse_ref("owner/name@abc123") == ("owner/name", "abc123")


def make_files(dest, names):
    for n in names:
        f = dest / n
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")


def test_downloaded_files_are_sorted_into_train_and_validation_by_their_names(tmp_path):
    make_files(tmp_path, ["data/train-00000.parquet", "data/train-00001.parquet", "data/validation-00000.parquet", "data/test-00000.parquet",
                          "README.md", ".gitattributes"])
    train, val = fetch.make_splits(tmp_path)
    assert sorted(f.name[5:] for f in train.iterdir()) == ["train-00000.parquet", "train-00001.parquet"]
    assert [f.name[5:] for f in val.iterdir()] == ["validation-00000.parquet"]                    # validation is preferred to test
    assert all(f.is_symlink() for f in train.iterdir())


def test_test_is_used_when_there_is_no_validation_part_and_a_single_file_is_all_training(tmp_path):
    make_files(tmp_path, ["train.jsonl", "test.jsonl"])
    train, val = fetch.make_splits(tmp_path)
    assert [f.name[5:] for f in val.iterdir()] == ["test.jsonl"]
    only = tmp_path / "only"
    make_files(only, ["alpaca_data.json"])
    train, val = fetch.make_splits(only)
    assert [f.name[5:] for f in train.iterdir()] == ["alpaca_data.json"] and val is None


def test_folders_named_after_the_split_count_and_every_supported_format_is_found(tmp_path):
    make_files(tmp_path, ["train/a.csv", "train/b.csv.gz", "val/a.csv", "x/readme.txt"])
    train, val = fetch.make_splits(tmp_path)
    assert len(list(train.iterdir())) == 2 and len(list(val.iterdir())) == 1
    with pytest.raises(dl.DownloadError, match="no data files"):
        fetch.make_splits(tmp_path / "x")


def test_splitting_again_replaces_the_links(tmp_path):
    make_files(tmp_path, ["train.csv", "val.csv"])
    fetch.make_splits(tmp_path)
    (tmp_path / "train.csv").unlink()
    make_files(tmp_path, ["train2.csv"])
    train, _ = fetch.make_splits(tmp_path)
    assert [f.name[5:] for f in train.iterdir()] == ["train2.csv"]


def test_a_recipe_without_names_to_download_is_used_as_it_is(run):
    p = recipe(run, model={"name": "/models/Qwen3-14B"}, data={"name": "local", "tokenized_path": "/cache/x"})
    assert fetch.resolve(p, fetch=lambda *a: pytest.fail("nothing to fetch")) == (p, [])


def fake_fetch(run):
    calls = []

    def f(kind, repo, rev, say=print):
        d = run / ("models" if kind == "model" else "data/hf") / repo.replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        if kind == "model":
            (d / "config.json").write_text("{}")
        else:
            make_files(d, ["train-0.parquet", "validation-0.parquet"])
        calls.append((kind, repo, rev))
        return d
    return f, calls


def test_a_model_and_a_dataset_named_in_the_recipe_are_replaced_by_their_folders(run):
    p = recipe(run, model={"name": "Qwen/Qwen3-1.7B", "dtype": "bfloat16"}, data={"name": "local", "train_path": "hf://owner/chat@v1", "format": "chatml"})
    f, calls = fake_fetch(run)
    new, changes = fetch.resolve(p, say=lambda m: None, fetch=f)
    assert calls == [("model", "Qwen/Qwen3-1.7B", "main"), ("dataset", "owner/chat", "v1")]
    r = yaml.safe_load(Path(new).read_text())
    assert new == str(run / ".bhaskera" / "resolved" / "my.yaml")
    assert r["model"] == {"name": str(run / "models" / "Qwen__Qwen3-1.7B"), "dtype": "bfloat16"}
    assert r["data"]["train_path"].endswith("owner__chat/_splits/train") and r["data"]["val_path"].endswith("owner__chat/_splits/val")
    assert r["data"]["format"] == "chatml" and len(changes) == 3
    assert yaml.safe_load(Path(p).read_text())["model"]["name"] == "Qwen/Qwen3-1.7B"                 # the recipe itself is untouched


def test_a_validation_path_the_recipe_names_is_kept(run):
    p = recipe(run, model={"name": "/m"}, data={"name": "local", "train_path": "hf://owner/chat", "val_path": "/my/own/val.jsonl"})
    f, _ = fake_fetch(run)
    new, _ = fetch.resolve(p, say=lambda m: None, fetch=f)
    assert yaml.safe_load(Path(new).read_text())["data"]["val_path"] == "/my/own/val.jsonl"


def test_a_dataset_given_as_the_data_name_becomes_the_local_dataset(run):
    p = recipe(run, model={"name": "/m"}, data={"name": "owner/chat", "format": "alpaca"})
    f, calls = fake_fetch(run)
    new, _ = fetch.resolve(p, say=lambda m: None, fetch=f)
    d = yaml.safe_load(Path(new).read_text())["data"]
    assert calls == [("dataset", "owner/chat", "main")] and d["name"] == "local" and d["train_path"].endswith("_splits/train") and d["format"] == "alpaca"


def test_a_plan_only_run_downloads_nothing(run):
    p = recipe(run, model={"name": "Qwen/Qwen3-1.7B"}, data={"name": "local", "train_path": "hf://owner/chat"})
    said = []
    out, changes = fetch.resolve(p, say=said.append, fetch=lambda *a: pytest.fail("a plan must not download"), plan_only=True)
    assert out == p and changes == ["plan"] and any("would download model Qwen/Qwen3-1.7B" in m for m in said)


def test_a_download_that_finished_is_not_repeated_and_the_marker_is_checked(run, monkeypatch):
    dest = run / "models" / "Qwen__Qwen3-1.7B"
    dest.mkdir(parents=True)
    (dest / "config.json").write_text("{}")
    (dest / fetch.MARKER).write_text(json.dumps({"repo": "Qwen/Qwen3-1.7B", "revision": "main"}))
    monkeypatch.setattr(dl, "preview", lambda *a, **k: pytest.fail("no network for a finished download"))
    assert fetch.fetch_repo("model", "Qwen/Qwen3-1.7B", "main", say=lambda m: None) == dest
    (dest / fetch.MARKER).write_text(json.dumps({"repo": "Other/Model", "revision": "main"}))
    assert not fetch._fetched(dest, "Qwen/Qwen3-1.7B", "main")


def test_fetching_checks_the_room_and_runs_the_download_that_routes(run, monkeypatch):
    monkeypatch.setattr(dl, "preview", lambda kind, repo, rev, **k: {"bytes": 5 << 30, "files": 3, "sha": "abc", "via": "login01", "gated": True})
    ran = []

    def fake_run(cmd, **k):
        ran.append(cmd)
        d = Path(cmd[cmd.index("--target") + 1])
        (d / "config.json").write_text("{}")
        return NS(returncode=0)
    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    said = []
    d = fetch.fetch_repo("model", "a/b", "main", say=said.append)
    assert "get" in ran[0] and "--kind" in ran[0] and (d / fetch.MARKER).is_file()
    assert any("on login01" in m for m in said) and any("accept its licence" in m for m in said)
    monkeypatch.setattr("bhaskera.diskguard.shutil.disk_usage", lambda p: NS(free=1 << 30))
    with pytest.raises(dl.DownloadError, match="not enough space"):
        fetch.fetch_repo("model", "c/d", "main", say=lambda m: None)


def test_the_command_line_prints_only_the_recipe_path_on_stdout(run, capsys, monkeypatch):
    p = recipe(run, model={"name": "/m"}, data={"name": "local", "tokenized_path": "/c"})
    assert fetch.main([p]) == 0
    assert capsys.readouterr().out.strip() == p
    monkeypatch.setenv("BHASKERA_NO_FETCH", "1")
    p2 = recipe(run, model={"name": "Qwen/Qwen3-1.7B"}, data={"name": "local"})
    assert fetch.main([p2]) == fetch.PLAN_ONLY
    out = capsys.readouterr()
    assert out.out.strip() == p2 and "would download model" in out.err
    assert fetch.main([]) == 2
