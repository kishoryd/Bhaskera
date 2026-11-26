"""A tokenised cache is named after the data it was made from, so different data never reuses another's cache."""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

DATA = Path(__file__).resolve().parents[2] / "data"


def load(name, file, monkeypatch, stubs=()):
    for n in stubs:
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    spec = importlib.util.spec_from_file_location(name, DATA / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_signature_follows_the_files_and_ignores_where_they_are_listed(tmp_path, monkeypatch):
    ck = load("cache_key_t", "cache_key.py", monkeypatch)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("one\n")
    b.write_text("two\n")
    s = ck.files_signature([str(a), str(b)])
    assert len(s) == 16 and ck.files_signature([str(b), str(a)]) == s                       # the order given does not matter
    assert ck.files_signature([str(a)]) != s                                                # other files, other signature
    a.write_text("one more line\n")
    assert ck.files_signature([str(a), str(b)]) != s                                        # an edited file changes it
    os.utime(b, (1, 1))
    assert ck.files_signature([str(b)]) != ck.files_signature([str(b), str(a)])
    assert ck.files_signature([]) == ""


def test_a_folder_counts_every_file_inside_it(tmp_path, monkeypatch):
    ck = load("cache_key_t2", "cache_key.py", monkeypatch)
    d = tmp_path / "shards"
    (d / "sub").mkdir(parents=True)
    (d / "x.parquet").write_text("1")
    s1 = ck.files_signature([str(d)])
    (d / "sub" / "y.parquet").write_text("22")
    assert ck.files_signature([str(d)]) != s1
    moved = tmp_path / "moved"
    d.rename(moved)
    assert ck.files_signature([str(moved)]) == ck.files_signature([str(moved)]) != ""        # a folder is named by its contents, not its place


def stubbed_tokenize(monkeypatch):
    ray, rdata, torch, dist = (types.ModuleType(n) for n in ("ray", "ray.data", "torch", "torch.distributed"))
    ray.data, torch.distributed = rdata, dist
    rdata.Dataset = object
    np = types.ModuleType("numpy")
    for n, m in (("ray", ray), ("ray.data", rdata), ("torch", torch), ("torch.distributed", dist), ("numpy", np)):
        monkeypatch.setitem(sys.modules, n, m)
    pkg = types.ModuleType("bhaskera.data")
    pkg.__path__ = [str(DATA)]
    monkeypatch.setitem(sys.modules, "bhaskera.data", pkg)
    return load("tokenize_t", "tokenize.py", monkeypatch)


def test_the_cache_name_changes_with_the_source_and_is_unchanged_without_one(monkeypatch):
    tk = stubbed_tokenize(monkeypatch)
    plain = tk._cache_version_hash("/m/Qwen", 512, "local_train", "alpaca", {})
    assert tk._cache_version_hash("/m/Qwen", 512, "local_train", "alpaca", {}, "") == plain          # built-in datasets and old caches keep their names
    a = tk._cache_version_hash("/m/Qwen", 512, "local_train", "alpaca", {}, "aaaa")
    b = tk._cache_version_hash("/m/Qwen", 512, "local_train", "alpaca", {}, "bbbb")
    assert len({plain, a, b}) == 3 and a == tk._cache_version_hash("/m/Qwen", 512, "local_train", "alpaca", {}, "aaaa")


def test_the_signature_of_a_split_comes_from_its_own_files(monkeypatch, tmp_path):
    tk = stubbed_tokenize(monkeypatch)
    lc = types.ModuleType("bhaskera.data.datasets.local_chat")
    lc._resolve_files = lambda path: [path]
    monkeypatch.setitem(sys.modules, "bhaskera.data.datasets", types.ModuleType("bhaskera.data.datasets"))
    monkeypatch.setitem(sys.modules, "bhaskera.data.datasets.local_chat", lc)
    ck = load("cache_key_real", "cache_key.py", monkeypatch)
    monkeypatch.setitem(sys.modules, "bhaskera.data.cache_key", ck)
    t, v = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    t.write_text("train rows\n")
    v.write_text("validation rows, a different size\n")
    cfg = NS(data=NS(name="local", train_path=str(t), val_path=str(v), path=None))
    st, sv = tk._source_signature(cfg, "local_train"), tk._source_signature(cfg, "local_val")
    assert st and sv and st != sv
    assert tk._source_signature(NS(data=NS(name="ultrachat")), "ultrachat_train") == ""                # a built-in dataset has no local files
    assert tk._source_signature(NS(data=NS(name="local", train_path=None, val_path=None, path=None)), "local_train") == ""
    t.write_text("edited\n")
    assert tk._source_signature(cfg, "local_train") != st
