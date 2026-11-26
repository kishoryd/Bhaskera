"""Which splits the tokenise command works on: a recipe with no validation file is tokenised for training only, and says so."""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def tok(monkeypatch):
    pkg = types.ModuleType("bhaskera.data")
    pkg.__path__ = []
    reg = types.ModuleType("bhaskera.data.registry")
    reg.RAW_REGISTRY = {"local": lambda cfg, split=None: None, "whole": lambda cfg: None}
    monkeypatch.setitem(sys.modules, "bhaskera.data", pkg)
    monkeypatch.setitem(sys.modules, "bhaskera.data.registry", reg)
    path = Path(__file__).resolve().parents[1] / "tokenize.py"
    spec = importlib.util.spec_from_file_location("tokenize_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cfg(name="local", val=None):
    return NS(data=NS(name=name, val_path=val))


def test_both_splits_when_a_validation_file_is_named(tok):
    assert tok._resolve_splits("both", cfg(val="/data/val.jsonl")) == ["train", "val"]


def test_training_only_when_the_recipe_has_no_validation_file_and_it_says_so(tok, caplog):
    with caplog.at_level(logging.INFO):
        assert tok._resolve_splits("both", cfg(val=None)) == ["train"]
    assert "only the training data is tokenised" in caplog.text


def test_the_other_choices_are_unchanged(tok):
    assert tok._resolve_splits("train", cfg()) == ["train"]
    assert tok._resolve_splits("val", cfg(val="/v")) == ["val"]
    assert tok._resolve_splits("none", cfg()) == [None]
    assert tok._resolve_splits("both", cfg(name="whole")) == [None]               # a builder without splits: one pass
