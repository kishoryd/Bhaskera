"""The tokenise recipes shipped in slurm/recipes: there is one for each common starting point and each names what the tokeniser needs."""
from __future__ import annotations

from pathlib import Path

import yaml

RECIPES = sorted((Path(__file__).resolve().parents[4] / "slurm" / "recipes").glob("tokenize-*.yaml"))


def test_one_for_each_starting_point():
    names = {p.stem for p in RECIPES}
    assert {"tokenize-chatml", "tokenize-alpaca", "tokenize-hf-name", "tokenize-plaintext", "tokenize-prompt-completion"} <= names


def test_each_names_what_the_tokeniser_needs():
    for p in RECIPES:
        d = yaml.safe_load(p.read_text())
        assert d["model"]["name"], p.name
        assert d["data"]["name"] == "local" and d["data"]["seq_len"] and d["data"]["cache_dir"] and d["data"]["train_path"], p.name


def test_the_layout_matches_the_name():
    want = {"tokenize-alpaca": "alpaca", "tokenize-chatml": "chatml", "tokenize-hf-name": "alpaca", "tokenize-prompt-completion": "prompt_completion"}
    for p in RECIPES:
        d = yaml.safe_load(p.read_text())
        if p.stem in want:
            assert d["data"]["format"] == want[p.stem], p.name
    plain = yaml.safe_load((RECIPES[0].parent / "tokenize-plaintext.yaml").read_text())["data"]
    assert "format" not in plain and plain["format_options"]["chunk"] is True
