"""The prompt_completion data format: rows with a prompt and a completion, under any column names."""
from __future__ import annotations

import pytest

pytest.importorskip("ray.data")
from bhaskera.data.formats import FORMAT_REGISTRY, _ensure_builtins_loaded, render_with_format


class Tok:
    chat_template = "x"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_registered_and_joins_prompt_and_completion():
    _ensure_builtins_loaded()
    assert "prompt_completion" in FORMAT_REGISTRY
    row = {"prompt": "2+2?", "completion": "4"}
    assert render_with_format("prompt_completion", row, Tok(), {}) == "2+2?\n4"
    assert render_with_format("prompt_completion", row, Tok(), {"separator": " => "}) == "2+2? => 4"


def test_other_column_names_and_the_chat_template():
    row = {"question": "Capital of France?", "answer": "Paris"}
    opts = {"prompt_field": "question", "completion_field": "answer"}
    assert render_with_format("prompt_completion", row, Tok(), opts) == "Capital of France?\nParis"
    assert render_with_format("prompt_completion", row, Tok(), {**opts, "use_chat_template": True}) == "user:Capital of France?|assistant:Paris"


def test_an_empty_row_renders_to_nothing():
    assert render_with_format("prompt_completion", {"prompt": "", "completion": None}, Tok(), {}) == ""
