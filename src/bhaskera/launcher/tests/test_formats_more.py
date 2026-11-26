"""Data formats beyond chatml, alpaca and sharegpt: dolly, qa, scored_response, tool_calling and reasoning."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("ray.data")
from bhaskera.data.formats import FORMAT_REGISTRY, _ensure_builtins_loaded, render_with_format  # noqa: E402


class Tok:
    """A stand-in with a chat template that records what it was given."""
    chat_template = "x"

    def __init__(self):
        self.tools = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        self.tools = tools
        return "|".join(f"{m['role']}:{m['content']}" + (f"<calls {json.dumps(m['tool_calls'])}>" if m.get("tool_calls") else "") for m in messages)


class NoTemplate:
    chat_template = None


def r(name, row, opts=None, tok=None):
    return render_with_format(name, row, tok or Tok(), opts or {})


def test_all_are_registered():
    _ensure_builtins_loaded()
    for n in ("dolly", "qa", "scored_response", "tool_calling", "reasoning"):
        assert n in FORMAT_REGISTRY


def test_dolly_with_and_without_context():
    row = {"instruction": "Summarise", "context": "Long text.", "response": "Short."}
    assert r("dolly", row) == "### Instruction:\nSummarise\n\n### Context:\nLong text.\n\n### Response:\nShort."
    assert r("dolly", {"instruction": "Hi", "response": "Hello"}) == "### Instruction:\nHi\n\n### Response:\nHello"
    assert r("dolly", row, {"use_chat_template": True}) == "user:Summarise\n\nLong text.|assistant:Short."
    assert r("dolly", {"instruction": "", "response": ""}) == ""


def test_qa_with_context_and_other_names():
    assert r("qa", {"question": "Q?", "answer": "A"}) == "Question: Q?\nAnswer: A"
    assert r("qa", {"question": "Q?", "answer": "A", "context": "Passage"}) == "Context: Passage\n\nQuestion: Q?\nAnswer: A"
    assert r("qa", {"q": "Q?", "a": "A"}, {"question_field": "q", "answer_field": "a"}) == "Question: Q?\nAnswer: A"


def test_scored_response_keeps_only_good_answers():
    good, bad = {"prompt": "p", "response": "r", "score": 0.9}, {"prompt": "p", "response": "r", "score": "0.2"}
    assert r("scored_response", good, {"min_score": 0.5}) == "p\nr"
    assert r("scored_response", bad, {"min_score": 0.5}) == ""
    assert r("scored_response", {"prompt": "p", "response": "r"}, {"min_score": 0.5}) == ""          # no score: not trusted
    assert r("scored_response", bad) == "p\nr"                                                          # no threshold: nothing filtered


def test_tool_calling_passes_tools_and_decodes_json_text():
    tok = Tok()
    row = {"messages": [{"role": "user", "content": "Weather in Pune?"},
                        {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": '{"city": "Pune"}'}}]},
                        {"role": "tool", "content": "31C"},
                        {"role": "assistant", "content": "It is 31C."}],
           "tools": json.dumps([{"type": "function", "function": {"name": "weather"}}])}
    out = r("tool_calling", row, tok=tok)
    assert tok.tools == [{"type": "function", "function": {"name": "weather"}}]
    assert '"arguments": {"city": "Pune"}' in out and "tool:31C" in out and out.endswith("assistant:It is 31C.")


def test_tool_calling_without_a_chat_template_writes_tool_call_blocks():
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}]}
    out = r("tool_calling", row, tok=NoTemplate())
    assert "<tool_call>" in out and '"name": "f"' in out and "<|im_start|>assistant" in out


def test_reasoning_from_messages_and_from_columns():
    msgs = {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4", "reasoning_content": "Add them."}]}
    assert r("reasoning", msgs) == "user:2+2?|assistant:<think>\nAdd them.\n</think>\n\n4"
    cols = {"prompt": "2+2?", "reasoning": "Add them.", "answer": "4"}
    assert r("reasoning", cols) == "user:2+2?|assistant:<think>\nAdd them.\n</think>\n\n4"
    done = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>x</think>a", "thinking": "y"}]}
    assert r("reasoning", done) == "user:q|assistant:<think>x</think>a"                               # already tagged: left alone
    assert r("reasoning", {"prompt": "", "answer": ""}) == ""
