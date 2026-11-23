"""Inspect, dedupe, filter and split a JSONL file. No tokenizer or cluster needed."""
from __future__ import annotations

import json

from bhaskera.launcher import data_tools as dt


def _chat(user, assistant="ok"):
    msgs = [{"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


def _write(path, rows, extra_lines=()):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        for line in extra_lines:
            f.write(line + "\n")
    return str(path)


def test_inspect_counts_problems(tmp_path):
    rows = [_chat("What is HPC?", "High-performance computing."), _chat("what  is HPC?", "high-performance computing."),
            _chat("No reply here", None), _chat("x" * 2000, "long"), {"messages": []}]
    path = _write(tmp_path / "d.jsonl", rows, extra_lines=["not json", ""])
    s = dt.inspect(path, "chatml", seq_len=100, preview=1)
    assert s["rows"] == 6 and s["unreadable"] == 1 and s["empty"] == 1 and s["no_answer"] == 1
    assert s["duplicates"] == 1 and s["over_seq_len"] == 1 and s["usable"] == 4
    assert s["bad_lines"] == [6] and len(s["examples"]) == 1
    text = dt.report(s)
    for needle in ("not a JSON object", "no reply to learn from", "repeat an earlier row", "longer than seq_len 100", "estimate"):
        assert needle in text


def test_inspect_of_clean_data_reports_none(tmp_path):
    path = _write(tmp_path / "d.jsonl", [_chat(f"question {i}", f"answer {i}") for i in range(5)])
    s = dt.inspect(path, "chatml", seq_len=1000)
    assert s["usable"] == 5 and "none found" in dt.report(s)


def test_alpaca_and_sharegpt_are_understood(tmp_path):
    a = _write(tmp_path / "a.jsonl", [{"instruction": "Say hi", "input": "", "output": "hi"},
                                       {"instruction": "No answer", "input": "", "output": ""}])
    assert dt.inspect(a, "alpaca")["no_answer"] == 1
    g = _write(tmp_path / "g.jsonl", [{"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}])
    assert dt.inspect(g, "sharegpt")["no_answer"] == 0


def test_dedupe_ignores_case_and_spacing(tmp_path):
    src = _write(tmp_path / "d.jsonl", [_chat("Hello  there"), _chat("hello there"), _chat("different")], extra_lines=["junk"])
    out = tmp_path / "clean.jsonl"
    r = dt.dedupe(src, str(out), "chatml")
    assert r == {"kept": 2, "removed_duplicates": 1, "removed_unreadable": 1}
    assert len(out.read_text().splitlines()) == 2


def test_filter_by_length_and_reply(tmp_path):
    src = _write(tmp_path / "d.jsonl", [_chat("a", "b"), _chat("x" * 400, "y" * 400), _chat("mid " * 20, "ok " * 20), _chat("q", None)])
    out = tmp_path / "kept.jsonl"
    r = dt.filter_rows(src, str(out), "chatml", min_tokens=20, max_tokens=200)
    assert r == {"kept": 1, "too_short": 1, "too_long": 1, "no_answer": 1}
    kept_all = dt.filter_rows(src, str(tmp_path / "all.jsonl"), "chatml", require_answer=False)
    assert kept_all["kept"] == 4


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_exact_counts_use_the_tokenizer():
    assert dt.token_count("a b c d", _Tok()) == 4
    assert dt.token_count("a" * 40) == 10 and dt.token_count("") == 0


def test_split_is_repeatable_and_complete(tmp_path):
    src = _write(tmp_path / "d.jsonl", [_chat(f"q{i}") for i in range(500)])
    r1 = dt.split(src, str(tmp_path / "t1"), str(tmp_path / "v1"), 0.1, seed=3)
    r2 = dt.split(src, str(tmp_path / "t2"), str(tmp_path / "v2"), 0.1, seed=3)
    assert r1 == r2 and r1["train"] + r1["validation"] == 500 and 25 <= r1["validation"] <= 80
    assert (tmp_path / "v1").read_text() == (tmp_path / "v2").read_text()
    assert not set((tmp_path / "t1").read_text().splitlines()) & set((tmp_path / "v1").read_text().splitlines())


def test_command_line(tmp_path, capsys):
    src = _write(tmp_path / "d.jsonl", [_chat(f"q{i}") for i in range(50)])
    assert dt.main(["inspect", src, "--seq-len", "512"]) == 0 and "rows            50" in capsys.readouterr().out
    assert dt.main(["split", src, "--val-fraction", "0.2"]) == 0
    assert (tmp_path / "d.train.jsonl").exists() and (tmp_path / "d.val.jsonl").exists()
    assert dt.main(["split", src, "--val-fraction", "2"]) == 1
