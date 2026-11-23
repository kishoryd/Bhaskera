"""Look at, clean and split a JSONL training file before it is tokenised.

    python -m bhaskera.launcher.data_tools inspect train.jsonl --format chatml --seq-len 4096 [--tokenizer <model folder>]
    python -m bhaskera.launcher.data_tools dedupe  train.jsonl clean.jsonl --format chatml
    python -m bhaskera.launcher.data_tools filter  train.jsonl kept.jsonl --format chatml --tokenizer <folder> --min-tokens 8 --max-tokens 4096
    python -m bhaskera.launcher.data_tools split   clean.jsonl --val-fraction 0.02 --seed 0

``inspect`` reports the size and shape of the data and the problems that waste GPU time: rows that cannot be read,
rows with nothing to learn from, exact duplicates, and rows longer than ``--seq-len`` (they are cut off). Lengths are
exact with ``--tokenizer`` and a rough estimate (four characters per token) without it. Everything runs on a login
node and needs nothing but this package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

APPROX_CHARS_PER_TOKEN = 4


def read_rows(path: str | Path) -> Iterator[tuple[int, Optional[dict], str]]:
    """(line number, row or None when the line is not a JSON object, the raw line) for every non-blank line."""
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = None
            yield number, row if isinstance(row, dict) else None, line.rstrip("\n")


def render(row: dict, fmt: str, tokenizer: Any = None) -> str:
    """The text the trainer would see for this row."""
    from bhaskera.data.formats import render_with_format
    return render_with_format(fmt, row, tokenizer, {})


def token_count(text: str, tokenizer: Any = None) -> int:
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return max(1, round(len(text) / APPROX_CHARS_PER_TOKEN)) if text else 0


def normalise(text: str) -> str:
    """Two rows that differ only in case or spacing count as the same."""
    return re.sub(r"\s+", " ", text).strip().lower()


def has_answer(row: dict, fmt: str) -> bool:
    """Whether the row contains a reply to learn from (an assistant message, an output, a gpt turn)."""
    if fmt == "chatml":
        return any(isinstance(m, dict) and m.get("role") == "assistant" and str(m.get("content", "")).strip()
                   for m in row.get("messages") or [])
    if fmt == "alpaca":
        return bool(str(row.get("output", "")).strip())
    if fmt == "sharegpt":
        return any(isinstance(m, dict) and m.get("from") in ("gpt", "assistant") and str(m.get("value", "")).strip()
                   for m in row.get("conversations") or [])
    return True


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0


def inspect(path: str, fmt: str, tokenizer: Any = None, seq_len: int = 0, preview: int = 0) -> dict:
    """Facts about the file: counts, length statistics, problems, and a few rendered examples."""
    total = unreadable = empty = no_answer = duplicates = over = 0
    lengths: list[int] = []
    seen: set[str] = set()
    examples: list[str] = []
    bad_lines: list[int] = []
    for number, row, _raw in read_rows(path):
        total += 1
        if row is None:
            unreadable += 1
            bad_lines.append(number)
            continue
        text = render(row, fmt, tokenizer)
        if not text.strip():
            empty += 1
            continue
        if not has_answer(row, fmt):
            no_answer += 1
        digest = hashlib.sha1(normalise(text).encode()).hexdigest()
        if digest in seen:
            duplicates += 1
        seen.add(digest)
        n = token_count(text, tokenizer)
        lengths.append(n)
        if seq_len and n > seq_len:
            over += 1
        if len(examples) < preview:
            examples.append(text if len(text) <= 700 else text[:700] + " ...")
    usable = len(lengths)
    return {
        "rows": total, "usable": usable, "unreadable": unreadable, "empty": empty, "no_answer": no_answer,
        "duplicates": duplicates, "over_seq_len": over, "seq_len": seq_len, "exact_tokens": tokenizer is not None,
        "tokens_total": sum(lengths),
        "length": {"min": min(lengths, default=0), "mean": round(statistics.mean(lengths)) if lengths else 0,
                   "median": round(statistics.median(lengths)) if lengths else 0,
                   "p95": _percentile(lengths, 0.95), "max": max(lengths, default=0)},
        "bad_lines": bad_lines[:10], "examples": examples,
    }


def report(stats: dict) -> str:
    def pct(n: int) -> str:
        return f"{100 * n / stats['rows']:.1f}%" if stats["rows"] else "0%"
    L = stats["length"]
    unit = "tokens" if stats["exact_tokens"] else "tokens (estimate: 4 characters each; give --tokenizer for exact)"
    lines = [f"rows            {stats['rows']:,}   usable {stats['usable']:,}",
             f"length ({unit})",
             f"                min {L['min']:,}   median {L['median']:,}   mean {L['mean']:,}   p95 {L['p95']:,}   max {L['max']:,}",
             f"total           {stats['tokens_total']:,} tokens"]
    problems = []
    if stats["unreadable"]:
        problems.append(f"{stats['unreadable']:,} lines are not a JSON object ({pct(stats['unreadable'])}); first lines: {stats['bad_lines']}")
    if stats["empty"]:
        problems.append(f"{stats['empty']:,} rows give no text ({pct(stats['empty'])}): wrong --format, or empty rows")
    if stats["no_answer"]:
        problems.append(f"{stats['no_answer']:,} rows have no reply to learn from ({pct(stats['no_answer'])})")
    if stats["duplicates"]:
        problems.append(f"{stats['duplicates']:,} rows repeat an earlier row ({pct(stats['duplicates'])}); `dedupe` removes them")
    if stats["seq_len"] and stats["over_seq_len"]:
        problems.append(f"{stats['over_seq_len']:,} rows are longer than seq_len {stats['seq_len']:,} and will be cut off "
                        f"({pct(stats['over_seq_len'])}); `filter --max-tokens` removes them")
    lines.append("problems        " + ("none found" if not problems else "\n                ".join(problems)))
    for i, text in enumerate(stats["examples"], start=1):
        lines.append(f"\n--- example {i} ---\n{text}")
    return "\n".join(lines)


def dedupe(src: str, dst: str, fmt: str) -> dict:
    """Write the file without rows that repeat an earlier row (ignoring case and spacing). Unreadable lines are dropped."""
    seen: set[str] = set()
    kept = dropped = unreadable = 0
    with open(dst, "w", encoding="utf-8") as out:
        for _n, row, raw in read_rows(src):
            if row is None:
                unreadable += 1
                continue
            digest = hashlib.sha1(normalise(render(row, fmt)).encode()).hexdigest()
            if digest in seen:
                dropped += 1
                continue
            seen.add(digest)
            out.write(raw + "\n")
            kept += 1
    return {"kept": kept, "removed_duplicates": dropped, "removed_unreadable": unreadable}


def filter_rows(src: str, dst: str, fmt: str, tokenizer: Any = None, min_tokens: int = 0, max_tokens: int = 0,
                require_answer: bool = True) -> dict:
    """Keep rows with a reply and a length inside the limits."""
    kept = short = long = no_answer = 0
    with open(dst, "w", encoding="utf-8") as out:
        for _n, row, raw in read_rows(src):
            if row is None:
                continue
            if require_answer and not has_answer(row, fmt):
                no_answer += 1
                continue
            n = token_count(render(row, fmt, tokenizer), tokenizer)
            if min_tokens and n < min_tokens:
                short += 1
            elif max_tokens and n > max_tokens:
                long += 1
            else:
                out.write(raw + "\n")
                kept += 1
    return {"kept": kept, "too_short": short, "too_long": long, "no_answer": no_answer}


def split(src: str, train_out: str, val_out: str, val_fraction: float = 0.02, seed: int = 0) -> dict:
    """A repeatable random split into a training and a validation file."""
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    rng = random.Random(seed)
    train = val = 0
    with open(train_out, "w", encoding="utf-8") as t, open(val_out, "w", encoding="utf-8") as v:
        for _n, row, raw in read_rows(src):
            if row is None:
                continue
            if rng.random() < val_fraction:
                v.write(raw + "\n")
                val += 1
            else:
                t.write(raw + "\n")
                train += 1
    return {"train": train, "validation": val}


def _tokenizer(path: Optional[str]) -> Any:
    if not path:
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=False)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="bhaskera-data", description="Inspect, clean and split a JSONL training file.")
    sub = p.add_subparsers(dest="action", required=True)

    def common(sp, out: bool = True):
        sp.add_argument("input")
        if out:
            sp.add_argument("output")
        sp.add_argument("--format", default="chatml", help="chatml (default), alpaca or sharegpt")
        sp.add_argument("--tokenizer", help="model folder: exact token counts")

    i = sub.add_parser("inspect")
    common(i, out=False)
    i.add_argument("--seq-len", type=int, default=0)
    i.add_argument("--preview", type=int, default=0, help="show this many rendered rows")
    common(sub.add_parser("dedupe"))
    f = sub.add_parser("filter")
    common(f)
    f.add_argument("--min-tokens", type=int, default=0)
    f.add_argument("--max-tokens", type=int, default=0)
    f.add_argument("--keep-without-answer", action="store_true")
    s = sub.add_parser("split")
    s.add_argument("input")
    s.add_argument("--val-fraction", type=float, default=0.02)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--train-out")
    s.add_argument("--val-out")
    args = p.parse_args(argv)

    if args.action == "inspect":
        print(report(inspect(args.input, args.format, _tokenizer(args.tokenizer), args.seq_len, args.preview)))
    elif args.action == "dedupe":
        r = dedupe(args.input, args.output, args.format)
        print(f"kept {r['kept']:,} rows; removed {r['removed_duplicates']:,} duplicates and {r['removed_unreadable']:,} unreadable lines -> {args.output}")
    elif args.action == "filter":
        r = filter_rows(args.input, args.output, args.format, _tokenizer(args.tokenizer), args.min_tokens, args.max_tokens,
                        require_answer=not args.keep_without_answer)
        print(f"kept {r['kept']:,} rows; dropped {r['too_short']:,} too short, {r['too_long']:,} too long, "
              f"{r['no_answer']:,} without a reply -> {args.output}")
    else:
        stem = Path(args.input)
        train_out = args.train_out or str(stem.with_name(stem.stem + ".train.jsonl"))
        val_out = args.val_out or str(stem.with_name(stem.stem + ".val.jsonl"))
        try:
            r = split(args.input, train_out, val_out, args.val_fraction, args.seed)
        except ValueError as e:
            print(f"error: {e}")
            return 1
        print(f"{r['train']:,} training rows -> {train_out}\n{r['validation']:,} validation rows -> {val_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
