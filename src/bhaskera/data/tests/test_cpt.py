"""
CPT row layout and agentic rendering (CPU; uses a local tokenizer):
    BHASKERA_TEST_TOKENIZER=/path/to/model python -m bhaskera.data.tests.test_cpt

  1. pack(): rows are full, the stream is reproduced exactly in order, one
     EOS per document (never doubled), empty docs skipped, remainder counted
  2. TokenizerActor(cpt=True) on raw text: labels == input_ids, all-ones
     mask, decoded rows are the source documents joined by EOS
  3. TokenizerActor(cpt=True, format=agentic): tool definitions and the
     tool call reach the rows in the model's own syntax
  4. cache key: CPT changes it; non-CPT keys are unchanged from before
"""
from __future__ import annotations

import os
import sys

import numpy as np

from bhaskera.data.cpt import pack
from bhaskera.data.tokenize import TokenizerActor, _cache_version_hash


def check_pack() -> bool:
    docs = [[1, 2, 3], [], [4, 5, 9], [6, 7, 8, 9, 10]]   # 9 is EOS
    rows, dropped = pack(docs, seq_len=4, eos_id=9)
    stream = [1, 2, 3, 9, 4, 5, 9, 6, 7, 8, 9, 10, 9]  # doc 2 already ends in EOS
    ok = (rows.shape == (3, 4) and rows.reshape(-1).tolist() == stream[:12]
          and dropped == len(stream) - 12)
    empty, d0 = pack([[], []], 4, 9)
    ok &= empty.shape == (0, 4) and d0 == 0
    no_eos, _ = pack([[1, 2], [3, 4]], 2, None)
    ok &= no_eos.tolist() == [[1, 2], [3, 4]]
    print(f"1. pack: rows {rows.tolist()}, dropped {dropped}  {'OK' if ok else 'FAIL'}")
    return ok


def check_actor_text(model: str) -> bool:
    actor = TokenizerActor(model, seq_len=16, cpt=True)
    docs = ["The quick brown fox jumps over the lazy dog. " * 3,
            "Continual pre-training packs documents into full rows.", ""]
    out = actor({"text": np.array(docs, dtype=object)})
    ids, mask, labels = out["input_ids"], out["attention_mask"], out["labels"]
    tok = actor.tokenizer
    joined = tok.decode(ids.reshape(-1))
    expect = tok.eos_token.join(d for d in docs if d)
    ok = (ids.shape[1] == 16 and len(ids) > 0 and (mask == 1).all()
          and np.array_equal(labels, ids) and expect.startswith(joined))
    print(f"2. actor raw text: {ids.shape[0]} full rows, labels==ids, mask all 1, "
          f"decoded stream is a prefix of the documents joined by EOS  {'OK' if ok else 'FAIL'}")
    return ok


def check_actor_agentic(model: str) -> bool:
    actor = TokenizerActor(model, seq_len=32, cpt=True, format_name="agentic",
                           format_options={"template_kwargs": {"enable_thinking": False}})
    tools = '[{"name": "get_weather", "description": "Weather for a city", ' \
            '"parameters": {"type": "dict", "properties": {"city": {"type": "string"}}}}]'
    msgs = '[{"role": "user", "content": "Weather in Pune?"}, ' \
           '{"role": "assistant", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Pune"}}]}]'
    out = actor({"messages": np.array([msgs], dtype=object), "tools": np.array([tools], dtype=object)})
    text = actor.tokenizer.decode(out["input_ids"].reshape(-1))
    ok = "get_weather" in text and "Pune" in text and np.array_equal(out["labels"], out["input_ids"])
    print(f"3. actor agentic: tool schema and call present in packed rows  {'OK' if ok else 'FAIL'}")
    return ok


def check_cache_key() -> bool:
    base = _cache_version_hash("m", 4096, "d", "chatml", {"a": 1})
    same = _cache_version_hash("m", 4096, "d", "chatml", {"a": 1}, cpt=False)
    cpt = _cache_version_hash("m", 4096, "d", "chatml", {"a": 1}, cpt=True)
    ok = base == same and cpt != base
    print(f"4. cache key: non-CPT unchanged, CPT distinct  {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    model = os.environ.get("BHASKERA_TEST_TOKENIZER")
    results = [check_pack(), check_cache_key()]
    if model:
        results += [check_actor_text(model), check_actor_agentic(model)]
    else:
        print("2-3 skipped: set BHASKERA_TEST_TOKENIZER to a local model directory")
    ok = all(results)
    print("=== CPT:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
