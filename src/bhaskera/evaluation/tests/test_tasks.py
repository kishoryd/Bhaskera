"""
Task construction and metrics on real data with a real tokenizer (CPU):
    BHASKERA_TEST_TOKENIZER=/path/to/model BHASKERA_EVAL_DATA=/path/to/eval \
        python -m bhaskera.evaluation.tests.test_tasks

  1. prompts match lm-evaluation-harness: HellaSwag example 0 renders to the
     harness's query; encode_pair splits exactly at the context boundary
  2. every task builds non-empty requests and a tiny random model (with the
     tokenizer's vocabulary) runs end to end: accuracies in [0, 1],
     perplexities > 1, BFCL continuations are the model's own tool-call
     syntax ending in the gold function name
"""
from __future__ import annotations

import os
import sys
import types

import torch

from bhaskera.evaluation import runner
from bhaskera.evaluation import tasks as T


def _cfg(data_dir: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(data_dir=data_dir, limit=6, max_length=256, batch_size=4,
                                 tasks=list(T.TASKS), chat_template_kwargs={"enable_thinking": False})


def check_formats(tok, data_dir: str) -> bool:
    row = T.load_jsonl(data_dir, "hellaswag", 1)[0]
    run = T.hellaswag(tok, [row], None)
    ctx = tok.decode(run.requests[0].context)
    want = "Roof shingle removal: A man is sitting on a roof. He"
    ctx_ok = ctx == want
    c, x = T.encode_pair(tok, "Question: 2+2?\nAnswer:", " 4")
    split_ok = tok.decode(c + x) == "Question: 2+2?\nAnswer: 4" and tok.decode(x) == " 4"
    ok = ctx_ok and split_ok
    print(f"1. hellaswag query {ctx!r} (harness: {want!r}); encode_pair split exact: "
          f"{split_ok}  {'OK' if ok else 'FAIL'}")
    return ok


def check_end_to_end(tok, data_dir: str) -> bool:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64,
        attn_implementation="eager", max_position_embeddings=512)).float()
    cfg = _cfg(data_dir)
    res = runner.evaluate(model, tok, cfg, torch.device("cpu"))
    ok = True
    for name, m in res.items():
        acc_keys = [k for k in ("acc", "acc_norm", "exact_match", "tool_selection_acc") if k in m]
        good = bool(acc_keys) and all(0.0 <= m[k] <= 1.0 for k in acc_keys)
        if name == "wikitext2":
            good = m["word_perplexity"] > 1 and m["token_perplexity"] > 1
        good &= m.get("n", 0) > 0
        ok &= good
        shown = {k: round(v, 4) for k, v in m.items() if k in acc_keys + ["word_perplexity", "n"]}
        print(f"   {name:14s} {shown}  {'OK' if good else 'FAIL'}")
    row = T.load_jsonl(data_dir, "bfcl_multiple", 1)[0]
    run = T.bfcl(tok, [row], cfg, selection=True)
    gold = tok.decode(run.requests[0].continuation)
    names = [t["name"] for t in row["tools"]]
    prefixes = [tok.decode(r.continuation) for r in run.requests[1:]]
    tool_ok = gold.lstrip().startswith("<tool_call>") and all(
        p.endswith(n) for p, n in zip(prefixes, names))
    ok &= tool_ok
    print(f"2. all tasks ran; bfcl gold call {gold[:60]!r}..., candidate prefixes end in "
          f"their names: {tool_ok}  {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    tok_path = os.environ.get("BHASKERA_TEST_TOKENIZER")
    data_dir = os.environ.get("BHASKERA_EVAL_DATA")
    if not (tok_path and data_dir):
        print("set BHASKERA_TEST_TOKENIZER and BHASKERA_EVAL_DATA")
        return 2
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_path)
    ok = check_formats(tok, data_dir) & check_end_to_end(tok, data_dir)
    print("=== TASKS:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
