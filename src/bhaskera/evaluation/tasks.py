"""
bhaskera.evaluation.tasks
=========================
Benchmark definitions: each task turns local JSONL examples into scoring
requests and turns the scores back into metrics.

Prompt formats follow lm-evaluation-harness (v0.4 default task configs) so
numbers are comparable with published results: same query text, same
" " target delimiter, same context/continuation tokenization (encode the
pair jointly, then split, so a word merged across the boundary is not
mis-tokenized), no BOS added.

Knowledge / reasoning (multiple choice, metric acc and acc_norm, where
acc_norm divides each choice's log-likelihood by its length in bytes):
    hellaswag, arc_easy, arc_challenge, mmlu, winogrande (acc only)
Language modelling (rolling disjoint windows over whole documents):
    wikitext2 -> word_perplexity, byte_perplexity, bits_per_byte
Agentic tool use (BFCL v3, rendered with the model's own tool-calling
chat template):
    bfcl_simple, bfcl_multiple -> exact_match (every token of the gold call
    is the model's greedy choice: greedy decoding would emit exactly that
    call), gold_call_nll (mean per-token NLL of the gold call), and for
    bfcl_multiple tool_selection_acc (the correct function name out of the
    offered ones is the most likely). exact_match is strict: one canonical
    argument serialization, so it lower-bounds BFCL's AST accuracy.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass

from bhaskera.evaluation.scoring import Request, Score


@dataclass
class TaskRun:
    requests: list[Request]
    finalize: Callable[[list[Score]], dict]


class TaskUnsupported(Exception):
    """This model cannot run this task (e.g. no tool-calling template)."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_jsonl(data_dir: str, name: str, limit: int = 0) -> list[dict]:
    path = os.path.join(data_dir, f"{name}.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found; run scripts/fetch_eval_data.py --out {data_dir} "
            "on a node with internet access")
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def encode_pair(tok, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """lm-eval's _encode_pair: trailing context whitespace moves to the
    continuation, the pair is encoded jointly and split at the context."""
    n = len(context) - len(context.rstrip())
    if n:
        continuation = context[-n:] + continuation
        context = context[:-n]
    whole = tok(context + continuation, add_special_tokens=False)["input_ids"]
    ctx = tok(context, add_special_tokens=False)["input_ids"]
    return ctx, whole[len(ctx):]


def _stderr(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else 0.0


def multiple_choice(tok, examples: list[tuple[list[str], list[str], int]],
                    byte_norm: bool = True) -> TaskRun:
    """examples: (contexts, continuations, gold). One context is repeated
    for every choice (most tasks), or one per choice (winogrande)."""
    requests, spans = [], []
    for contexts, conts, gold in examples:
        if len(contexts) == 1:
            contexts = contexts * len(conts)
        start = len(requests)
        for c, x in zip(contexts, conts):
            requests.append(Request(*encode_pair(tok, c, x)))
        spans.append((start, len(conts), [len(x.encode()) for x in conts], gold))

    def finalize(scores: list[Score]) -> dict:
        acc = acc_norm = 0
        for start, k, nbytes, gold in spans:
            lls = [scores[start + j].logprob for j in range(k)]
            acc += max(range(k), key=lambda j: lls[j]) == gold
            acc_norm += max(range(k), key=lambda j: lls[j] / max(nbytes[j], 1)) == gold
        n = len(spans)
        out = {"acc": acc / n, "acc_stderr": _stderr(acc / n, n), "n": n}
        if byte_norm:
            out.update(acc_norm=acc_norm / n, acc_norm_stderr=_stderr(acc_norm / n, n))
        return out

    return TaskRun(requests, finalize)


# ---------------------------------------------------------------------------
# Knowledge / reasoning
# ---------------------------------------------------------------------------

def _hellaswag_pre(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hellaswag(tok, rows, _cfg) -> TaskRun:
    ex = []
    for r in rows:
        query = _hellaswag_pre(r["activity_label"] + ": " + r["ctx_a"] + " "
                               + r["ctx_b"].capitalize())
        ex.append(([query], [" " + _hellaswag_pre(e) for e in r["endings"]], r["label"]))
    return multiple_choice(tok, ex)


def arc(tok, rows, _cfg) -> TaskRun:
    ex = [([f"Question: {r['question']}\nAnswer:"], [" " + c for c in r["choices"]], r["label"])
          for r in rows]
    return multiple_choice(tok, ex)


def mmlu(tok, rows, _cfg) -> TaskRun:
    ex = []
    for r in rows:
        subject = " ".join(r["subject"].split("_"))
        opts = "".join(f"\n{'ABCD'[i]}. {c}" for i, c in enumerate(r["choices"]))
        text = (f"The following are multiple choice questions (with answers) about "
                f"{subject}.\n\n{r['question'].strip()}{opts}\nAnswer:")
        ex.append(([text], [" A", " B", " C", " D"], r["label"]))
    return multiple_choice(tok, ex, byte_norm=False)


def winogrande(tok, rows, _cfg) -> TaskRun:
    ex = []
    for r in rows:
        i = r["sentence"].index("_")
        contexts = [r["sentence"][:i] + r["option1"], r["sentence"][:i] + r["option2"]]
        target = " " + r["sentence"][i + 1:].strip()
        ex.append((contexts, [target, target], r["label"]))
    return multiple_choice(tok, ex, byte_norm=False)


# ---------------------------------------------------------------------------
# Language modelling
# ---------------------------------------------------------------------------

def wikitext_detokenize(s: str) -> str:
    """lm-eval's wikitext_detokenizer."""
    s = s.replace("s '", "s'")
    s = re.sub(r"/' [0-9]/", r"/'[0-9]/", s)
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    for p in (":", ";", ".", "!", "?", ","):
        s = s.replace(f" {p} ", f"{p} ")
    s = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", s)
    s = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", s)
    s = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", s)
    s = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', s)
    s = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", s)
    s = s.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    s = s.replace(" " + chr(176) + " ", chr(176))
    s = s.replace(" \n", "\n").replace("\n ", "\n")
    s = s.replace(" N ", " 1 ").replace(" 's", "'s")
    return s


def rolling_windows(tokens: list[int], max_length: int) -> list[Request]:
    """Disjoint windows predicting every token once (context_len=1 in
    lm-eval terms); the first window is scored from the prefix token."""
    reqs = []
    for start in range(0, len(tokens), max_length):
        reqs.append(Request(tokens[max(0, start - max_length):start],
                            tokens[start:start + max_length]))
    return reqs


def wikitext2(tok, rows, cfg) -> TaskRun:
    texts = [wikitext_detokenize(r["text"]) for r in rows]
    requests, spans = [], []
    for t in texts:
        start = len(requests)
        requests += rolling_windows(tok(t, add_special_tokens=False)["input_ids"], cfg.max_length)
        spans.append((start, len(requests)))
    n_words = sum(len(re.split(r"\s+", t)) for t in texts)
    n_bytes = sum(len(t.encode()) for t in texts)

    def finalize(scores: list[Score]) -> dict:
        ll = sum(s.logprob for s in scores)
        n_tok = sum(s.n_tokens for s in scores)
        return {"word_perplexity": math.exp(-ll / n_words),
                "byte_perplexity": math.exp(-ll / n_bytes),
                "bits_per_byte": -ll / n_bytes / math.log(2),
                "token_perplexity": math.exp(-ll / n_tok),
                "n": len(texts), "tokens": n_tok}

    return TaskRun(requests, finalize)


# ---------------------------------------------------------------------------
# Agentic tool use (BFCL)
# ---------------------------------------------------------------------------

def _gold_args(call_args: dict, tool: dict) -> dict:
    """First acceptable value per argument, in the schema's parameter order;
    "" in BFCL means "may be omitted", so such arguments are left out."""
    props = list((tool.get("parameters") or {}).get("properties", {}))
    order = props + [k for k in call_args if k not in props]
    out = {}
    for k in order:
        vals = call_args.get(k)
        if vals and vals[0] != "":
            out[k] = vals[0]
    return out


def _call_continuation(tok, row: dict, prompt: str, fn: str, fn_args: dict, kw: dict) -> str:
    """Text the model should generate after `prompt` to call fn(fn_args),
    in the model's own tool-call syntax."""
    from bhaskera.data.formats.agentic import AgenticTemplateError, render

    full = render(tok, row["messages"] + [{"role": "assistant", "tool_calls": [
        {"name": fn, "arguments": fn_args}]}], row["tools"], False, kw)
    if not full.startswith(prompt):
        raise AgenticTemplateError("rendered call does not extend the prompt")
    return full[len(prompt):]


def bfcl(tok, rows, cfg, selection: bool) -> TaskRun:
    from bhaskera.data.formats.agentic import AgenticTemplateError, render

    kw = cfg.chat_template_kwargs
    requests, items, skipped = [], [], 0
    for r in rows:
        call = r["ground_truth"][0]
        name = next(iter(call))
        tools = {t["name"]: t for t in r["tools"]}
        args = _gold_args(call[name], tools.get(name, {}))
        try:
            prompt = render(tok, r["messages"], r["tools"], True, kw)
            gold_cont = _call_continuation(tok, r, prompt, name, args, kw)
            names = list(tools) if selection else []
            prefixes = []
            for fn in names:
                # Candidates differ only in the function name; scoring stops
                # right after it, so arguments cannot favour any candidate.
                c = _call_continuation(tok, r, prompt, fn, {}, kw)
                prefixes.append(c[:c.index(fn) + len(fn)])
        except AgenticTemplateError as exc:
            if not requests and not skipped:   # first row: the template itself can't
                raise TaskUnsupported(str(exc)) from exc
            skipped += 1
            continue
        start = len(requests)
        requests.append(Request(*encode_pair(tok, prompt, gold_cont)))
        requests += [Request(*encode_pair(tok, prompt, p)) for p in prefixes]
        items.append((start, names.index(name) if selection else None, len(names)))

    def finalize(scores: list[Score]) -> dict:
        n = len(items)
        exact = sum(scores[s].greedy for s, _, _ in items)
        nll = sum(-scores[s].logprob for s, _, _ in items)
        ntok = sum(scores[s].n_tokens for s, _, _ in items)
        out = {"exact_match": exact / n, "exact_match_stderr": _stderr(exact / n, n),
               "gold_call_nll": nll / ntok, "n": n, "skipped": skipped}
        if selection:
            sel = sum(max(range(k), key=lambda j: scores[s + 1 + j].logprob) == g
                      for s, g, k in items)
            out.update(tool_selection_acc=sel / n,
                       tool_selection_acc_stderr=_stderr(sel / n, n))
        return out

    return TaskRun(requests, finalize)


TASKS: dict[str, Callable] = {
    "hellaswag": hellaswag,
    "arc_easy": arc,
    "arc_challenge": arc,
    "mmlu": mmlu,
    "winogrande": winogrande,
    "wikitext2": wikitext2,
    "bfcl_simple": lambda tok, rows, cfg: bfcl(tok, rows, cfg, selection=False),
    "bfcl_multiple": lambda tok, rows, cfg: bfcl(tok, rows, cfg, selection=True),
}


def build(name: str, tok, cfg) -> TaskRun:
    """cfg is an EvaluationConfig (data_dir, limit, max_length, ...)."""
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    if not cfg.data_dir:
        raise ValueError("evaluation.data_dir is not set")
    return TASKS[name](tok, load_jsonl(cfg.data_dir, name, cfg.limit), cfg)
