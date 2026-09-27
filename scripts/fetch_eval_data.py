"""
Snapshot evaluation data to local JSONL, for clusters whose compute nodes
have no internet. Run once on a node that does (e.g. the login node):

    python scripts/fetch_eval_data.py --out data1/eval

Writes <out>/<task>.jsonl (one example per line, fields normalised per task)
and <out>/MANIFEST.json (source, split, row count, sha256). `bhaskera-eval`
reads only these files, never the Hub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def _write(out_dir: str, name: str, rows: list, source: str, split: str, manifest: dict) -> None:
    path = os.path.join(out_dir, f"{name}.jsonl")
    with open(path, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    manifest[name] = {"source": source, "split": split, "rows": len(rows), "sha256": digest}
    print(f"  {name:22s} {len(rows):>7d} rows  <- {source} [{split}]", flush=True)


def hellaswag(ds_load):
    ds = ds_load("Rowan/hellaswag", split="validation")
    return [{"ctx_a": r["ctx_a"], "ctx_b": r["ctx_b"], "activity_label": r["activity_label"],
             "endings": r["endings"], "label": int(r["label"])} for r in ds]


def arc(ds_load, subset):
    ds = ds_load("allenai/ai2_arc", subset, split="test")
    rows = []
    for r in ds:
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        rows.append({"question": r["question"], "choices": r["choices"]["text"],
                     "label": labels.index(r["answerKey"])})
    return rows


def mmlu(ds_load):
    ds = ds_load("cais/mmlu", "all", split="test")
    return [{"question": r["question"], "subject": r["subject"], "choices": r["choices"],
             "label": int(r["answer"])} for r in ds]


def winogrande(ds_load):
    ds = ds_load("allenai/winogrande", "winogrande_xl", split="validation")
    return [{"sentence": r["sentence"], "option1": r["option1"], "option2": r["option2"],
             "label": int(r["answer"]) - 1} for r in ds]


def wikitext(ds_load, config, split):
    ds = ds_load("Salesforce/wikitext", config, split=split)
    # Join lines back into articles: wikitext marks titles as " = Title = ".
    docs, cur = [], []
    for line in ds["text"]:
        if line.startswith(" = ") and not line.startswith(" = = ") and cur:
            docs.append("".join(cur).strip())
            cur = []
        cur.append(line)
    if cur:
        docs.append("".join(cur).strip())
    return [{"text": d} for d in docs if d]


def bfcl(category):
    from huggingface_hub import hf_hub_download

    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    q = hf_hub_download(repo, f"BFCL_v3_{category}.json", repo_type="dataset")
    a = hf_hub_download(repo, f"possible_answer/BFCL_v3_{category}.json", repo_type="dataset")
    with open(a) as f:
        answers = {j["id"]: j["ground_truth"] for j in map(json.loads, filter(str.strip, f))}
    with open(q) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    rows = []
    for j in questions:
        gt = answers.get(j["id"])
        if not gt:
            continue
        messages = j["question"][0] if isinstance(j["question"][0], list) else j["question"]
        rows.append({"id": j["id"], "messages": messages, "tools": j["function"],
                     "ground_truth": gt})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="", help="comma-separated subset of tasks")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset

    jobs = {
        "hellaswag":      (lambda: hellaswag(load_dataset), "Rowan/hellaswag", "validation"),
        "arc_easy":       (lambda: arc(load_dataset, "ARC-Easy"), "allenai/ai2_arc", "ARC-Easy/test"),
        "arc_challenge":  (lambda: arc(load_dataset, "ARC-Challenge"), "allenai/ai2_arc", "ARC-Challenge/test"),
        "mmlu":           (lambda: mmlu(load_dataset), "cais/mmlu", "all/test"),
        "winogrande":     (lambda: winogrande(load_dataset), "allenai/winogrande", "winogrande_xl/validation"),
        "wikitext2":      (lambda: wikitext(load_dataset, "wikitext-2-raw-v1", "test"),
                           "Salesforce/wikitext", "wikitext-2-raw-v1/test"),
        "wikitext103_train": (lambda: wikitext(load_dataset, "wikitext-103-raw-v1", "train"),
                              "Salesforce/wikitext", "wikitext-103-raw-v1/train"),
        "bfcl_simple":    (lambda: bfcl("simple"), "gorilla-llm/Berkeley-Function-Calling-Leaderboard", "v3/simple"),
        "bfcl_multiple":  (lambda: bfcl("multiple"), "gorilla-llm/Berkeley-Function-Calling-Leaderboard", "v3/multiple"),
    }
    only = {t for t in args.only.split(",") if t}
    manifest_path = os.path.join(args.out, "MANIFEST.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    failed = []
    for name, (fn, source, split) in jobs.items():
        if only and name not in only:
            continue
        try:
            _write(args.out, name, fn(), source, split, manifest)
        except (OSError, ValueError, KeyError) as e:  # network/Hub, schema: keep going
            failed.append(name)
            print(f"  {name:22s} FAILED: {type(e).__name__}: {e}", flush=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: {manifest_path}" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
