#!/usr/bin/env python3
"""Prepare a shared Bhaskera quick-start folder (administrator step, run once per model / dataset update).

    export BHASKERA_DIR=/home/apps/bhaskera/1.0.0/src      # or: module load bhaskera/1.0.0
    python make-quickstart.py /home/apps/bhaskera/quickstart  train.jsonl  val.jsonl

Expected layout before the run:

    <QS>/models/<ModelDir>/        Hugging Face model folders (e.g. Qwen3-14B, Llama-3.3-70B-Instruct)

The script, for every recipe in ./recipes whose model folder exists under <QS>/models:
  1. tokenises train/val once per (model, sequence length) into <QS>/cache, as a one-node CPU job
     (skipped when a matching cache already exists);
  2. writes <QS>/recipes/<recipe>.yaml with absolute paths to the model and the cache. Checkpoints go to
     ./checkpoints/<recipe> relative to the user's run folder, so a user needs to change nothing.

Users then only do:  cp -r <QS>/recipes <their run folder>  and  ./submit.sh recipes/<recipe>.yaml <nodes> <hours>
"""
import copy
import glob
import json
import getpass
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def find_cache(cache_dir, model_path, dataset, seq_len):
    for meta in glob.glob(os.path.join(cache_dir, "*", "metadata.json")):
        try:
            m = json.load(open(meta))
        except Exception:
            continue
        if m.get("model_name") == model_path and m.get("dataset_name") == dataset and int(m.get("seq_len", 0)) == seq_len:
            return os.path.dirname(meta)
    return None


def tokenise(qs, model_path, seq_len, train, val, trust):
    name = os.path.basename(model_path)
    cfg = {
        "model": {"name": model_path, "trust_remote_code": trust},
        "data": {"name": "local", "format": "chatml", "format_options": {"messages_field": "messages"},
                 "train_path": train, "val_path": val, "seq_len": seq_len,
                 "cache_dir": os.path.join(qs, "cache"), "overwrite_cache": False,
                 "num_workers": 46, "tokenize_batch_size": 64, "tokenize_compression": "snappy"},
    }
    os.makedirs(os.path.join(qs, "cache"), exist_ok=True)
    cfg_path = os.path.join(qs, "cache", f"tokenize-{name}.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    os.makedirs("logs", exist_ok=True)
    print(f"  tokenising for {name} (seq_len {seq_len}) ...", flush=True)
    cmd = ["sbatch", "--wait", "-A", os.environ.get("SLURM_ACCOUNT") or getpass.getuser()]
    if os.environ.get("BHASKERA_EXCLUDE"):
        cmd.append(f"--exclude={os.environ['BHASKERA_EXCLUDE']}")
    cmd += [os.path.join(HERE, "tokenize.sbatch"), cfg_path]
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    qs, train, val = (os.path.abspath(a) for a in sys.argv[1:4])
    for need in ("BHASKERA_DIR",):
        if need not in os.environ:
            sys.exit(f"export {need} first (see the top of this script)")
    models = os.path.join(qs, "models")
    if not os.path.isdir(models):
        sys.exit(f"{models} does not exist: put the Hugging Face model folders there")
    out = os.path.join(qs, "recipes")
    os.makedirs(out, exist_ok=True)
    made, skipped = [], []
    for path in sorted(glob.glob(os.path.join(HERE, "recipes", "*.yaml"))):
        name = os.path.basename(path)[:-5]
        if name.startswith("tokenize"):
            continue
        cfg = yaml.safe_load(open(path))
        mdir = os.path.basename(cfg["model"]["name"].rstrip("/"))
        model_path = os.path.join(models, mdir)
        if not os.path.isdir(model_path):
            skipped.append(f"{name} (no {mdir})")
            continue
        seq_len = int(cfg["data"]["seq_len"])
        trust = bool(cfg["model"].get("trust_remote_code", False))
        for ds in ("local_train", "local_val"):
            if not find_cache(os.path.join(qs, "cache"), model_path, ds, seq_len):
                tokenise(qs, model_path, seq_len, train, val, trust)
                break
        tr = find_cache(os.path.join(qs, "cache"), model_path, "local_train", seq_len)
        va = find_cache(os.path.join(qs, "cache"), model_path, "local_val", seq_len)
        if not (tr and va):
            sys.exit(f"tokenising {mdir} finished but no cache with seq_len {seq_len} was found in {qs}/cache")
        cfg = copy.deepcopy(cfg)
        cfg["model"]["name"] = model_path
        cfg["data"]["tokenized_path"], cfg["data"]["val_tokenized_path"] = tr, va
        cfg["checkpoint"]["save_dir"] = f"./checkpoints/{name}"
        with open(os.path.join(out, f"{name}.yaml"), "w") as f:
            f.write(f"# Quick-start recipe '{name}': model and tokenised data are on shared storage; nothing to edit.\n")
            f.write(f"# Run from your own folder:  ./submit.sh recipes/{name}.yaml <nodes> <hours>\n")
            f.write(yaml.safe_dump(cfg, sort_keys=False))
        made.append(name)
    print(f"\nwrote {len(made)} recipes to {out}:")
    for n in made:
        print("  ", n)
    if skipped:
        print("skipped (model folder not found under models/):")
        for n in skipped:
            print("  ", n)


if __name__ == "__main__":
    main()
