"""Fetch the Hugging Face models and datasets a recipe names, before a job is submitted.

Compute nodes have no internet, so a recipe has to point at folders on disk. This lets a recipe name them instead:

    model:
      name: Qwen/Qwen3-1.7B                  # a Hugging Face model: downloaded to <run folder>/models/Qwen__Qwen3-1.7B
    data:
      name: local
      train_path: hf://owner/dataset         # a Hugging Face dataset (add @revision to pin one): <run folder>/data/hf/owner__dataset
      format: chatml

or ``data: {name: owner/dataset}`` (the same as ``name: local`` with that ``train_path``). ``./submit.sh`` runs this first, on the login node. It writes a
*resolved copy* of the recipe (``.bhaskera/resolved/<recipe>.yaml``) with the real folders in it and the job uses that copy; your recipe is not changed.
A download that finished before is not repeated. The download runs on a machine that has internet (this one, else a node named in the cluster
profile, ``download_hosts``).

For a dataset that has a train and a validation (or test) part, the files are sorted into ``_splits/train`` and ``_splits/val`` next to it, and ``val_path`` is
filled in when the recipe did not name one.

    python -m bhaskera.launcher.fetch recipes/my.yaml        # prints the path of the recipe to use
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

from .. import diskguard, downloads

PLAN_ONLY = 10                      # exit code when only a plan was asked for (BHASKERA_NO_FETCH=1) and there is something to download
_TRAIN = {"train", "training"}
_VAL_ORDER = ({"validation", "valid", "val"}, {"dev"}, {"test"})
_EXTS = (".jsonl", ".json", ".ndjson", ".parquet", ".csv", ".tsv", ".arrow", ".feather")
_TEXT = (".txt", ".text")            # plain text counts only when the dataset has nothing else (a stray readme.txt is not data)
MARKER = ".bhaskera_fetched.json"


def looks_like_repo(value) -> bool:
    """owner/name, not a path that exists (a recipe's own folder wins) and not an absolute or relative path."""
    if not isinstance(value, str) or value.startswith(("/", ".", "~")) or value.count("/") != 1 or not downloads.REPO.match(value):
        return False
    return not Path(value).exists()


def parse_ref(value: str) -> tuple[str, str]:
    """'hf://owner/name@rev' or 'owner/name@rev' -> ('owner/name', 'rev'); the revision is 'main' when absent."""
    v = value[len("hf://"):] if value.startswith("hf://") else value
    repo, _, rev = v.partition("@")
    return repo, rev or "main"


def _root() -> Path:
    return Path(os.environ.get("BHASKERA_WORKDIR") or os.getcwd())


def _dest(kind: str, repo: str, rev: str) -> Path:
    leaf = repo.replace("/", "__") + ("" if rev == "main" else "@" + rev.replace("/", "_"))
    return _root() / ("models" if kind == "model" else Path("data") / "hf") / leaf


def _fetched(dest: Path, repo: str, rev: str) -> bool:
    try:
        m = json.loads((dest / MARKER).read_text())
    except (OSError, ValueError):
        return False
    return m.get("repo") == repo and m.get("revision") == rev


def fetch_repo(kind: str, repo: str, rev: str, say=print) -> Path:
    """Download ``repo`` into its folder under the run folder (once) and return the folder. Raises downloads.DownloadError."""
    dest = _dest(kind, repo, rev)
    if _fetched(dest, repo, rev):
        say(f"already downloaded: {repo} in {dest}")
        return dest
    info = downloads.preview(kind, repo, rev)
    say(f"downloading {kind} {repo} ({rev}): {info['bytes'] / 2**30:.1f} GB in {info['files']} files -> {dest}"
        + (f", on {info['via']} (this machine has no internet)" if info.get("via") else ""))
    if info.get("gated"):
        say("note: this repository asks you to accept its licence on huggingface.co and to sign in (the Downloads menu of the home page)")
    try:
        diskguard.require(dest, info["bytes"], "this download", margin=1.05)
    except diskguard.DiskError as e:
        raise downloads.DownloadError(str(e))
    dest.mkdir(parents=True, exist_ok=True)
    # the download's own messages go to stderr: stdout is reserved for the path of the recipe to use (./submit.sh reads it)
    done = subprocess.run([sys.executable, "-m", "bhaskera.downloads", "get", "--kind", kind, "--repo", repo, "--revision", rev, "--target", str(dest)],
                          stdout=sys.stderr)
    if done.returncode != 0:
        raise downloads.DownloadError(f"the download of {repo} did not finish (exit code {done.returncode}); run the same command again to continue")
    if kind == "model" and not (dest / "config.json").is_file():
        raise downloads.DownloadError(f"{repo} was downloaded but {dest} has no config.json: it is not a model folder")
    (dest / MARKER).write_text(json.dumps({"repo": repo, "revision": rev, "sha": info.get("sha", ""), "kind": kind}))
    return dest


def _words(path: Path) -> set[str]:
    return {w for w in re.split(r"[_\-.\s/]+", str(path).lower()) if w}


def make_splits(dest: Path) -> tuple[Path, Optional[Path]]:
    """Sort a downloaded dataset's files into <dest>/_splits/train and _splits/val (links, nothing is copied).
    Returns (train folder, validation folder or None). Without any train-named file, every file is training data."""
    every = [f for f in sorted(dest.rglob("*")) if f.is_file() and "_splits" not in f.parts and not f.name.startswith(".")]
    ends = lambda f, exts: f.name.lower().removesuffix(".gz").endswith(exts)
    files = [f for f in every if ends(f, _EXTS)] or [f for f in every if ends(f, _TEXT)]
    if not files:
        raise downloads.DownloadError(f"{dest} has no data files of a supported format ({', '.join(_EXTS + _TEXT)}, optionally .gz)")
    rel = {f: _words(f.relative_to(dest)) for f in files}
    is_val = lambda f: any(rel[f] & group for group in _VAL_ORDER)
    train = [f for f in files if rel[f] & _TRAIN] or [f for f in files if not is_val(f)] or files        # train-named files, else all that are not validation, else all
    val: list[Path] = []
    for group in _VAL_ORDER:                                  # validation, then dev, then test: the first part that exists
        val = [f for f in files if rel[f] & group and f not in train]
        if val:
            break
    out = dest / "_splits"
    for name, group in (("train", train), ("val", val)):
        folder = out / name
        if folder.exists():
            for old in folder.iterdir():
                old.unlink()
        if group:
            folder.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(group):
                (folder / f"{i:04d}_{f.name}").symlink_to(f.resolve())
    return out / "train", (out / "val") if val else None


def resolve(config: str, say=print, fetch=fetch_repo, plan_only: bool = False) -> tuple[str, list[str]]:
    """(path of the recipe to use, what was resolved). The path is the recipe itself when it names nothing to download."""
    raw = yaml.safe_load(Path(config).read_text()) or {}
    model, data = raw.get("model") or {}, raw.get("data") or {}
    jobs: list[tuple[str, str, str]] = []                      # (what, repo, revision)
    if looks_like_repo(model.get("name")):
        jobs.append(("model", *parse_ref(model["name"])))
    train_key = None
    if str(data.get("name", "")).count("/") == 1 and looks_like_repo(data.get("name")) and not any(data.get(k) for k in ("train_path", "path")):
        train_key = "name"
    for key in ("train_path", "val_path", "path"):
        if isinstance(data.get(key), str) and data[key].startswith("hf://"):
            train_key = train_key or key
            jobs.append(("dataset", *parse_ref(data[key])))
    if train_key == "name":
        jobs.append(("dataset", *parse_ref(data["name"])))
    if not jobs:
        return config, []
    if plan_only:
        pending = [j for j in dict.fromkeys(jobs) if not _fetched(_dest(*j[:1], j[1], j[2]), j[1], j[2])]
        for kind, repo, rev in dict.fromkeys(jobs):
            say(f"would download {kind} {repo} ({rev})" if (kind, repo, rev) in pending else f"already downloaded: {repo}")
        if pending:                                             # something is still to be downloaded: only say so
            return config, ["plan"]
        # everything is on disk already: resolve for real (no network is used), so that the dry run can go on to check the rest
    changes: list[str] = []
    if looks_like_repo(model.get("name")):
        repo, rev = parse_ref(model["name"])
        model["name"] = str(fetch("model", repo, rev, say))
        changes.append(f"model.name -> {model['name']}")
    explicit_val = bool(data.get("val_path"))
    for key in ("train_path", "val_path", "path"):
        v = data.get(key)
        if isinstance(v, str) and v.startswith("hf://"):
            repo, rev = parse_ref(v)
            folder = fetch("dataset", repo, rev, say)
            train, val = make_splits(Path(folder))
            if key == "val_path":
                data[key] = str(val or train)
            else:
                data[key] = str(train)
                if val and not explicit_val and key == "train_path":
                    data["val_path"] = str(val)
                    changes.append(f"data.val_path -> {val}")
            changes.append(f"data.{key} -> {data[key]}")
    if train_key == "name":
        repo, rev = parse_ref(data["name"])
        folder = fetch("dataset", repo, rev, say)
        train, val = make_splits(Path(folder))
        data["name"], data["train_path"] = "local", str(train)
        changes.append(f"data.name {repo} -> local, data.train_path -> {train}")
        if val and not data.get("val_path"):
            data["val_path"] = str(val)
            changes.append(f"data.val_path -> {val}")
    raw["model"], raw["data"] = model, data
    out = _root() / ".bhaskera" / "resolved" / Path(config).name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# A copy of " + str(Path(config).resolve()) + " with the Hugging Face names replaced by the folders they were downloaded to.\n" + yaml.safe_dump(raw, sort_keys=False))
    say("resolved: " + "; ".join(changes))
    return str(out), changes


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m bhaskera.launcher.fetch <recipe.yaml>   (prints the path of the recipe to use)", file=sys.stderr)
        return 2
    plan = os.environ.get("BHASKERA_NO_FETCH") == "1"
    try:
        path, changes = resolve(args[0], say=lambda m: print(m, file=sys.stderr), plan_only=plan)
    except downloads.DownloadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (OSError, yaml.YAMLError) as e:
        print(f"error: could not read {args[0]}: {e}", file=sys.stderr)
        return 1
    print(path)
    return PLAN_ONLY if changes == ["plan"] else 0          # only "there is something to download" ends a dry run early


if __name__ == "__main__":
    raise SystemExit(main())
