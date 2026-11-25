"""Dataset versions: a name, a number and a content fingerprint for the data a run trains on.

A version is one registered state of a data folder (raw files or a tokenised cache). The registry is one small JSON file in
the run folder's settings (``<run folder>/.bhaskera/datasets.json``, see ``paths.state_dir``), so it needs no server.

* Registering the same content again finds the version it already has; changed content becomes the next number. Versions are never rewritten.
* The fingerprint reads every byte once (SHA-256 over the file names, sizes and contents). A *quick signature* of names, sizes and
  modification times is stored too, so that "has this folder changed since it was registered?" can be answered without reading the files.
* A training run records the version of its data as run tags (``dataset.name``, ``dataset.version``, ``dataset.hash``, ``dataset.status``).

Command line: ``python -m bhaskera.dataversions add|list|show|verify``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import paths

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
KINDS = ("raw", "tokenized")
CHUNK = 8 << 20
ROWS_LIMIT = 256 << 20            # count the lines of raw JSONL only up to this many bytes
_lock = threading.Lock()


class DataError(ValueError):
    """A problem with a name, a folder or a version, in words for the person who asked."""


def registry_file() -> Path:
    return paths.state_dir() / "datasets.json"


def _load() -> dict:
    try:
        data = json.loads(registry_file().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: dict) -> None:
    f = registry_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(f.parent), prefix=".datasets-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(data, out, indent=1, sort_keys=True)
        os.replace(tmp, f)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _files(root: Path) -> list[Path]:
    """The files of a folder (or the file itself), sorted, without Hugging Face's download bookkeeping."""
    if root.is_file():
        return [root]
    out = []
    for dirpath, dirnames, names in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != ".cache")
        out += [Path(dirpath) / n for n in sorted(names)]
    return out


def _rel(root: Path, f: Path) -> str:
    return f.name if root.is_file() else str(f.relative_to(root))


def quick_signature(root: Path) -> str:
    """Names, sizes and modification times: cheap, and changes when a file is edited, added or removed."""
    h = hashlib.sha256()
    for f in _files(root):
        st = f.stat()
        h.update(f"{_rel(root, f)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:24]


def fingerprint(root: Path, progress: Optional[Callable[[int], None]] = None, stop: Optional[threading.Event] = None) -> tuple[str, int, int]:
    """(SHA-256 of the content, number of files, bytes). ``progress`` gets the bytes read so far."""
    h = hashlib.sha256()
    nfiles = nbytes = 0
    for f in _files(root):
        size = f.stat().st_size
        h.update(f"{_rel(root, f)}\0{size}\0".encode())
        with open(f, "rb") as fh:
            while True:
                if stop is not None and stop.is_set():
                    raise DataError("stopped")
                block = fh.read(CHUNK)
                if not block:
                    break
                h.update(block)
                nbytes += len(block)
                if progress:
                    progress(nbytes)
        nfiles += 1
    return h.hexdigest(), nfiles, nbytes


def describe(root: Path) -> dict:
    """What a folder holds: tokenised cache (rows, sequence length, tokenizer) or raw files (lines of JSONL, when small)."""
    from .launcher import recipes
    info = recipes.cache_info(root) if root.is_dir() else None
    if info:
        return {"kind": "tokenized", "rows": info.get("rows"), "seq_len": info.get("seq_len"), "tokenizer": info.get("model") or None}
    rows, small = 0, 0
    for f in _files(root):
        if f.suffix.lower() in (".jsonl", ".json"):
            size = f.stat().st_size
            small += size
            if small > ROWS_LIMIT:
                return {"kind": "raw", "rows": None, "seq_len": None, "tokenizer": None}
            with open(f, "rb") as fh:
                rows += sum(block.count(b"\n") for block in iter(lambda: fh.read(CHUNK), b""))
    return {"kind": "raw", "rows": rows or None, "seq_len": None, "tokenizer": None}


def _check_name(name) -> str:
    if not isinstance(name, str) or not NAME.match(name):
        raise DataError("a dataset name is letters, digits, _, - and . (at most 64), starting with a letter or digit")
    return name


def _folder(path) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise DataError("the data folder is missing")
    p = Path(path.strip()).expanduser()
    if not p.is_absolute():
        raise DataError("the data folder must be a full path (starting with /)")
    if not p.exists():
        raise DataError(f"{p} does not exist on this machine")
    if not (p.is_dir() or p.is_file()):
        raise DataError(f"{p} is not a file or a folder")
    return p.resolve()


def add(name, path, note: str = "", source: Optional[dict] = None, progress: Optional[Callable[[int], None]] = None,
        stop: Optional[threading.Event] = None) -> dict:
    """Register the data at ``path`` under ``name``. The same content again returns its version (``new`` False); changed content is the next number."""
    name = _check_name(name)
    root = _folder(path)
    if not isinstance(note, str) or len(note) > 2000:
        raise DataError("the note is text of at most 2000 characters")
    digest, nfiles, nbytes = fingerprint(root, progress, stop)
    meta = describe(root)
    with _lock:
        reg = _load()
        versions = reg.setdefault(name, {"versions": []})["versions"]
        for v in versions:
            if v["hash"] == digest:
                if str(root) != v["path"]:
                    v["path"], v["quick"] = str(root), quick_signature(root)      # the same content, kept in another place now
                    _save(reg)
                return {**v, "name": name, "new": False}
        v = {"version": max((x["version"] for x in versions), default=0) + 1, "hash": digest, "path": str(root), "quick": quick_signature(root),
             "files": nfiles, "bytes": nbytes, "created": int(time.time()), "note": note.strip(), "source": source or {"type": "path", "path": str(root)},
             "user": os.environ.get("USER", ""), **meta}
        versions.append(v)
        _save(reg)
    return {**v, "name": name, "new": True}


def status(v: dict) -> str:
    """'ok' when the files are as they were registered, 'changed' when not, 'missing' when the folder is gone (no file is read)."""
    root = Path(v["path"])
    if not root.exists():
        return "missing"
    try:
        return "ok" if quick_signature(root) == v.get("quick") else "changed"
    except OSError:
        return "missing"


def listing() -> list[dict]:
    """Every dataset with its versions, newest first, each with its quick status."""
    out = []
    for name, entry in sorted(_load().items()):
        versions = sorted(entry.get("versions", []), key=lambda v: -v["version"])
        out.append({"name": name, "latest": versions[0]["version"] if versions else None,
                    "versions": [{**v, "status": status(v)} for v in versions]})
    return out


def get(name, version=None) -> dict:
    entry = _load().get(name)
    if not entry or not entry.get("versions"):
        raise DataError(f"no dataset is registered as {name}")
    versions = entry["versions"]
    if version in (None, "", "latest"):
        v = max(versions, key=lambda x: x["version"])
    else:
        try:
            wanted = int(str(version).lstrip("v"))
        except ValueError:
            raise DataError("a version is a number, such as 2")
        found = [x for x in versions if x["version"] == wanted]
        if not found:
            raise DataError(f"{name} has no version {wanted}")
        v = found[0]
    return {**v, "name": name}


def verify(name, version=None, progress: Optional[Callable[[int], None]] = None, stop: Optional[threading.Event] = None) -> dict:
    """Read the files again and compare with the registered fingerprint. {'status': ok|changed|missing, ...}."""
    v = get(name, version)
    root = Path(v["path"])
    if not root.exists():
        return {"name": name, "version": v["version"], "status": "missing", "path": v["path"]}
    digest, _, _ = fingerprint(root, progress, stop)
    return {"name": name, "version": v["version"], "status": "ok" if digest == v["hash"] else "changed", "path": v["path"]}


def remove(name, version) -> dict:
    """Forget one version (the files stay where they are)."""
    v = get(name, version)
    with _lock:
        reg = _load()
        reg[name]["versions"] = [x for x in reg[name]["versions"] if x["version"] != v["version"]]
        if not reg[name]["versions"]:
            del reg[name]
        _save(reg)
    return {"name": name, "version": v["version"], "removed": True}


def for_path(path) -> Optional[dict]:
    """The registered version of the data at ``path``: the one that matches the files now, else the newest that was registered at that place (status 'changed')."""
    try:
        target = str(Path(path).resolve())
    except (OSError, TypeError, ValueError):
        return None
    best = None
    for name, entry in _load().items():
        for v in entry.get("versions", []):
            if v.get("path") != target:
                continue
            st = status(v)
            cand = {**v, "name": name, "status": st}
            if st == "ok":
                if best is None or best["status"] != "ok" or v["version"] > best["version"]:
                    best = cand
            elif best is None or (best["status"] != "ok" and v["version"] > best["version"]):
                best = cand
    return best


def run_tags(cfg) -> dict:
    """Run tags for the data a training config reads (tokenised training data, and validation data when it is registered)."""
    tags: dict[str, str] = {}
    data = getattr(cfg, "data", None)
    for field, prefix in (("tokenized_path", "dataset"), ("val_tokenized_path", "dataset_val")):
        p = getattr(data, field, None) if data is not None else None
        if not p:
            continue
        v = for_path(p)
        if v:
            tags[f"{prefix}.name"] = v["name"]
            tags[f"{prefix}.version"] = str(v["version"])
            tags[f"{prefix}.hash"] = v["hash"][:12]
            tags[f"{prefix}.status"] = v["status"]
    return tags


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bhaskera.dataversions", description="Dataset versions: a name, a number and a fingerprint for your data.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="register a data folder (or file) under a name")
    a.add_argument("--name", required=True)
    a.add_argument("--path", required=True)
    a.add_argument("--note", default="")
    sub.add_parser("list", help="every dataset and its versions")
    for c in ("show", "verify", "remove"):
        s = sub.add_parser(c)
        s.add_argument("ref", help="name or name:version")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "add":
            v = add(args.name, args.path, args.note)
            print(f"{args.name} v{v['version']} {'registered' if v['new'] else 'already registered (same content)'}: {v['files']} files, {v['bytes']} bytes, sha256 {v['hash'][:12]}")
        elif args.cmd == "list":
            for d in listing():
                for v in d["versions"]:
                    print(f"{d['name']}:{v['version']}\t{v['kind']}\t{v.get('rows') or '-'} rows\t{v['status']}\t{v['path']}")
        else:
            name, _, ver = args.ref.partition(":")
            if args.cmd == "show":
                print(json.dumps(get(name, ver or None), indent=1))
            elif args.cmd == "verify":
                r = verify(name, ver or None)
                print(f"{name}:{r['version']} {r['status']}")
                return 0 if r["status"] == "ok" else 1
            else:
                r = remove(name, ver)
                print(f"{name}:{r['version']} forgotten (files untouched)")
    except DataError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
