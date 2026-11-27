"""Nodes a job has found faulty, remembered so that the next jobs keep away from them.

The list is a small text file in the user's home (``~/.bhaskera/bad-nodes``), one line per node:
``<host> <unix time> <reason>``. A node stays on the list for a few days, because a fault is often fixed
by the cluster staff after that; ``bhaskera-nodes forget <host>`` removes it earlier.
``submit.sh`` adds the listed nodes to ``sbatch --exclude`` by itself.
"""
from __future__ import annotations

from bhaskera import paths

import argparse
import os
import time
from pathlib import Path
from typing import Optional

DEFAULT_PATH = paths.state_dir() / "bad-nodes"
MAX_AGE_DAYS = 3.0


def _path(path: Optional[Path]) -> Path:
    """The list file: the one given, else $BHASKERA_BAD_NODES, else ~/.bhaskera/bad-nodes."""
    return Path(path or os.environ.get("BHASKERA_BAD_NODES") or DEFAULT_PATH)


def load(path: Optional[Path] = None, now: float | None = None, max_age_days: float = MAX_AGE_DAYS) -> dict[str, tuple[float, str]]:
    """Nodes that are still on the list: host -> (when it was added, why). Entries older than the limit are ignored."""
    now = time.time() if now is None else now
    found: dict[str, tuple[float, str]] = {}
    try:
        lines = _path(path).read_text().splitlines()
    except OSError:
        return found
    for line in lines:
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        try:
            added = float(parts[1])
        except ValueError:
            continue
        if now - added <= max_age_days * 86400:
            found[parts[0]] = (added, parts[2] if len(parts) > 2 else "")
    return found


def add(host: str, reason: str, path: Optional[Path] = None, now: float | None = None) -> None:
    """Put a node on the list (or refresh it)."""
    current = load(path, now)
    current[host] = (time.time() if now is None else now, " ".join(reason.split()))
    write(current, path)


def forget(host: str, path: Optional[Path] = None) -> bool:
    current = load(path)
    removed = current.pop(host, None) is not None
    write(current, path)
    return removed


def write(nodes: dict[str, tuple[float, str]], path: Optional[Path] = None) -> None:
    path = _path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{h} {int(t)} {why}\n" for h, (t, why) in sorted(nodes.items())))
    except OSError:
        pass                                           # never fail a job over this note


def exclude_list(path: Optional[Path] = None, extra: str = "") -> str:
    """Comma-separated node names for ``sbatch --exclude``: the remembered ones plus ``extra`` (already given)."""
    names = [n for n in extra.split(",") if n] + [h for h in load(path) if h not in extra.split(",")]
    return ",".join(names)


def main() -> None:
    p = argparse.ArgumentParser(prog="bhaskera-nodes", description="Nodes that ExaAI jobs found faulty.")
    p.add_argument("action", nargs="?", default="list", choices=["list", "exclude", "forget"])
    p.add_argument("host", nargs="?")
    p.add_argument("--extra", default="", help="exclude: nodes to add to the output")
    args = p.parse_args()
    if args.action == "exclude":
        print(exclude_list(extra=args.extra))
    elif args.action == "forget":
        if not args.host:
            p.error("forget needs a node name")
        print("removed" if forget(args.host) else "not on the list")
    else:
        nodes = load()
        if not nodes:
            print("No faulty nodes remembered.")
        for host, (when, why) in sorted(nodes.items()):
            print(f"{host}  {time.strftime('%Y-%m-%d %H:%M', time.localtime(when))}  {why}")


if __name__ == "__main__":
    main()
