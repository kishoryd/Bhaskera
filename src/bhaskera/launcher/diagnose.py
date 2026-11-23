"""Say in plain words why a job failed, and on which node.

Reads the job log after a failed run and prints what went wrong, the node it happened on and what to do.
Faults that point at the hardware of a node (a missing GPU, memory errors) also put that node on the list
of faulty nodes (``bad_nodes``), so the next job keeps away from it.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import bad_nodes


@dataclass
class Finding:
    kind: str
    title: str
    hint: str
    hardware: bool                 # a fault of the node itself, not of the recipe
    host: Optional[str] = None


# (kind, regex, title, hint, hardware)
_PATTERNS = [
    ("gpu-missing", r"invalid device ordinal", "a GPU of the node is missing or not usable",
     "The node is faulty. Resubmit; the node is now skipped.", True),
    ("gpu-error", r"CUDA error: (unspecified launch failure|an illegal memory access|uncorrectable ECC|device-side assert)|"
                  r"\bXid\b|uncorrectable ECC", "a GPU reported a hardware error",
     "If it repeats on the same node, that node is faulty. Resubmit; the node is now skipped.", True),
    ("gpu-oom", r"CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError", "a GPU ran out of memory",
     "Lower training.batch_size (or raise grad_accum), shorten data.seq_len, or use more nodes.", False),
    ("host-oom", r"oom_kill|Out of memory: Killed|Detected \d+ oom_kill|Cannot allocate memory", "a node ran out of host memory",
     "Request more host memory: the GPU partitions give only 20 GB per node by default; submit.sh asks for 170 GB.", False),
    ("disk", r"No space left on device|Disk quota exceeded", "a disk is full",
     "Free space in /scratch (checkpoints, caches) or in /tmp, then resubmit.", False),
    ("network", r"NCCL (error|WARN.*(timeout|unhandled system error|Connection reset))|ncclUnhandledCudaError|"
                r"ncclSystemError|Socket Timeout|Watchdog caught collective operation timeout", "the nodes could not talk to each other",
     "Resubmit. If it repeats, a node's network is faulty: run the job with BHASKERA_EXCLUDE set to the node named.", True),
    ("ray-auth", r"InvalidAuthToken", "Ray could not start across the nodes (authentication token)",
     "The installed Ray or grpcio differs from the tested versions; use the central module or the lock file.", False),
    ("time-limit", r"DUE TO TIME LIMIT|TimeLimit", "the job reached its time limit",
     "Resubmit with the same checkpoint folder to continue, or ask for more hours.", False),
    ("node-failure", r"NODE_FAIL|node failure|Node failure", "Slurm reported a node failure",
     "Resubmit; if it repeats, tell the cluster staff which node it was.", True),
]


def _host_of(line: str, ip_map: dict[str, str]) -> Optional[str]:
    m = re.search(r"\bip=(\d+\.\d+\.\d+\.\d+)", line)
    if m and m.group(1) in ip_map:
        return ip_map[m.group(1)]
    m = re.search(r"srun: error: ([\w.-]+): task", line)          # Slurm names the node whose task failed
    if m:
        return m.group(1)
    for host in set(ip_map.values()):                              # or a node of this job named in the line
        if re.search(rf"(?<![\w.-]){re.escape(host)}(?![\w.-])", line):
            return host
    return None


def diagnose(text: str, ip_map: Optional[dict[str, str]] = None) -> list[Finding]:
    """The problems found in a job log, each with the node it happened on when that can be told."""
    ip_map = ip_map or {}
    found: dict[tuple[str, Optional[str]], Finding] = {}
    for line in text.splitlines():
        for kind, rx, title, hint, hardware in _PATTERNS:
            if re.search(rx, line):
                host = _host_of(line, ip_map)
                found.setdefault((kind, host), Finding(kind, title, hint, hardware, host))
    # one finding per kind without a node is dropped when the same kind was found with a node
    named = {k for (k, h) in found if h}
    return [f for (k, h), f in found.items() if h or k not in named]


def report(findings: list[Finding]) -> str:
    if not findings:
        return ("=== why the job failed ===\n  No known cause found in the log; see the first error above.\n"
                "  If it stopped on one node while others kept going, the node may be faulty: resubmit once.")
    lines = ["=== why the job failed ==="]
    for f in findings:
        where = f" on {f.host}" if f.host else ""
        lines.append(f"  - {f.title}{where}")
        lines.append(f"    {f.hint}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(prog="bhaskera-diagnose", description="Explain why a Bhaskera job failed.")
    p.add_argument("log", help="the job log file")
    p.add_argument("--map", help="JSON file {infiniband address: node}, written by the node check")
    p.add_argument("--no-remember", action="store_true", help="do not put faulty nodes on the list")
    args = p.parse_args()
    ip_map = {}
    if args.map:
        try:
            ip_map = json.load(open(args.map))
        except (OSError, ValueError):
            pass
    try:
        text = Path(args.log).read_text(errors="replace")
    except OSError as e:
        print(f"cannot read {args.log}: {e}")
        return
    findings = diagnose(text, ip_map)
    print(report(findings))
    if not args.no_remember:
        for f in findings:
            if f.hardware and f.host:
                bad_nodes.add(f.host, f.title)


if __name__ == "__main__":
    main()
