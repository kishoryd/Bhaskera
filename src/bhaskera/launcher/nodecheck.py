"""Check every node of a job before training starts, so a faulty node is named in seconds instead of
failing the job minutes later.

``train.sbatch`` runs one check per node (``python -m bhaskera.launcher.nodecheck``) and pipes the lines to
``python -m bhaskera.launcher.nodecheck summarize``, which prints a table, remembers faulty nodes
(see ``bad_nodes``) and exits with 1 when any node is faulty.

A node is checked for: all expected GPUs visible and working, no other process holding GPU memory, no
uncorrected GPU memory errors, enough free host memory and ``/tmp`` space, and an address on the cluster's network
interface (``BHASKERA_NET_INTERFACE``, from the cluster profile; not checked when it is not set).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from typing import Optional

from . import bad_nodes

PREFIX = "NODECHECK "
_GIB = 1024 ** 3


def _meminfo_gib(key: str) -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 0.0


def _ib_address(interface: str) -> str:
    try:
        out = subprocess.run(["ip", "-4", "addr", "show", interface], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out)
    return m.group(1) if m else ""


def _gpu_problems(expected: int) -> list[str]:
    """What is wrong with this node's GPUs (empty list when all is well)."""
    if expected <= 0:
        return []
    problems: list[str] = []
    try:
        import torch
        seen = torch.cuda.device_count()
    except Exception as e:                              # no driver, broken CUDA install, ...
        return [f"CUDA cannot start: {str(e)[:120]}"]
    if seen < expected:
        problems.append(f"only {seen} of {expected} GPUs visible")
    for i in range(seen):
        try:
            import torch
            with torch.cuda.device(i):
                block = torch.empty(256 * 1024 ** 2, dtype=torch.uint8, device=f"cuda:{i}")
                block.fill_(1)
                torch.cuda.synchronize()
                del block
                free, total = torch.cuda.mem_get_info()
            if free < 0.9 * total:
                problems.append(f"GPU {i} has {(total - free) / _GIB:.0f} GiB in use by another process")
        except Exception as e:
            problems.append(f"GPU {i} fails: {str(e).splitlines()[0][:100]}")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=ecc.errors.uncorrected.volatile.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        for i, v in enumerate(out.split()):
            if v.isdigit() and int(v) > 0:
                problems.append(f"GPU {i} reports {v} uncorrected memory errors")
    except (OSError, subprocess.SubprocessError):
        pass
    return problems


def check_node(gpus: int = 2, min_host_gib: float = 100.0, min_tmp_gib: float = 20.0, interface: Optional[str] = None) -> dict:
    """Run all checks on this node; the result is one small dict."""
    problems = _gpu_problems(gpus)
    avail = _meminfo_gib("MemAvailable")
    if min_host_gib and avail < min_host_gib:
        problems.append(f"only {avail:.0f} GiB of host memory free")
    tmp_free = shutil.disk_usage("/tmp").free / _GIB
    if tmp_free < min_tmp_gib:
        problems.append(f"only {tmp_free:.0f} GiB free in /tmp")
    if interface is None:
        interface = os.environ.get("BHASKERA_NET_INTERFACE", "")
    ib = _ib_address(interface) if interface else ""
    if interface and not ib:
        problems.append(f"network interface {interface} has no address")
    return {"host": socket.gethostname(), "ok": not problems, "problems": problems, "ib_ip": ib}


def parse_lines(text: str) -> list[dict]:
    """The per-node results found in the output of the per-node checks."""
    results = []
    for line in text.splitlines():
        if PREFIX in line:
            try:
                results.append(json.loads(line.split(PREFIX, 1)[1]))
            except ValueError:
                continue
    return results


def summarize(results: list[dict], expected_nodes: int = 0, remember: bool = True) -> tuple[str, list[str]]:
    """A table for the job log, and the names of the faulty nodes (also remembered for later jobs)."""
    lines = ["=== node check ==="]
    bad: list[str] = []
    for r in sorted(results, key=lambda r: r["host"]):
        lines.append(f"  {r['host']:<12} {'ok' if r['ok'] else 'PROBLEM: ' + '; '.join(r['problems'])}")
        if not r["ok"]:
            bad.append(r["host"])
            if remember:
                bad_nodes.add(r["host"], "; ".join(r["problems"]))
    silent = expected_nodes - len(results) if expected_nodes else 0
    if silent > 0:
        lines.append(f"  {silent} node(s) gave no answer")
        bad_nodes_flag = True
    else:
        bad_nodes_flag = False
    if bad:
        lines.append(f"=== {len(bad)} faulty node(s): {', '.join(bad)}")
        lines.append("    They are remembered: jobs you submit with submit.sh now keep away from them.")
        lines.append(f"    Resubmit the same job; to also exclude by hand: BHASKERA_EXCLUDE={','.join(bad)}")
    elif not bad_nodes_flag:
        lines.append(f"=== all {len(results)} nodes are healthy")
    return "\n".join(lines), bad


def main() -> None:
    p = argparse.ArgumentParser(prog="bhaskera-nodecheck")
    p.add_argument("action", nargs="?", default="check", choices=["check", "summarize"])
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--nodes", type=int, default=0, help="summarize: number of nodes that should have answered")
    p.add_argument("--save-map", help="summarize: write {infiniband address: host} to this file")
    args = p.parse_args()
    if args.action == "check":
        print(PREFIX + json.dumps(check_node(gpus=args.gpus)), flush=True)
        return
    results = parse_lines(sys.stdin.read())
    text, bad = summarize(results, args.nodes)
    print(text, flush=True)
    if args.save_map:
        try:
            json.dump({r["ib_ip"]: r["host"] for r in results if r.get("ib_ip")}, open(args.save_map, "w"))
        except OSError:
            pass
    sys.exit(1 if bad or (args.nodes and len(results) < args.nodes) else 0)


if __name__ == "__main__":
    main()
