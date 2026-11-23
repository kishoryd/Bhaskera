"""The limits Slurm puts on each GPU partition (its QoS), and how much room a user has left.

Slurm partitions often have a QoS with a time limit and per-user limits on running and submitted jobs
(for example a debug partition: 1 hour, 2 running, 4 submitted). Jobs over a limit are refused with a cryptic message
or wait for ever. ``submit.sh`` and the sweep tool use this module to say so first. Which partition a job of a given
size goes to comes from the cluster profile (``cluster_profile.py``); the limits themselves are read from Slurm.
Without a partition table the checks are skipped.

    python -m bhaskera.launcher.qos check <nodes> [hours]     # exit 1 when no job can be submitted now
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

class NoPartitionTable(ValueError):
    """The cluster profile says nothing about partitions."""


def partition_for(nodes: int) -> tuple[str, Optional[float]]:
    """(partition, its maximum hours) for a job of ``nodes`` nodes, from the cluster profile. Raises ValueError when
    the job is too large, and NoPartitionTable when there is no table (then ``$BHASKERA_PARTITION`` may still name one)."""
    from . import cluster_profile
    rule = cluster_profile.current().partition_for(nodes)
    if rule is not None:
        return rule.name, rule.max_hours
    name = os.environ.get("BHASKERA_PARTITION", "")
    if name:
        return name, None
    raise NoPartitionTable("no partition table in the cluster profile")


@dataclass
class QosLimits:
    name: str
    max_hours: Optional[float] = None
    max_running: Optional[int] = None          # jobs per user that may run at once
    max_submitted: Optional[int] = None        # jobs per user that may be queued or running


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_hours(text: str) -> Optional[float]:
    """Slurm time such as '1-00:00:00' or '06:00:00' in hours."""
    text = text.strip()
    if not text or text in ("UNLIMITED", "INFINITE"):
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600


def _int(text: str) -> Optional[int]:
    return int(text) if text.strip().isdigit() else None


def parse_partition_qos(scontrol_text: str) -> Optional[str]:
    """The QoS a partition applies to every job, from `scontrol show partition`."""
    for token in scontrol_text.replace("\n", " ").split():
        if token.startswith("QoS=") and token != "QoS=N/A":
            return token.split("=", 1)[1]
    return None


def parse_qos_limits(sacctmgr_text: str, qos: str) -> QosLimits:
    """From `sacctmgr -n -P show qos format=Name,MaxWall,MaxJobsPU,MaxSubmitPU`."""
    for line in sacctmgr_text.splitlines():
        cols = line.split("|")
        if cols and cols[0] == qos and len(cols) >= 4:
            return QosLimits(qos, parse_hours(cols[1]), _int(cols[2]), _int(cols[3]))
    return QosLimits(qos)


def limits_for(partition: str) -> QosLimits:
    qos = parse_partition_qos(_run(["scontrol", "show", "partition", partition])) or partition
    return parse_qos_limits(_run(["sacctmgr", "-n", "-P", "show", "qos", "format=Name,MaxWall,MaxJobsPU,MaxSubmitPU"]), qos)


def user_jobs(partition: str) -> tuple[int, int]:
    """(running, submitted) jobs of this user in the partition; submitted counts queued and running."""
    out = _run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-p", partition, "-o", "%T"]).split()
    running = sum(1 for s in out if s in ("RUNNING", "COMPLETING"))
    return running, len(out)


def room(limits: QosLimits, running: int, submitted: int) -> Optional[int]:
    """How many more jobs may be submitted now (None: no limit known)."""
    if limits.max_submitted is None:
        return None
    return max(0, limits.max_submitted - submitted)


def summary(partition: str, limits: QosLimits, running: int, submitted: int) -> str:
    bits = []
    if limits.max_hours is not None:
        bits.append(f"{limits.max_hours:g} h per job")
    if limits.max_running is not None:
        bits.append(f"{limits.max_running} running")
    if limits.max_submitted is not None:
        bits.append(f"{limits.max_submitted} submitted")
    return (f"{partition} (QoS {limits.name}): limits per user {', '.join(bits) or 'none known'}; "
            f"you have {submitted} submitted, {running} running")


def check(nodes: int, hours: Optional[float]) -> tuple[int, str]:
    """(exit code, message) for a job of ``nodes`` nodes and ``hours`` hours."""
    try:
        partition, default_hours = partition_for(nodes)
    except NoPartitionTable:
        return 0, ""                               # nothing is known about this cluster: nothing to check
    except ValueError as e:
        return 1, str(e)
    limits = limits_for(partition)
    running, submitted = user_jobs(partition)
    lines = [summary(partition, limits, running, submitted)]
    wall = hours if hours is not None else default_hours
    caps = [x for x in (limits.max_hours, default_hours) if x is not None]
    cap = min(caps) if caps else None
    code = 0
    if wall is not None and cap is not None and wall > cap:
        lines.append(f"note: {partition} allows at most {cap:g} h per job; {wall:g} h would be refused")
        code = 1
    left = room(limits, running, submitted)
    if left == 0:
        lines.append(f"no room: you already have {submitted} jobs submitted in {partition}, the limit. "
                     "Wait for one to finish (squeue -u $USER), then submit again.")
        code = 1
    elif left is not None and limits.max_running is not None and running >= limits.max_running:
        lines.append(f"note: {running} of your jobs already run here; a new one will wait in the queue until one ends.")
    return code, "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="bhaskera-qos", description="Check a job against the partition's QoS limits.")
    p.add_argument("action", choices=["check"])
    p.add_argument("nodes", type=int)
    p.add_argument("hours", type=float, nargs="?")
    args = p.parse_args(argv)
    code, message = check(args.nodes, args.hours)
    print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
