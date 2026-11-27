"""Facts about the cluster ExaAI runs on, so that nothing else in the framework has to assume one.

A profile is a small YAML file. It is found, in this order, at ``$BHASKERA_CLUSTER_PROFILE``, at
``~/.bhaskera/cluster.yaml``, or not at all:

    name: My Cluster                # shown on the dashboard
    gpus_per_node: 8
    cpus_per_node: 64               # optional: CPUs given to each job's node
    mem_per_node: 400G              # optional: host memory asked for on each node
    gpu_mem_gb: 80                  # memory of one GPU, for the "will it fit" check before a job is submitted
    download_hosts: [login01, login02]   # machines with internet, for downloads when the machine running the page has none
    net_interface: ib0              # the network interface NCCL and Ray use ("" = none, not checked)
    docs_url: https://example.org/bhaskera
    partitions:                     # smallest jobs first: a job of up to max_nodes nodes goes to that partition
      - {max_nodes: 4,   name: debug, max_hours: 1}
      - {max_nodes: 64,  name: gpu,   max_hours: 48}

Without a profile the framework still runs: ``submit.sh`` then uses ``$BHASKERA_PARTITION`` (or the cluster's default
partition), and the checks that need the numbers above (partition limits, GPU counts) are skipped. ``slurm/clusters/``
holds the profile of PARAM Rudra and an example for other sites.

    python -m bhaskera.launcher.cluster_profile plan <nodes>    # what submit.sh asks: partition, hours, GPUs, CPUs, ...
"""
from __future__ import annotations

from bhaskera import paths

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PartitionRule:
    max_nodes: int
    name: str
    max_hours: Optional[float] = None


@dataclass
class ClusterProfile:
    name: str = ""
    gpus_per_node: Optional[int] = None
    cpus_per_node: Optional[int] = None
    mem_per_node: str = ""
    gpu_mem_gb: Optional[float] = None
    net_interface: str = ""
    download_hosts: list = field(default_factory=list)
    docs_url: str = ""
    partitions: list[PartitionRule] = field(default_factory=list)
    source: str = ""

    def partition_for(self, nodes: int) -> Optional[PartitionRule]:
        """The partition for a job of ``nodes`` nodes; None when the profile has no partition table. Raises
        ValueError when the job is larger than every partition."""
        if not self.partitions:
            return None
        for rule in sorted(self.partitions, key=lambda r: r.max_nodes):
            if nodes <= rule.max_nodes:
                return rule
        raise ValueError(f"at most {max(r.max_nodes for r in self.partitions)} nodes per job on this cluster")


def _from_dict(raw: dict, source: str) -> ClusterProfile:
    def as_int(v):
        return int(v) if v not in (None, "") else None
    rules = [PartitionRule(int(p["max_nodes"]), str(p["name"]), float(p["max_hours"]) if p.get("max_hours") is not None else None)
             for p in (raw.get("partitions") or [])]
    return ClusterProfile(name=str(raw.get("name") or ""), gpus_per_node=as_int(raw.get("gpus_per_node")),
                          cpus_per_node=as_int(raw.get("cpus_per_node")), mem_per_node=str(raw.get("mem_per_node") or ""), gpu_mem_gb=(float(raw["gpu_mem_gb"]) if raw.get("gpu_mem_gb") not in (None, "") else None),
                          net_interface=str(raw.get("net_interface") or ""), download_hosts=[str(h) for h in (raw.get("download_hosts") or [])], docs_url=str(raw.get("docs_url") or ""),
                          partitions=rules, source=source)


def load(path: Optional[str | Path] = None) -> ClusterProfile:
    """The profile at ``path``, or the one found as described above, or an empty profile."""
    import yaml
    candidates = [path] if path else [os.environ.get("BHASKERA_CLUSTER_PROFILE"), paths.state_dir() / "cluster.yaml"]
    for c in candidates:
        if c and Path(c).is_file():
            try:
                raw = yaml.safe_load(Path(c).read_text()) or {}
                return _from_dict(raw, str(c))
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as e:
                print(f"warning: cluster profile {c} is not usable: {e}", file=sys.stderr)
    return ClusterProfile()


_cached: Optional[ClusterProfile] = None


def current() -> ClusterProfile:
    """The profile in use (read once per process)."""
    global _cached
    if _cached is None:
        _cached = load()
    return _cached


def reset() -> None:
    """Forget the cached profile (tests)."""
    global _cached
    _cached = None


def plan(nodes: int) -> str:
    """One line for submit.sh: 'partition max_hours gpus_per_node cpus_per_node net_interface mem', '-' for unknown."""
    p = current()
    try:
        rule = p.partition_for(nodes)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)
    part = (rule.name if rule else os.environ.get("BHASKERA_PARTITION", "")) or "-"
    hours = (f"{rule.max_hours:g}" if rule and rule.max_hours is not None else os.environ.get("BHASKERA_MAX_HOURS", "")) or "-"
    gpus = str(p.gpus_per_node or os.environ.get("BHASKERA_GPUS_PER_NODE", "") or "-")
    cpus = str(p.cpus_per_node or os.environ.get("BHASKERA_CPUS_PER_NODE", "") or "-")
    net = p.net_interface or os.environ.get("BHASKERA_NET_INTERFACE", "") or "-"
    mem = p.mem_per_node or os.environ.get("BHASKERA_MEM_PER_NODE", "") or "-"
    return f"{part} {hours} {gpus} {cpus} {net} {mem}"


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "plan" and args[1].isdigit():
        print(plan(int(args[1])))
        return 0
    if args == ["show"]:
        p = current()
        print(f"profile: {p.source or 'none (generic)'}\nname: {p.name or '-'}  gpus/node: {p.gpus_per_node or '-'}  "
              f"interface: {p.net_interface or '-'}\npartitions: " + (", ".join(f"<= {r.max_nodes} nodes: {r.name} ({r.max_hours or '?'} h)" for r in p.partitions) or "none"))
        return 0
    print("usage: python -m bhaskera.launcher.cluster_profile plan <nodes> | show", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
