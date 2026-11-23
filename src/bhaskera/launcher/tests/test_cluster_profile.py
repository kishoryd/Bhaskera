"""The cluster profile: where partitions, GPU counts and the network interface come from. Nothing here is PARAM Rudra."""
from __future__ import annotations

import pytest

from bhaskera.launcher import cluster_profile as cp
from bhaskera.launcher.tests.conftest import PARAM_RUDRA

OTHER = """
name: Other Cluster
gpus_per_node: 8
cpus_per_node: 64
mem_per_node: 400G
net_interface: eth1
partitions:
  - {max_nodes: 32, name: gpu, max_hours: 48}
  - {max_nodes: 2,  name: debug, max_hours: 1}
"""


def test_the_shipped_profiles_load():
    p = cp.load(PARAM_RUDRA)
    assert (p.name, p.gpus_per_node, p.net_interface) == ("PARAM Rudra", 2, "ib0")
    assert [r.name for r in p.partitions] == ["gpu-debug", "gpu-small", "gpu-large", "gpu-massive"]
    example = cp.load(PARAM_RUDRA.replace("param-rudra", "example"))
    assert example.name == "My Cluster" and example.partitions


def test_a_job_goes_to_the_smallest_partition_it_fits(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(OTHER)
    p = cp.load(f)
    assert p.partition_for(2).name == "debug" and p.partition_for(3).name == "gpu" and p.partition_for(32).name == "gpu"
    with pytest.raises(ValueError, match="at most 32 nodes"):
        p.partition_for(33)


def test_plan_line_for_another_cluster(tmp_path, use_profile, capsys):
    f = tmp_path / "c.yaml"
    f.write_text(OTHER)
    use_profile(f)
    assert cp.plan(2) == "debug 1 8 64 eth1 400G"
    assert cp.plan(10) == "gpu 48 8 64 eth1 400G"
    assert cp.main(["plan", "10"]) == 0 and capsys.readouterr().out.strip() == "gpu 48 8 64 eth1 400G"


def test_without_a_profile_everything_is_unknown_and_the_environment_fills_in(use_profile, monkeypatch):
    use_profile(None)
    assert cp.current().partitions == [] and cp.plan(4) == "- - - - - -"
    monkeypatch.setenv("BHASKERA_PARTITION", "batch")
    monkeypatch.setenv("BHASKERA_GPUS_PER_NODE", "4")
    assert cp.plan(4) == "batch - 4 - - -"


def test_a_broken_profile_is_ignored_with_a_warning(tmp_path, use_profile, capsys):
    f = tmp_path / "bad.yaml"
    f.write_text("partitions: [{name: x}]")                        # max_nodes is missing
    use_profile(f)
    assert cp.current().partitions == []
    assert "not usable" in capsys.readouterr().err


def test_profile_in_the_home_folder_is_found(tmp_path, use_profile, monkeypatch):
    use_profile(None)
    (tmp_path / ".bhaskera").mkdir()
    (tmp_path / ".bhaskera" / "cluster.yaml").write_text("name: Home Cluster\n")
    cp.reset()
    assert cp.current().name == "Home Cluster"
