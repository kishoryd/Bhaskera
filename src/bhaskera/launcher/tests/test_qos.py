"""Partition limits (QoS) and how much room a user has. No Slurm needed."""
from __future__ import annotations

import pytest

from bhaskera.launcher import qos

SACCTMGR = """normal|||||0|
gpu-debug|01:00:00|2|4
gpu-small|1-00:00:00|4|8
gpu-large|12:00:00|2|4
gpu-massive|06:00:00|2|2
"""
SCONTROL = "PartitionName=gpu-debug AllowQos=ALL QoS=gpu-debug DefaultTime=00:30:00 MaxTime=01:00:00"


def test_partition_follows_the_node_count(param_rudra):
    assert qos.partition_for(1) == ("gpu-debug", 1) and qos.partition_for(4) == ("gpu-debug", 1)
    assert qos.partition_for(5) == ("gpu-small", 24) and qos.partition_for(26) == ("gpu-large", 12)
    assert qos.partition_for(128) == ("gpu-massive", 6)
    with pytest.raises(ValueError):
        qos.partition_for(201)


def test_slurm_times_and_limits_are_read():
    assert qos.parse_hours("1-00:00:00") == 24 and qos.parse_hours("06:00:00") == 6 and qos.parse_hours("") is None
    assert qos.parse_partition_qos(SCONTROL) == "gpu-debug" and qos.parse_partition_qos("QoS=N/A") is None
    lim = qos.parse_qos_limits(SACCTMGR, "gpu-debug")
    assert (lim.max_hours, lim.max_running, lim.max_submitted) == (1, 2, 4)
    assert qos.parse_qos_limits(SACCTMGR, "missing").max_submitted is None


def test_room_counts_down_and_unknown_means_unlimited():
    lim = qos.QosLimits("gpu-debug", 1, 2, 4)
    assert qos.room(lim, 1, 3) == 1 and qos.room(lim, 2, 4) == 0 and qos.room(qos.QosLimits("x"), 0, 99) is None


def _patch(monkeypatch, running, submitted):
    monkeypatch.setattr(qos, "limits_for", lambda part: qos.parse_qos_limits(SACCTMGR, part))
    monkeypatch.setattr(qos, "user_jobs", lambda part: (running, submitted))


def test_gpu_debug_job_of_one_hour_is_fine(monkeypatch, param_rudra):
    _patch(monkeypatch, 0, 1)
    code, msg = qos.check(2, 1)
    assert code == 0 and "gpu-debug" in msg and "1 h per job" in msg and "2 running" in msg and "4 submitted" in msg


def test_gpu_debug_refuses_more_than_one_hour(monkeypatch, param_rudra):
    _patch(monkeypatch, 0, 0)
    code, msg = qos.check(2, 3)
    assert code == 1 and "at most 1 h" in msg


def test_a_full_queue_is_explained(monkeypatch, param_rudra):
    _patch(monkeypatch, 2, 4)
    code, msg = qos.check(2, 1)
    assert code == 1 and "no room" in msg and "4 jobs submitted" in msg


def test_a_job_that_will_wait_is_a_note_not_an_error(monkeypatch, param_rudra):
    _patch(monkeypatch, 2, 3)
    code, msg = qos.check(2, 1)
    assert code == 0 and "will wait in the queue" in msg


def test_default_hours_are_the_partition_maximum(monkeypatch, param_rudra):
    _patch(monkeypatch, 0, 0)
    assert qos.check(8, None)[0] == 0 and qos.check(8, 30)[0] == 1


def test_without_a_partition_table_nothing_is_checked(use_profile):
    use_profile(None)
    assert qos.check(8, 5) == (0, "")
    with pytest.raises(qos.NoPartitionTable):
        qos.partition_for(8)


def test_another_cluster_with_other_names_and_sizes(tmp_path, use_profile, monkeypatch):
    f = tmp_path / "c.yaml"
    f.write_text("name: X\npartitions:\n  - {max_nodes: 2, name: quick, max_hours: 2}\n  - {max_nodes: 16, name: main, max_hours: 72}\n")
    use_profile(f)
    assert qos.partition_for(2) == ("quick", 2) and qos.partition_for(9) == ("main", 72)
    monkeypatch.setattr(qos, "limits_for", lambda part: qos.QosLimits(part, 72, 3, 6))
    monkeypatch.setattr(qos, "user_jobs", lambda part: (0, 6))
    code, msg = qos.check(9, 100)
    assert code == 1 and "main (QoS main)" in msg and "at most 72 h" in msg and "no room" in msg
