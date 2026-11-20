"""Per-node telemetry gathered to rank 0: selection, aggregation, per-node cut-off, the gather. CPU only."""
from __future__ import annotations

import pytest

from bhaskera.utils import node_stats


def _report(host, util=(90.0, 80.0), mem_mib=(18000.0, 18100.0), temp=(55.0, 60.0), power=(250.0, 260.0), cpu=7.5):
    stats = {"sys/cpu_pct": cpu, "sys/mem_used_pct": 12.0, "sys/loadavg_1m": 1.5}
    for i in range(len(util)):
        stats[f"gpu/{i}/util_pct"] = util[i]
        stats[f"gpu/{i}/mem_used_mib"] = mem_mib[i]
        stats[f"gpu/{i}/temp_c"] = temp[i]
        stats[f"gpu/{i}/power_w"] = power[i]
    return {"host": host, "stats": stats}


def test_cluster_aggregates_cover_every_gpu_of_every_node():
    reports = [_report("n1", util=(100.0, 50.0)), _report("n2", util=(80.0, 70.0), temp=(70.0, 40.0))]
    m = node_stats.flatten(reports)
    assert m["cluster/nodes"] == 2 and m["cluster/gpus"] == 4
    assert m["cluster/gpu_util_pct_min"] == 50.0 and m["cluster/gpu_util_pct_max"] == 100.0
    assert m["cluster/gpu_util_pct_mean"] == pytest.approx((100 + 50 + 80 + 70) / 4)
    assert m["cluster/gpu_temp_c_max"] == 70.0
    assert m["cluster/gpu_power_w_sum"] == pytest.approx(4 * 255.0)
    assert m["cluster/gpu_mem_used_gib_max"] == pytest.approx(18100.0 / 1024.0)


def test_per_node_curves_for_small_jobs():
    m = node_stats.flatten([_report("cbgpu0001"), _report("cbgpu0002", util=(10.0, 20.0))])
    assert m["node/cbgpu0001/gpu0/util_pct"] == 90.0
    assert m["node/cbgpu0002/gpu1/util_pct"] == 20.0
    assert m["node/cbgpu0002/gpu1/mem_used_gib"] == pytest.approx(18100.0 / 1024.0)
    assert m["node/cbgpu0001/cpu_pct"] == 7.5 and "node/cbgpu0001/mem_used_pct" in m


def test_no_per_node_curves_beyond_the_limit_but_aggregates_stay():
    reports = [_report(f"n{i}") for i in range(20)]
    m = node_stats.flatten(reports, per_node_max=16)
    assert m["cluster/nodes"] == 20 and m["cluster/gpus"] == 40
    assert not any(k.startswith("node/") for k in m)
    assert any(k.startswith("node/") for k in node_stats.flatten(reports, per_node_max=20))


def test_local_report_keeps_only_the_selected_readings(monkeypatch):
    raw = {"gpu/count": 2.0, "gpu/0/util_pct": 55.0, "gpu/0/fan_pct": 30.0, "gpu/0/temp_c": 61.0,
           "gpu/0/mem_used_mib": 1000.0, "gpu/0/power_w": 200.0, "proc/cpu_pct": 3.0,
           "sys/cpu_pct": 9.0, "sys/mem_used_pct": 20.0, "sys/loadavg_5m": 2.0}
    monkeypatch.setattr(node_stats, "system_stats", lambda gpu=True, cpu=True: raw)
    r = node_stats.local_report()
    assert set(r["stats"]) == {"gpu/0/util_pct", "gpu/0/temp_c", "gpu/0/mem_used_mib", "gpu/0/power_w",
                               "sys/cpu_pct", "sys/mem_used_pct"}


def test_only_the_first_rank_of_a_node_reports_without_a_process_group(monkeypatch):
    monkeypatch.setattr(node_stats, "system_stats", lambda gpu=True, cpu=True: {"gpu/0/util_pct": 1.0})
    assert node_stats.gather_node_reports(local_rank=1) is None
    got = node_stats.gather_node_reports(local_rank=0)
    assert got is not None and len(got) == 1 and "gpu/0/util_pct" in got[0]["stats"]


def test_gather_through_a_real_process_group(tmp_path, monkeypatch):
    import torch.distributed as dist
    monkeypatch.setattr(node_stats, "system_stats", lambda gpu=True, cpu=True: {"gpu/0/util_pct": 42.0})
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    try:
        got = node_stats.gather_node_reports(local_rank=0)
        assert got is not None and got[0]["stats"]["gpu/0/util_pct"] == 42.0
    finally:
        dist.destroy_process_group()


def test_summary_line_matches_the_step_log_format():
    m = node_stats.flatten([_report("n1"), _report("n2")])
    line = node_stats.summary_line(30, m)
    assert line.startswith("[step 30] nodes=2 gpus=4 gpu_util=")
    assert "(min/mean/max)" in line and "gpu_mem_max=" in line and line.endswith("C")
