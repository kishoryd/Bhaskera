"""Node check, failure diagnosis and the list of faulty nodes. No cluster needed."""
from __future__ import annotations

import json

from bhaskera.launcher import bad_nodes, diagnose, nodecheck


def test_bad_nodes_expire_and_can_be_forgotten(tmp_path):
    path = tmp_path / "bad-nodes"
    import time
    now = time.time()
    bad_nodes.add("cbgpu0264", "only 1 of 2 GPUs visible", path, now=now)
    bad_nodes.add("cbgpu0001", "old fault", path, now=now - 5 * 86400)
    assert list(bad_nodes.load(path)) == ["cbgpu0264"]                             # the 5-day-old entry is ignored
    assert bad_nodes.exclude_list(path, extra="cbgpu0999") == "cbgpu0999,cbgpu0264"
    assert bad_nodes.forget("cbgpu0264", path) is True
    assert bad_nodes.load(path) == {} and bad_nodes.forget("cbgpu0264", path) is False


def test_missing_list_file_is_fine(tmp_path):
    assert bad_nodes.load(tmp_path / "none") == {}
    assert bad_nodes.exclude_list(tmp_path / "none", extra="a,b") == "a,b"


def _line(host, ok=True, problems=(), ip="1.2.3.4"):
    return nodecheck.PREFIX + json.dumps({"host": host, "ok": ok, "problems": list(problems), "ib_ip": ip})


def test_summary_names_the_faulty_node_and_remembers_it(tmp_path, monkeypatch):
    monkeypatch.setenv("BHASKERA_BAD_NODES", str(tmp_path / "bad"))
    text = "\n".join(["srun noise", _line("cbgpu0263"), _line("cbgpu0264", False, ["only 1 of 2 GPUs visible"])])
    table, bad = nodecheck.summarize(nodecheck.parse_lines(text), expected_nodes=2)
    assert bad == ["cbgpu0264"]
    assert "cbgpu0264" in table and "only 1 of 2 GPUs visible" in table and "remembered" in table
    assert "cbgpu0264" in bad_nodes.load(tmp_path / "bad")


def test_a_node_that_gave_no_answer_is_reported(tmp_path, monkeypatch):
    table, bad = nodecheck.summarize(nodecheck.parse_lines(_line("n1")), expected_nodes=3, remember=False)
    assert bad == [] and "2 node(s) gave no answer" in table


def test_all_healthy():
    table, bad = nodecheck.summarize(nodecheck.parse_lines(_line("a") + "\n" + _line("b")), expected_nodes=2, remember=False)
    assert bad == [] and "all 2 nodes are healthy" in table


def test_check_node_on_a_machine_without_gpus_checks_host_and_network():
    result = nodecheck.check_node(gpus=0, min_host_gib=0, min_tmp_gib=0, interface="")
    assert result["ok"] is True and result["host"]


LOG = """\
(RayTrainWorker pid=11, ip=172.41.11.50) RuntimeError: CUDA error: invalid device ordinal
(RayTrainWorker pid=12, ip=172.41.11.51) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB
srun: error: cbgpu0262: task 0: Exited with exit code 1
slurmstepd: error: Detected 1 oom_kill event in StepId=1.0. Some of the step tasks have been OOM Killed.
"""


def test_diagnosis_names_cause_and_node():
    findings = diagnose.diagnose(LOG, {"172.41.11.50": "cbgpu0264", "172.41.11.51": "cbgpu0265"})
    by_kind = {f.kind: f for f in findings}
    assert by_kind["gpu-missing"].host == "cbgpu0264" and by_kind["gpu-missing"].hardware
    assert by_kind["gpu-oom"].host == "cbgpu0265" and not by_kind["gpu-oom"].hardware
    assert "host-oom" in by_kind
    text = diagnose.report(findings)
    assert "a GPU of the node is missing or not usable on cbgpu0264" in text and "Lower training.batch_size" in text


def test_unknown_failure_gets_an_honest_message():
    assert "No known cause" in diagnose.report(diagnose.diagnose("everything looked fine\n"))


def test_only_hardware_faults_go_on_the_list(tmp_path, monkeypatch, capsys):
    log = tmp_path / "job.out"
    log.write_text(LOG)
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"172.41.11.50": "cbgpu0264", "172.41.11.51": "cbgpu0265"}))
    store = tmp_path / "bad"
    monkeypatch.setenv("BHASKERA_BAD_NODES", str(store))
    monkeypatch.setattr("sys.argv", ["bhaskera-diagnose", str(log), "--map", str(mapping)])
    diagnose.main()
    assert "why the job failed" in capsys.readouterr().out
    listed = bad_nodes.load(store)
    assert "cbgpu0264" in listed and "cbgpu0265" not in listed
