"""bhaskera-dashboard ray: find the Bhaskera job (also while queued), wait for nodes, link its dashboard.
No Slurm or network needed."""
from __future__ import annotations

import pytest

from bhaskera.launcher import dashboard


def _cfg(login_node="paramrudra.cdacb.in", user="dlapp", login_port=4422):
    c = dashboard.DashboardConfig()
    c.login_node, c.user, c.login_port = login_node, user, login_port
    return c


def _job(id_, state="RUNNING", nodes="cbgpu[0241-0245]", comment="bhaskera", name="train", reason="None"):
    return {"id": id_, "name": name, "state": state, "nodes": nodes, "reason": reason, "comment": comment}


def test_my_jobs_parses_squeue(monkeypatch):
    monkeypatch.setattr(dashboard, "_slurm", lambda cmd: "128783|qwen|RUNNING|cbgpu[0241-0245]|None|bhaskera\n"
                                                         "128790|other|PENDING||Priority|\n |bad\n")
    jobs = dashboard.my_jobs()
    assert [(j["id"], j["state"], j["comment"]) for j in jobs] == [("128783", "RUNNING", "bhaskera"),
                                                                    ("128790", "PENDING", "")]
    assert jobs[1]["reason"] == "Priority"


def test_only_bhaskera_jobs_are_considered_without_job_flag():
    jobs = [_job("1", comment=""), _job("2", comment="bhaskera"), _job("3", comment="other")]
    chosen, note = dashboard.choose_job(jobs, None)
    assert chosen["id"] == "2" and note == ""


def test_running_job_beats_a_queued_one_and_newest_wins():
    jobs = [_job("5", "PENDING", nodes=""), _job("7"), _job("9")]
    chosen, note = dashboard.choose_job(jobs, None)
    assert chosen["id"] == "9" and "7" in note and "5" in note
    only_queued, _ = dashboard.choose_job([_job("5", "PENDING", nodes="")], None)
    assert only_queued["state"] == "PENDING"


def test_job_flag_selects_any_job_and_reports_unknown_ones():
    jobs = [_job("1", comment=""), _job("2")]
    assert dashboard.choose_job(jobs, "1")[0]["id"] == "1"
    chosen, msg = dashboard.choose_job(jobs, "99")
    assert chosen is None and "99" in msg
    none, msg = dashboard.choose_job([_job("1", comment="")], None)
    assert none is None and "no Bhaskera job" in msg


def test_head_node(monkeypatch):
    monkeypatch.setattr(dashboard, "_slurm", lambda cmd: "cbgpu0241\ncbgpu0242\n")
    assert dashboard.head_node("cbgpu[0241-0242]") == "cbgpu0241"
    monkeypatch.setattr(dashboard, "_slurm", lambda cmd: "")
    assert dashboard.head_node("cbgpu[0241-0245]") == "cbgpu0241"
    assert dashboard.head_node("cbgpu[0100,0105-0107]") == "cbgpu0100"
    assert dashboard.head_node("cbgpu0300") == "cbgpu0300"


def test_ports_are_per_user_and_stable(monkeypatch):
    monkeypatch.setattr(dashboard.os, "getuid", lambda: 80029)
    assert dashboard.user_ports() == (20029, 30029)
    monkeypatch.setattr(dashboard.os, "getuid", lambda: 80030)
    assert dashboard.user_ports() == (20030, 30030)


def test_laptop_command_is_fixed_and_contains_no_node(monkeypatch):
    monkeypatch.setattr(dashboard.os, "getuid", lambda: 80029)
    monkeypatch.setenv("BHASKERA_MODULE", "bhaskera/1.1.0")
    cmd = dashboard.laptop_command(_cfg())
    assert cmd == ('ssh -p 4422 -L 8265:localhost:20029 -L 3000:localhost:30029 -L 5055:localhost:40029 dlapp@paramrudra.cdacb.in '
                   '"bash -lc \'module load bhaskera/1.1.0 && bhaskera-dashboard ray\'"')
    assert "cbgpu" not in cmd


def test_forward_command_links_both_ports_to_the_head():
    cmd = dashboard.forward_command("cbgpu0241", 20029, 30029)
    assert cmd[0] == "ssh" and cmd[-1] == "cbgpu0241"
    assert "20029:localhost:8265" in cmd and "30029:localhost:3000" in cmd
    assert "ExitOnForwardFailure=yes" in cmd


def test_info_prints_the_command_and_nothing_else(monkeypatch, capsys):
    monkeypatch.setattr(dashboard, "my_jobs", lambda: pytest.fail("must not look for jobs"))
    dashboard.cmd_ray(_cfg(), None, info=True)
    out = capsys.readouterr().out
    assert "bhaskera-dashboard ray" in out and "http://localhost:8265" in out and "http://localhost:5055" in out


class _FakeProc:
    def __init__(self, cmd):
        self.cmd, self.returncode, self._polls = cmd, None, 0

    def poll(self):
        self._polls += 1
        if self._polls > 2:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


def test_waits_for_a_queued_job_then_links_its_head_node(monkeypatch, capsys):
    monkeypatch.setattr(dashboard, "ensure_mlflow", lambda cfg: None)
    states = iter([[_job("7", "PENDING", nodes="", reason="Priority")],
                   [_job("7", "PENDING", nodes="", reason="Priority")],
                   [_job("7")]])
    last = {"jobs": [_job("7")]}

    def jobs():
        try:
            last["jobs"] = next(states)
        except StopIteration:
            pass
        return last["jobs"]

    started = []
    monkeypatch.setattr(dashboard, "my_jobs", jobs)
    monkeypatch.setattr(dashboard, "head_node", lambda nodes: "cbgpu0241")
    def sleep(seconds):
        if started and started[0] and getattr(sleep, "after_link", 0) > 3:
            raise KeyboardInterrupt
        if started:
            sleep.after_link = getattr(sleep, "after_link", 0) + 1

    monkeypatch.setattr(dashboard.time, "sleep", sleep)
    monkeypatch.setattr(dashboard.subprocess, "Popen", lambda cmd: started.append(cmd) or _FakeProc(cmd))
    dashboard.cmd_ray(_cfg(), None)
    out = capsys.readouterr().out
    assert "job 7 (train) is pending (Priority); waiting for nodes" in out
    assert "its Ray head is on cbgpu0241" in out
    assert started and started[0][-1] == "cbgpu0241"


def test_no_job_keeps_waiting_and_mlflow_open(monkeypatch, capsys):
    monkeypatch.setattr(dashboard, "ensure_mlflow", lambda cfg: None)
    monkeypatch.setattr(dashboard, "my_jobs", lambda: [])
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) > 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(dashboard.time, "sleep", sleep)
    dashboard.cmd_ray(_cfg(), None)
    out = capsys.readouterr().out
    assert "waiting for one" in out and len(calls) == 3


def test_close_orphan_runs_marks_runs_of_finished_jobs(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    store = tmp_path / "runs"
    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{store}")
    exp = client.create_experiment("bhaskera")
    gone = client.create_run(exp, tags={"slurm_job": "11"}).info.run_id
    alive = client.create_run(exp, tags={"slurm_job": "22"}).info.run_id
    untagged = client.create_run(exp).info.run_id
    monkeypatch.setattr(dashboard, "_active_job_ids", lambda: {"22"})
    cfg = dashboard.DashboardConfig()
    cfg.store = str(store)
    assert dashboard.close_orphan_runs(cfg) == 1
    assert client.get_run(gone).info.status == "KILLED"
    assert client.get_run(alive).info.status == "RUNNING"
    assert client.get_run(untagged).info.status == "RUNNING"
    monkeypatch.setattr(dashboard, "_active_job_ids", lambda: None)   # Slurm unavailable: change nothing
    assert dashboard.close_orphan_runs(cfg) == 0


def test_ssh_target_and_options(monkeypatch):
    monkeypatch.setenv("USER", "bob")
    monkeypatch.setenv("BHASKERA_LOGIN_HOST", "paramrudra")
    monkeypatch.delenv("BHASKERA_LOGIN_PORT", raising=False)
    assert dashboard.ssh_target(_cfg(login_node="", user="")) == "bob@paramrudra"
    assert dashboard.ssh_target(_cfg(login_node="gateway", user="alice")) == "alice@gateway"
    assert dashboard.ssh_options(_cfg(login_port=0)) == ""
    assert dashboard.ssh_options(_cfg(login_port=4422)) == "-p 4422 "
    monkeypatch.setenv("BHASKERA_LOGIN_PORT", "4422")
    assert dashboard.ssh_options(_cfg(login_port=0)) == "-p 4422 "
    monkeypatch.setenv("BHASKERA_LOGIN_PORT", "not-a-number")
    assert dashboard.ssh_options(_cfg(login_port=0)) == ""


def test_mlflow_tunnel_uses_the_same_target(capsys):
    cfg = _cfg(login_node="paramrudra", user="dlapp", login_port=0)
    cfg.port = 5055
    dashboard._print_tunnel(cfg)
    assert "ssh -L 5055:localhost:5055 dlapp@paramrudra" in capsys.readouterr().out
    cfg.login_port = 4422
    dashboard._print_tunnel(cfg)
    assert "ssh -p 4422 -L 5055:localhost:5055 dlapp@paramrudra" in capsys.readouterr().out


def test_ssh_config_block(monkeypatch):
    monkeypatch.setenv("BHASKERA_LOGIN_HOST", "login.example.org")
    monkeypatch.setenv("BHASKERA_LOGIN_PORT", "4422")
    monkeypatch.setenv("BHASKERA_MODULE", "bhaskera/1.1.0")
    monkeypatch.setenv("USER", "alice")
    block = dashboard.ssh_config_block(dashboard.DashboardConfig())
    assert "Host bhaskera-ray" in block
    assert "HostName login.example.org" in block and "User alice" in block and "Port 4422" in block
    assert "LocalForward 8265 localhost:" in block and "LocalForward 3000 localhost:" in block
    assert "module load bhaskera/1.1.0" in block and "ServerAliveInterval 30" in block


def test_remote_command_without_module_uses_tool_path(monkeypatch):
    monkeypatch.delenv("BHASKERA_MODULE", raising=False)
    monkeypatch.setattr(dashboard.shutil, "which", lambda name: "/home/u/venv/bin/bhaskera-dashboard")
    assert dashboard._remote_command() == "bash -lc '/home/u/venv/bin/bhaskera-dashboard ray'"


def test_mlflow_port_is_per_user_unless_chosen(monkeypatch):
    monkeypatch.setattr(dashboard.os, "getuid", lambda: 80029)
    cfg = dashboard.DashboardConfig()
    assert dashboard.mlflow_port(cfg) == 40029
    cfg.port = 5055
    assert dashboard.mlflow_port(cfg) == 5055


def test_ensure_mlflow_starts_it_once(monkeypatch, capsys):
    monkeypatch.setattr(dashboard.os, "getuid", lambda: 80029)
    started = []
    monkeypatch.setattr(dashboard, "_read_pid", lambda: None)
    monkeypatch.setattr(dashboard, "cmd_start", lambda c: started.append(c.port))
    dashboard.ensure_mlflow(dashboard.DashboardConfig())
    assert started == [40029]
    monkeypatch.setattr(dashboard, "_read_pid", lambda: 123)
    monkeypatch.setattr(dashboard, "_pid_alive", lambda pid: True)
    dashboard.ensure_mlflow(dashboard.DashboardConfig())
    assert started == [40029] and "MLflow is running" in capsys.readouterr().out


def test_ssh_config_block_forwards_mlflow(monkeypatch):
    monkeypatch.setattr(dashboard.os, "getuid", lambda: 80029)
    assert "LocalForward 5055 localhost:40029" in dashboard.ssh_config_block(_cfg())
