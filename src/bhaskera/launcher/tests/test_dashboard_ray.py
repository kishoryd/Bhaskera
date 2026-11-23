"""bhaskera-dashboard ray: command lines, ports and job selection. No Slurm or network needed."""
from __future__ import annotations

import pytest

from bhaskera.launcher import dashboard


@pytest.fixture(autouse=True)
def _no_site_settings(monkeypatch, tmp_path):
    """These tests must not depend on the shell they run in (the module sets BHASKERA_LOGIN_*) or on settings saved in ~/.bhaskera."""
    for var in ("BHASKERA_LOGIN_HOST", "BHASKERA_LOGIN_PORT", "BHASKERA_MODULE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(dashboard, "_CONFIG_DIR", tmp_path / ".bhaskera")
    monkeypatch.setattr(dashboard, "_CONFIG_PATH", tmp_path / ".bhaskera" / "mlflow-ui.json")


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


def test_head_node(monkeypatch):
    monkeypatch.setattr(dashboard, "_slurm", lambda cmd: "cbgpu0241\ncbgpu0242\n")
    assert dashboard.head_node("cbgpu[0241-0242]") == "cbgpu0241"
    monkeypatch.setattr(dashboard, "_slurm", lambda cmd: "")
    assert dashboard.head_node("cbgpu[0241-0245]") == "cbgpu0241"
    assert dashboard.head_node("cbgpu[0100,0105-0107]") == "cbgpu0100"
    assert dashboard.head_node("cbgpu0300") == "cbgpu0300"


def test_ports_are_per_user_and_stable(monkeypatch):
    monkeypatch.setattr(dashboard.os, "getuid", lambda: 80029)
    assert dashboard.hub_port() == 20029
    monkeypatch.setattr(dashboard.os, "getuid", lambda: 80030)
    assert dashboard.hub_port() == 20030


def test_laptop_command_is_fixed_and_contains_no_node(monkeypatch):
    monkeypatch.setattr(dashboard.os, "getuid", lambda: 80029)
    monkeypatch.setenv("BHASKERA_MODULE", "bhaskera/1.1.0")
    cmd = dashboard.laptop_command(_cfg())
    assert cmd == ('ssh -p 4422 -L 8265:localhost:20029 dlapp@paramrudra.cdacb.in '
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
    assert "bhaskera-dashboard ray" in out and "http://localhost:8265" in out and "localhost:3000" not in out


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
    assert "LocalForward 8265 localhost:" in block and block.count("LocalForward") == 1
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


def test_ssh_config_block_has_one_forward(monkeypatch):
    monkeypatch.setattr(dashboard.os, "getuid", lambda: 80029)
    assert dashboard.ssh_config_block(_cfg()).count("LocalForward") == 1


def test_hub_jobs_are_the_running_bhaskera_jobs_or_the_one_asked_for():
    jobs = [_job("1", comment=""), _job("2"), _job("3", state="PENDING", nodes=""), _job("4", comment="other"),
            _job("5")]
    assert [j["id"] for j in dashboard.hub_jobs(jobs)] == ["2", "5"]
    assert [j["id"] for j in dashboard.hub_jobs(jobs, "4")] == ["4"]       # any running job, by id
    assert dashboard.hub_jobs(jobs, "3") == []                              # queued: nothing to link yet


def test_mlflow_page_state():
    assert dashboard.mlflow_page_state(1, fetch=lambda p: (200, b"<html>MLflow</html>")) is True
    assert dashboard.mlflow_page_state(1, fetch=lambda p: (200, b"Unable to display MLflow UI - landing page (index.html) not found.")) is False
    assert dashboard.mlflow_page_state(1, fetch=lambda p: (500, b"error")) is False
    assert dashboard.mlflow_page_state(1, fetch=lambda p: (_ for _ in ()).throw(OSError("refused"))) is None


def test_a_broken_mlflow_on_the_port_is_left_alone_and_a_new_one_starts_elsewhere(monkeypatch, tmp_path, capsys):
    started = []
    monkeypatch.setattr(dashboard, "_read_pid", lambda: None)
    monkeypatch.setattr(dashboard, "close_orphan_runs", lambda cfg: 0)
    monkeypatch.setattr(dashboard, "mlflow_page_state", lambda port, fetch=None: False)
    monkeypatch.setattr(dashboard.ray_hub, "free_port", lambda: 45678)
    monkeypatch.setattr(dashboard, "cmd_start", lambda cfg: started.append((cfg.port, cfg.host)))
    cfg = _cfg(login_node="x", user="u")
    cfg.port = 5055
    dashboard.ensure_mlflow(cfg)
    assert started == [(45678, "127.0.0.1")] and "does not work" in capsys.readouterr().out
    assert dashboard.mlflow_port(cfg) == 45678, "the hub follows the new port"


def test_a_working_mlflow_on_the_port_is_reused(monkeypatch, capsys):
    started = []
    monkeypatch.setattr(dashboard, "_read_pid", lambda: None)
    monkeypatch.setattr(dashboard, "close_orphan_runs", lambda cfg: 0)
    monkeypatch.setattr(dashboard, "mlflow_page_state", lambda port, fetch=None: True)
    monkeypatch.setattr(dashboard, "cmd_start", lambda cfg: started.append(cfg.port))
    cfg = _cfg(login_node="x", user="u")
    cfg.port = 5055
    dashboard.ensure_mlflow(cfg)
    assert started == [5055] and "does not work" not in capsys.readouterr().out
