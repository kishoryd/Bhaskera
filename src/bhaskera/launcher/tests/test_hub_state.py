"""What the dashboard shows: parsing the job log, the MLflow store, the combined state and the endpoints."""
from __future__ import annotations

import http.client
import json
import time

import pytest

from bhaskera.launcher import hub_state as hs
from bhaskera.launcher import ray_hub

LOG_TAIL = (
    "\x1b[36m(RayTrainWorker pid=11, ip=10.0.0.5)\x1b[0m [epoch 0][step 141] loss=1.2000 lr=1.0e-04 grad_norm=0.1 tok/s=17000 MFU=34.0% HFU=50.0% pad=0% step_s=4.80 MFU_step=34.0% mem=16.0/16.3GiB\n"
    "\x1b[36m(RayTrainWorker pid=11, ip=10.0.0.5)\x1b[0m [epoch 0][step 142] loss=1.1180 lr=1.0e-04 grad_norm=0.0983 tok/s=17400 MFU=35.2% HFU=51.0% pad=0% step_s=4.70 MFU_step=35.2% mem=16.0/16.3GiB\n"
    "(RayTrainWorker pid=11, ip=10.0.0.5) [epoch 0][step 100] eval_loss=1.4020 eval_acc=0.6710\n"
    "(PlacementGroupCleaner pid=9, ip=10.0.0.5) Failed to query Ray Train Controller actor state.\n")
LOG_HEAD = "Running command on head node: ['python', '-m', 'bhaskera.launcher.train', '--config', 'recipes/demo.yaml', '--num-workers', '4']\n"


def test_durations_and_cleaning():
    assert hs.parse_duration("1-00:00:00") == 86400 and hs.parse_duration("12:30:05") == 45005 and hs.parse_duration("7:44") == 464
    assert hs.parse_duration("UNLIMITED") is None and hs.parse_duration("junk") is None
    assert "\x1b" not in hs.clean(LOG_TAIL) and "RayTrainWorker" not in hs.clean(LOG_TAIL)


def test_newest_training_numbers_come_from_the_log_tail():
    t = hs.parse_training(hs.clean(LOG_TAIL))
    assert (t["step"], t["loss"], t["tok_s"], t["mfu"], t["step_s"]) == (142, 1.118, 17400.0, 35.2, 4.7)
    assert (t["eval_loss"], t["eval_acc"]) == (1.402, 0.671)
    assert hs.parse_training("loading the model ...") == {}


def test_recipe_steps_progress_and_eta(tmp_path):
    assert hs.config_path_from_log(LOG_HEAD) == "recipes/demo.yaml"
    recipe = tmp_path / "r.yaml"
    recipe.write_text("model:\n  name: m\ntraining:\n  lr: 0.1\n  max_steps: 400\n  eval_steps: 100\nlora:\n  max_steps: 7\n")
    assert hs.max_steps_of(recipe) == 400                                    # the one under training:, not another section
    assert hs.max_steps_of(tmp_path / "missing.yaml") is None
    frac, eta = hs.progress_and_eta(142, 400, 4.7)
    assert round(frac, 3) == 0.355 and eta == int(258 * 4.7)
    assert hs.progress_and_eta(None, 400, 4.7) == (None, None) and hs.progress_and_eta(5, None, 4.7) == (None, None)


def test_squeue_rows():
    rows = hs.parse_squeue("130450|qwen|RUNNING|gpu-debug|2|cbgpu[0100-0101]|1:00:00|25:20|None|bhaskera|2026-11-22T10:00:00\n"
                           "130460|big|PENDING|gpu-small|16||12:00:00|0:00|Priority||2026-11-22T10:01:00\nbad|line\n")
    assert [(j["id"], j["state"], j["node_count"], j["limit_s"], j["bhaskera"], j["reason"]) for j in rows] == [
        ("130450", "RUNNING", 2, 3600, True, ""), ("130460", "PENDING", 16, 43200, False, "Priority")]


def _make_run(store, exp, run_id, name, status, start, job="", group="", eval_losses=(), loss=None, ours=True):
    d = store / exp / run_id
    (d / "metrics").mkdir(parents=True)
    (d / "tags").mkdir()
    (d / "meta.yaml").write_text(f"run_name: {name}\nstatus: {status}\nstart_time: {int(start * 1000)}\nend_time: {int((start + 60) * 1000)}\n"
                                 f"lifecycle_stage: active\nexperiment_id: '{exp}'\n")
    if ours:
        (d / "tags" / "framework").write_text("bhaskera")
    if job:
        (d / "tags" / "slurm_job").write_text(job)
    if group:
        (d / "tags" / "group").write_text(group)
    if eval_losses:
        (d / "metrics" / "eval_loss").write_text("".join(f"{i} {v} {i}\n" for i, v in enumerate(eval_losses)))
    if loss is not None:
        (d / "metrics" / "loss").write_text(f"1 {loss} 1\n")


def test_a_run_that_stopped_writing_is_stalled_unless_its_job_is_alive(tmp_path, use_profile):
    import os
    use_profile(None)
    now = time.time()
    store = tmp_path / "store"
    _make_run(store, "1", "dead", "dead-run", 1, now - 7200, job="55", loss=1.0)       # marked running, silent for 2 hours
    _make_run(store, "1", "live", "live-run", 1, now - 60, job="7", loss=1.0)
    _make_run(store, "1", "old", "old-run", 1, now - 7200, job="7", loss=1.0)           # silent, but its job (7) is still running
    for rid in ("dead", "old"):
        os.utime(store / "1" / rid / "metrics" / "loss", (now - 7000, now - 7000))
    b, _ = _builder(tmp_path)
    runs = {r["name"]: r for r in b.build()["runs"]}
    assert runs["dead-run"]["stale"] is True and runs["live-run"]["stale"] is False and runs["old-run"]["stale"] is False


def test_recent_runs_from_the_store_newest_first(tmp_path):
    now = time.time()
    _make_run(tmp_path, "1", "a", "old", 3, now - 5000, job="1")
    _make_run(tmp_path, "1", "b", "new", 4, now - 100, job="2", group="sweep", eval_losses=(2.0, 1.5), loss=1.2)
    (tmp_path / "1" / "c").mkdir()                                      # a folder that is no run
    runs = hs.recent_runs(tmp_path, limit=5, now=now)
    assert [r["name"] for r in runs] == ["new", "old"] or [r["name"] for r in runs] == ["old", "new"]
    new = next(r for r in runs if r["name"] == "new")
    assert (new["status"], new["job"], new["group"], new["eval_loss"], new["loss"]) == ("FAILED", "2", "sweep", 1.5, 1.2)
    assert hs.recent_runs(tmp_path / "none") == []


class _Proc:
    def poll(self):
        return None


def _builder(tmp_path, user="alice", jobs=None, files=None):
    jobs = jobs if jobs is not None else [
        {"id": "7", "name": "demo", "state": "RUNNING", "partition": "gpu", "node_count": 2, "node_list": "n[1-2]",
         "limit_s": 3600, "elapsed_s": 600, "reason": "", "bhaskera": True, "kind": "train", "submitted": ""},
        {"id": "8", "name": "later", "state": "PENDING", "partition": "gpu", "node_count": 4, "node_list": "",
         "limit_s": 3600, "elapsed_s": 0, "reason": "Priority", "bhaskera": True, "kind": "train", "submitted": ""}]
    log = tmp_path / "demo_7.out"
    log.write_text(LOG_HEAD + "x\n" * 5 + LOG_TAIL)
    (tmp_path / "recipes").mkdir(exist_ok=True)
    (tmp_path / "recipes" / "demo.yaml").write_text("training:\n  max_steps: 400\n")
    files = files or {"7": {"user": user, "stdout": str(log), "workdir": str(tmp_path)}}
    hub = ray_hub.Hub(lambda: [{"id": j["id"], "name": j["name"], "nodes": j["node_list"]} for j in jobs if j["state"] == "RUNNING"],
                      lambda n: n, lambda h, r, g: [h],
                      spawn=lambda cmd: _Proc(), pick_port=lambda: 40000)
    hub.refresh()
    nodes = [{"host": "n1", "alive": True, "cpu": 10, "gpus": [{"util": 90}, {"util": 70}]},
             {"host": "n2", "alive": True, "cpu": 10, "gpus": [{"util": 100}, {"util": 80}]}]
    return hs.StateBuilder(hub, str(tmp_path / "store"), user, list_jobs=lambda: jobs, job_files=lambda j: files.get(j, {}),
                           ray_nodes=lambda port: nodes, bad_nodes=lambda: {"bad1": (1000.0, "only 1 of 2 GPUs visible")},
                           partitions=lambda names: [{"partition": n, "running": 1, "submitted": 2, "max_running": 4,
                                                      "max_submitted": 8, "max_hours": 24} for n in names],
                           version="9.9", ttl=0.0), hub


def test_state_combines_slurm_log_ray_and_nodes(tmp_path, use_profile):
    use_profile(None)
    b, hub = _builder(tmp_path)
    s = b.build()
    running = next(j for j in s["jobs"] if j["id"] == "7")
    assert running["linked"] and running["train"]["step"] == 142 and running["train"]["max_steps"] == 400
    assert round(running["train"]["progress"], 3) == 0.355 and running["train"]["eta_s"] == int(258 * 4.7)
    assert [n["host"] for n in running["nodes"]] == ["n1", "n2"]
    queued = next(j for j in s["jobs"] if j["id"] == "8")
    assert not queued["linked"] and "train" not in queued and queued["reason"] == "Priority"
    assert s["totals"] == {"running": 1, "queued": 1, "nodes": 2, "gpus": 4, "gpu_util": 85}      # GPUs counted from Ray's data
    assert s["bad_nodes"][0]["host"] == "bad1" and s["version"] == "9.9" and s["cluster"] == ""


def test_cluster_name_and_gpus_come_from_the_profile(tmp_path, use_profile):
    f = tmp_path / "c.yaml"
    f.write_text("name: Some Cluster\ngpus_per_node: 8\ndocs_url: https://example.org/docs\npartitions:\n  - {max_nodes: 8, name: gpu, max_hours: 24}\n")
    use_profile(f)
    s = _builder(tmp_path)[0].build()
    assert s["cluster"] == "Some Cluster" and s["gpus_per_node"] == 8 and s["totals"]["gpus"] == 16
    assert s["docs_url"] == "https://example.org/docs" and [p["partition"] for p in s["partitions"]] == ["gpu"]


def test_logs_are_only_for_the_users_own_jobs(tmp_path, use_profile):
    use_profile(None)
    b, _ = _builder(tmp_path)
    out = b.log("7", 5)
    assert out["job"] == "7" and len(out["lines"]) == 10 and "[step 142]" in " ".join(out["lines"])      # at least 10 lines, newest last
    assert b.log("../etc/passwd")["error"] == "bad job id" and b.log("")["error"] == "bad job id"
    assert b.log("99")["error"] == "no log for this job"
    other, _ = _builder(tmp_path, user="alice", files={"7": {"user": "mallory", "stdout": str(tmp_path / "demo_7.out")}})
    assert other.log("7")["error"] == "no log for this job"


def _get(port, path, host="localhost"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", path, headers={"Host": host})
    r = c.getresponse()
    return r.status, r.getheader("Content-Type"), r.read().decode()


def test_the_page_and_its_endpoints(tmp_path, use_profile):
    use_profile(None)
    b, hub = _builder(tmp_path)
    hub.state_provider, hub.log_provider = b.build, b.log
    srv = ray_hub.serve(hub, ray_hub.free_port())
    try:
        status, ctype, page = _get(srv.server_port, "/", "localhost:8265")
        assert status == 200 and ctype.startswith("text/html") and "<title>Bhaskera" in page and "api/state" in page
        status, ctype, body = _get(srv.server_port, "/api/state", "localhost:8265")
        assert status == 200 and ctype == "application/json" and json.loads(body)["jobs"][0]["id"] in ("7", "8")
        status, _, body = _get(srv.server_port, "/api/log?job=7&lines=3", "localhost:8265")
        assert status == 200 and len(json.loads(body)["lines"]) == 10
        assert json.loads(_get(srv.server_port, "/api/log?job=abc", "localhost:8265")[2])["error"] == "bad job id"
        assert _get(srv.server_port, "/api/state", "job-7.localhost:8265")[0] in (404, 502)      # not served under a job address
    finally:
        srv.shutdown()


def test_the_page_does_not_name_a_cluster():
    from bhaskera.launcher.hub_ui import INDEX_HTML
    assert "PARAM Rudra" not in INDEX_HTML and "paramrudra" not in INDEX_HTML.lower()


def test_only_runs_made_by_bhaskera_are_listed(tmp_path):
    now = time.time()
    _make_run(tmp_path, "1", "mine", "mine", 3, now - 10)                                         # tagged framework: bhaskera
    _make_run(tmp_path, "2", "legacy", "legacy-run", 3, now - 20, ours=False)                     # an older run in a bhaskera* experiment, from before the tag
    (tmp_path / "2" / "meta.yaml").write_text("name: bhaskera-multimodal\nartifact_location: x\n")
    _make_run(tmp_path, "3", "foreign", "someone-elses", 3, now - 5, ours=False)                  # another tool, in an experiment of its own
    (tmp_path / "3" / "meta.yaml").write_text("name: MLflow Demo\nartifact_location: x\n")
    _make_run(tmp_path, "4", "loose", "no-experiment-file", 3, now - 1, ours=False)               # no tag, an experiment without a name
    runs = hs.recent_runs(tmp_path, limit=10, now=now)
    assert sorted(r["name"] for r in runs) == ["legacy-run", "mine"]
    assert hs.is_bhaskera_run(tmp_path / "3" / "foreign", hs._experiment_names(tmp_path)) is False


def test_energy_is_the_area_under_the_gpu_power_curve(tmp_path, use_profile):
    use_profile(None)
    store = tmp_path / "store"
    _make_run(store, "1", "a", "with-power", 3, time.time() - 600, job="1", loss=1.0)
    _make_run(store, "1", "b", "no-power", 3, time.time() - 600, job="2", loss=1.0)
    # 2000 W for 1800 s, then 1000 W for 1800 s: 1.0 kWh + 0.5 kWh over one hour, so 1.5 kWh at 1.5 kW on average
    t0 = 1_700_000_000_000
    (store / "1" / "a" / "metrics" / "cluster_gpu_power_w_sum").write_text(
        f"{t0} 2000 0\n{t0 + 1_800_000} 2000 100\n{t0 + 1_800_000} 1000 100\n{t0 + 3_600_000} 1000 200\n")
    runs = {r["name"]: r for r in hs.recent_runs(store)}
    assert runs["with-power"]["energy_kwh"] == 1.5 and runs["with-power"]["avg_kw"] == 1.5
    assert runs["no-power"]["energy_kwh"] is None and runs["no-power"]["avg_kw"] is None


def test_one_power_reading_is_not_enough_for_an_energy_figure(tmp_path):
    f = tmp_path / "p"
    f.write_text("1700000000000 2000 0\n")
    assert hs._energy_kwh(f) is None
    assert hs._energy_kwh(tmp_path / "missing") is None
