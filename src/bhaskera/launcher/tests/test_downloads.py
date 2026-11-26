"""Downloads from Hugging Face and signing in: names and folders are checked, the key is kept private, and a finished download is remembered."""
from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from bhaskera import dataversions as dv
from bhaskera import downloads as dl
from bhaskera.launcher import cluster_profile, launch

KEY = "hf_" + "a1B2c3D4e5" * 4


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    run = tmp_path / "runs"
    run.mkdir()
    (run / "submit.sh").write_text("")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for var in ("MLFLOW_TRACKING_URI", "HF_HOME", "HF_DATASETS_CACHE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BHASKERA_WORKDIR", str(run))
    return run, home


def info(kind, repo, rev, tok, size=1000, gated=False):
    sib = [NS(rfilename="config.json", size=100), NS(rfilename="model.safetensors", size=size - 100)]
    return NS(siblings=sib, gated=gated, private=False, card_data=NS(to_dict=lambda: {"license": "apache-2.0"}), sha="abcdef0123456789")


class Proc:
    """A stand-in for the download process: finishes with the given exit code after writing the given files."""
    def __init__(self, target, code=0, files=(), lines=()):
        self.target, self.code, self.pid, self.stdout = target, code, 0, iter(lines)
        self.returncode = None
        for name, size in files:
            Path(target, name).write_bytes(b"x" * size)

    def poll(self):
        self.returncode = self.code
        return self.code


def wait_for(mgr, states, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if mgr.status()["state"] in states:
            return mgr.status()
        time.sleep(0.05)
    raise AssertionError(f"still {mgr.status()}")


def test_repository_names_and_revisions_are_checked():
    assert dl.check_repo("model", "Qwen/Qwen3-14B", None) == ("model", "Qwen/Qwen3-14B", "main")
    assert dl.check_repo("dataset", "gpt2", "refs/pr/3")[2] == "refs/pr/3"
    for bad in ("", "https://huggingface.co/x/y", "a/b/c", "../etc", "x y/z", "a" * 200, None):
        with pytest.raises(dl.DownloadError):
            dl.check_repo("model", bad, "main")
    with pytest.raises(dl.DownloadError):
        dl.check_repo("space", "a/b", "main")
    with pytest.raises(dl.DownloadError):
        dl.check_repo("model", "a/b", "main; rm -rf /")


def test_the_target_is_a_full_writable_path_and_never_a_system_folder(world, tmp_path):
    assert dl.check_target(str(tmp_path / "models" / "qwen")) == (tmp_path / "models" / "qwen").resolve()
    for bad in ("", "relative/dir", "/etc/x", "/usr/local/models", "/proc", "/var/tmp/x"):
        with pytest.raises(dl.DownloadError):
            dl.check_target(bad)
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(dl.DownloadError, match="is a file"):
        dl.check_target(str(f))
    link = tmp_path / "sneaky"
    link.symlink_to("/etc")
    with pytest.raises(dl.DownloadError, match="system folder"):
        dl.check_target(str(link / "x"))


def test_signing_in_checks_the_key_keeps_it_private_and_never_returns_it(world):
    run, home = world
    seen = []
    ok = lambda k: (seen.append(k), {"name": "kishoryd"})[1]
    for bad in ("", "abc", "hf_short", KEY + " extra", None):
        with pytest.raises(dl.DownloadError, match="does not look like"):
            dl.sign_in(bad, whoami=ok)
    status = dl.sign_in(f"  {KEY}\n", whoami=ok)
    assert status == {"signed_in": True, "user": "kishoryd", "source": "saved"} and KEY not in json.dumps(status)
    f = dl.token_file()
    assert f.read_text() == KEY and stat.S_IMODE(f.stat().st_mode) == 0o600
    assert dl.token() == KEY
    assert dl.auth_status(whoami=ok)["user"] == "kishoryd"
    out = dl.sign_out()
    assert out["signed_in"] is False and out["removed"] is True and not f.exists() and dl.token() is None


def test_a_refused_key_is_not_kept_and_the_error_does_not_contain_it(world):
    def refuse(k):
        raise RuntimeError(f"401 Client Error: Invalid user token {k}")
    with pytest.raises(dl.DownloadError) as e:
        dl.sign_in(KEY, whoami=refuse)
    assert KEY not in str(e.value) and "did not accept" in str(e.value)
    assert not dl.token_file().exists()
    st = dl.auth_status(whoami=refuse)
    assert st["signed_in"] is False


def test_the_environment_key_wins_and_is_reported_as_such(world, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", KEY)
    st = dl.auth_status(whoami=lambda k: {"name": "me"})
    assert st["signed_in"] and st["source"] == "environment"
    assert "still used" in dl.sign_out()["note"]


def test_the_preview_reports_size_licence_and_access_without_downloading(world):
    p = dl.preview("model", "a/b", None, info_fn=lambda *a: info(*a, size=5 << 30, gated=True))
    assert p["bytes"] == 5 << 30 and p["files"] == 2 and p["gated"] is True and p["license"] == "apache-2.0" and p["large"] is False
    assert p["largest"][0]["name"] == "model.safetensors" and p["sha"] == "abcdef012345"
    big = dl.preview("model", "a/b", None, info_fn=lambda *a: info(*a, size=80 << 30))
    assert big["large"] is True

    class GatedRepoError(Exception):
        pass

    def gated(*a):
        raise GatedRepoError("403")
    with pytest.raises(dl.DownloadError, match="needs access"):
        dl.preview("model", "a/b", None, info_fn=gated)
    with pytest.raises(dl.DownloadError, match="no internet"):
        dl.preview("model", "a/b", None, info_fn=lambda *a: (_ for _ in ()).throw(ConnectionError("Name resolution failed")))


def test_a_finished_model_download_is_remembered_and_offered(world, tmp_path):
    target = tmp_path / "models" / "b"
    procs = []

    def popen(cmd, **kw):
        assert KEY not in " ".join(cmd) and "--target" in cmd
        assert kw["env"].get("HF_TOKEN") == KEY                               # the key travels in the environment only
        procs.append(Proc(Path(cmd[cmd.index("--target") + 1]), 0, [("config.json", 10), ("w.bin", 90)]))
        return procs[-1]

    dl.token_file().parent.mkdir(parents=True, exist_ok=True)
    dl.token_file().write_text(KEY)
    m = dl.Manager(popen=popen)
    s = m.start_download("model", "a/b", "main", str(target), info_fn=lambda *a: info(*a, size=100))
    assert s["state"] == "running" and s["total"] == 100
    done = wait_for(m, ("done", "failed"))
    assert done["state"] == "done" and done["result"]["path"] == str(target.resolve())
    assert dl.downloaded_models() == [str(target.resolve())]
    assert dl.records()[0]["repo"] == "a/b" and dl.records()[0]["bytes"] == 100


def test_a_finished_dataset_download_becomes_dataset_version_one_with_its_source(world, tmp_path):
    target = tmp_path / "data" / "ds"
    m = dl.Manager(popen=lambda cmd, **kw: Proc(Path(cmd[cmd.index("--target") + 1]), 0, [("train.jsonl", 40)]))
    m.start_download("dataset", "owner/My Data".replace(" ", "-"), "main", str(target), info_fn=lambda *a: info(*a, size=40))
    done = wait_for(m, ("done", "failed"))
    assert done["state"] == "done" and done["result"]["dataset"] == "My-Data" and done["result"]["version"] == 1
    v = dv.get("My-Data")
    assert v["source"]["type"] == "huggingface" and v["source"]["repo"] == "owner/My-Data" and v["path"] == str(target.resolve())


def test_a_failed_download_says_why_and_a_model_without_config_is_refused(world, tmp_path):
    m = dl.Manager(popen=lambda cmd, **kw: Proc(Path(cmd[cmd.index("--target") + 1]), 1, lines=["error: 401 Unauthorized", "x"]))
    m.start_download("model", "a/b", "main", str(tmp_path / "t1"), info_fn=lambda *a: info(*a, size=10))
    s = wait_for(m, ("done", "failed"))
    assert s["state"] == "failed" and "401" in s["message"]
    m2 = dl.Manager(popen=lambda cmd, **kw: Proc(Path(cmd[cmd.index("--target") + 1]), 0, [("w.bin", 10)]))
    m2.start_download("model", "a/b", "main", str(tmp_path / "t2"), info_fn=lambda *a: info(*a, size=10))
    s2 = wait_for(m2, ("done", "failed"))
    assert s2["state"] == "failed" and "config.json" in s2["message"]
    assert dl.downloaded_models() == []


def test_large_downloads_need_confirmation_and_space_is_checked(world, tmp_path, monkeypatch):
    m = dl.Manager(popen=lambda cmd, **kw: Proc(Path(cmd[cmd.index("--target") + 1]), 0, [("config.json", 1)]))
    with pytest.raises(dl.DownloadError, match="confirm"):
        m.start_download("model", "a/b", "main", str(tmp_path / "x"), info_fn=lambda *a: info(*a, size=80 << 30))
    monkeypatch.setattr("bhaskera.diskguard.shutil.disk_usage", lambda p: NS(free=1 << 30))
    with pytest.raises(dl.DownloadError, match="not enough space"):
        m.start_download("model", "a/b", "main", str(tmp_path / "y"), confirm_large=True, info_fn=lambda *a: info(*a, size=80 << 30))


def test_only_one_task_runs_at_a_time_and_a_download_can_be_cancelled(world, tmp_path, monkeypatch):
    killed = []

    class Slow(Proc):
        def poll(self):
            return None if not killed else 1

    m = dl.Manager(popen=lambda cmd, **kw: Slow(Path(cmd[cmd.index("--target") + 1]), 0))
    monkeypatch.setattr(dl.os, "killpg", lambda pid, sig: killed.append(sig))
    m.start_download("model", "a/b", "main", str(tmp_path / "s"), info_fn=lambda *a: info(*a, size=10))
    with pytest.raises(dl.DownloadError, match="still running"):
        m.start_download("model", "c/d", "main", str(tmp_path / "s2"), info_fn=lambda *a: info(*a, size=10))
    m.cancel()
    s = wait_for(m, ("cancelled", "failed"), timeout=8)
    assert s["state"] == "cancelled" and killed


def test_fingerprinting_a_dataset_runs_in_the_background(world, tmp_path):
    data = tmp_path / "d"
    data.mkdir()
    (data / "a.jsonl").write_text('{"x":1}\n')
    m = dl.Manager()
    m.hash_dataset("mine", str(data), "note")
    s = wait_for(m, ("done", "failed"))
    assert s["state"] == "done" and s["result"] == {"name": "mine", "version": 1, "new": True}
    m.hash_dataset("mine", None, verify_only=True)
    assert wait_for(m, ("done", "failed"))["result"]["status"] == "ok"


# ---- through the launcher (the page's requests)
@pytest.fixture
def launcher(world, tmp_path, monkeypatch):
    kit = tmp_path / "kit"
    (kit / "recipes").mkdir(parents=True)
    (kit / "submit.sh").write_text("")
    monkeypatch.setattr(cluster_profile, "current", lambda: NS(gpus_per_node=4, partitions=[]))
    return kit, launch.Launcher(kit, runs=lambda: [], env={})


def test_these_requests_need_no_slurm_account(launcher):
    kit, L = launcher
    assert L.disabled_reason()                                   # no BHASKERA_DIR / SLURM_ACCOUNT in this environment
    assert L.launch({"kind": "dataset", "action": "list"}) == {"ok": True, "datasets": []}
    with pytest.raises(launch.LaunchError, match="does not look like"):
        L.launch({"kind": "hf", "action": "login", "key": "nope"})
    assert L.launch({"kind": "download", "action": "status"})["task"]["state"] == "idle"
    with pytest.raises(launch.LaunchError, match="owner/name"):
        L.launch({"kind": "download", "action": "preview", "what": "model", "repo": "https://x.y/z"})
    with pytest.raises(launch.LaunchError):
        L.launch({"kind": "dataset", "action": "explode"})


def test_training_on_data_that_changed_after_it_was_registered_is_refused(launcher, tmp_path):
    kit, L = launcher
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "x.parquet").write_bytes(b"1234")
    dv.add("sup", str(cache))
    import yaml
    r = kit / "recipes" / "r.yaml"
    r.write_text(yaml.safe_dump({"model": {"name": "/m/X"}, "data": {"tokenized_path": str(cache)}, "training": {"distributed": {"strategy": "ddp"}}}))
    L._check_data(r)                                              # unchanged: fine
    (cache / "x.parquet").write_bytes(b"12345678")
    with pytest.raises(launch.LaunchError, match="changed since it was registered as sup v1"):
        L._check_data(r)
    assert dv.add("sup", str(cache))["version"] == 2
    L._check_data(r)


def test_the_options_list_datasets_downloads_and_the_version_of_each_cache(launcher, tmp_path):
    kit, L = launcher
    cache = tmp_path / "cache" / "local_train_abc"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text(json.dumps({"model_name": "/m/Q", "seq_len": 128, "num_rows": 5}))
    (cache / "p.parquet").write_bytes(b"abcd")
    dv.add("sup", str(cache))
    o = L.options()
    assert [d["name"] for d in o["datasets"]] == ["sup"] and o["downloaded"] == []


def test_a_registered_cache_carries_its_version_without_touching_its_own_dataset_field(launcher, tmp_path):
    kit, L = launcher
    cache = tmp_path / "cache" / "local_train_abc"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text(json.dumps({"model_name": "/m/Q", "seq_len": 128, "num_rows": 5, "dataset_name": "tickets"}))
    (cache / "p.parquet").write_bytes(b"abcd")
    L._env["BHASKERA_CACHE"] = str(cache.parent)
    before = [c for c in L._caches() if c["path"] == str(cache.resolve())][0]
    assert "data_version" not in before and before["dataset"] == "tickets"
    dv.add("sup", str(cache))
    after = [c for c in L._caches() if c["path"] == str(cache.resolve())][0]
    assert after["data_version"] == {"name": "sup", "version": 1, "status": "ok"} and after["dataset"] == "tickets"


# ---- taking runs off the page
def test_runs_can_be_removed_but_never_a_running_one_or_one_that_is_not_ours(world, tmp_path, monkeypatch):
    from bhaskera.launcher import register
    kit = tmp_path / "kit"
    (kit / "recipes").mkdir(parents=True)
    (kit / "submit.sh").write_text("")
    monkeypatch.setattr(cluster_profile, "current", lambda: NS(gpus_per_node=4, partitions=[]))
    ids = {n: (n * 32) for n in "abcde"}
    runs = [{"run_id": ids["a"], "status": "FINISHED"}, {"run_id": ids["b"], "status": "FAILED"},
            {"run_id": ids["c"], "status": "RUNNING", "stale": False}, {"run_id": ids["d"], "status": "RUNNING", "stale": True}]
    deleted, changed = [], []
    monkeypatch.setattr(register, "make_client", lambda store: NS(delete_run=lambda i: deleted.append(i)))
    L = launch.Launcher(kit, runs=lambda: runs, env={})
    L.after_change = lambda: changed.append(1)
    r = L.launch({"kind": "runs", "action": "remove", "runs": [ids["a"], ids["b"], ids["c"], ids["d"], ids["e"], ids["a"]]})
    assert r["ok"] and r["removed"] == 3 and deleted == [ids["a"], ids["b"], ids["d"]] and changed == [1]
    reasons = {s["run_id"]: s["reason"] for s in r["skipped"]}
    assert reasons == {ids["c"]: "still running", ids["e"]: "not one of your Bhaskera runs"}
    for bad in (None, [], "abc", ["not-a-run-id"], [1], ["a" * 32] * 501):
        with pytest.raises(launch.LaunchError, match="choose the runs"):
            L.launch({"kind": "runs", "action": "remove", "runs": bad})
    changed.clear()
    L.launch({"kind": "runs", "action": "remove", "runs": [ids["c"]]})
    assert changed == []                                                     # nothing removed, nothing to refresh


# ---- a machine without internet: the preview and the download run on one that has it
def test_this_machine_is_used_when_it_has_internet(world, monkeypatch):
    dl._route_cache.clear()
    assert dl.route(probe=lambda: True) is None


def test_another_machine_is_used_when_this_one_has_no_internet(world, monkeypatch):
    dl._route_cache.clear()
    monkeypatch.setenv("BHASKERA_DOWNLOAD_HOSTS", "login04, login01 login02")
    monkeypatch.setattr("socket.gethostname", lambda: "login04")
    asked = []

    def run(cmd, **kw):
        asked.append(cmd[-1] if cmd[0] == "ssh" else cmd)
        host = [c for c in cmd if c.startswith("login")][0]
        return NS(returncode=0 if host == "login02" else 255, stdout="", stderr="")
    assert dl.route(probe=lambda: False, run=run) == "login02"
    assert len(asked) == 2 and all("bhaskera.downloads online" in a for a in asked)        # login04 (this one) is skipped; login01 failed; login02 answered
    assert dl.route(probe=lambda: False, run=lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked again"))) == "login02"      # remembered


def test_with_no_internet_anywhere_the_message_names_this_machine_and_what_to_set(world, monkeypatch):
    dl._route_cache.clear()
    monkeypatch.delenv("BHASKERA_DOWNLOAD_HOSTS", raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "login06")
    monkeypatch.setattr(cluster_profile, "current", lambda: NS(download_hosts=[]))
    with pytest.raises(dl.DownloadError, match="login06.*cannot reach the internet.*download_hosts"):
        dl.route(probe=lambda: False)
    dl._route_cache.clear()
    monkeypatch.setenv("BHASKERA_DOWNLOAD_HOSTS", "login01")
    with pytest.raises(dl.DownloadError, match="login06.*login01"):
        dl.route(probe=lambda: False, run=lambda *a, **k: NS(returncode=255, stdout="", stderr=""))


def test_download_hosts_come_from_the_environment_or_the_profile_and_are_checked(world, monkeypatch):
    monkeypatch.setenv("BHASKERA_DOWNLOAD_HOSTS", "a1,b-2;rm,$(x), ok.host,-oProxyCommand=x")
    assert dl.download_hosts() == ["a1", "ok.host"]            # a name that is not a plain host name (an ssh option, a command) is dropped
    monkeypatch.delenv("BHASKERA_DOWNLOAD_HOSTS")
    monkeypatch.setattr(cluster_profile, "current", lambda: NS(download_hosts=["login01", "bad host"]))
    assert dl.download_hosts() == ["login01"]


def test_the_command_for_another_machine_has_no_key_and_is_quoted(world, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/a b/src")
    cmd = dl._remote("login01", ["run", "--repo", "x/y", "--target", "/t t"], tty=True, python="/venv/bin/python")
    assert cmd[:2] == ["ssh", "-o"] and "-tt" in cmd and "login01" in cmd and "BatchMode=yes" in cmd
    remote = cmd[-1]
    assert "'PYTHONPATH=/a b/src'" in remote and "/venv/bin/python -m bhaskera.downloads run" in remote and "'/t t'" in remote
    assert "hf_" not in " ".join(cmd) and "HF_TOKEN" not in " ".join(cmd)


def test_the_preview_is_asked_of_the_machine_with_internet(world, monkeypatch):
    dl._route_cache.clear()
    answer = {"kind": "model", "repo": "a/b", "revision": "main", "bytes": 10, "files": 1, "largest": [], "gated": False, "private": False, "license": "", "sha": "", "large": False}
    seen = []
    monkeypatch.setattr(dl, "route", lambda env=None, **k: "login01")
    monkeypatch.setattr(dl.subprocess, "run", lambda cmd, **k: (seen.append(cmd), NS(returncode=0, stdout="noise\n" + json.dumps(answer) + "\n", stderr=""))[1])
    out = dl.preview("model", "a/b", None)
    assert out["via"] == "login01" and out["bytes"] == 10 and "preview" in seen[0][-1]
    monkeypatch.setattr(dl.subprocess, "run", lambda cmd, **k: NS(returncode=2, stdout="", stderr="error: Hugging Face has no such repository\n"))
    with pytest.raises(dl.DownloadError, match="no such repository"):
        dl.preview("model", "a/b", None)


def test_a_download_runs_on_the_machine_with_internet_and_says_so(world, tmp_path, monkeypatch):
    seen = []

    def popen(cmd, **kw):
        seen.append((cmd, kw))
        return Proc(tmp_path / "m", 0, [("config.json", 10)])
    (tmp_path / "m").mkdir()
    m = dl.Manager(popen=popen)
    s = m.start_download("model", "a/b", "main", str(tmp_path / "m"), info_fn=lambda *a: info(*a, size=10))
    assert "via" in s
    cmd = seen[0][0]
    assert cmd[0] == "python3" or "bhaskera.downloads" in " ".join(cmd)
    wait_for(m, ("done", "failed"))
