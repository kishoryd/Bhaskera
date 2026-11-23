"""The Open OnDemand app: the installer, the checker and the templates, against temporary folders (no portal needed)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

OOD = Path(__file__).resolve().parents[4] / "ood"


def sh(*args, cwd=OOD, **kw):
    return subprocess.run(["bash", *map(str, args)], cwd=cwd, capture_output=True, text=True, **kw)


@pytest.fixture
def site(tmp_path):
    (tmp_path / "clusters.d").mkdir()
    (tmp_path / "clusters.d" / "grace.yml").write_text("v2: {}\n")
    return tmp_path


def install(site, *extra):
    return sh(OOD / "install.sh", "--partition", "interactive", "--hours", "6", "--root", "/opt/bhaskera", "--dest", site / "sys" / "bhaskera",
              "--clusters-dir", site / "clusters.d", "--skip-slurm-check", *extra)


def test_install_writes_every_site_setting_and_leaves_no_placeholder(site):
    r = install(site)
    assert r.returncode == 0, r.stderr
    dest = site / "sys" / "bhaskera"
    form = yaml.safe_load((dest / "form.yml.erb").read_text())
    assert form["cluster"] == "grace" and form["form"] == ["bc_queue", "bc_num_hours"]
    assert form["attributes"]["bc_queue"]["value"] == "interactive" and form["attributes"]["bc_num_hours"]["value"] == 6
    script = (dest / "template" / "script.sh.erb").read_text()
    assert 'ROOT="/opt/bhaskera"' in script and 'BHASKERA_EXTERNAL_PREFIX="/rnode/' in script
    assert 'action="/rnode/' in (dest / "view.html.erb").read_text()
    for p in dest.rglob("*"):
        if p.is_file():
            assert "__CLUSTER__" not in p.read_text() and "__PARTITION__" not in p.read_text() and "__ROOT__" not in p.read_text() and "__PROXY__" not in p.read_text(), p


def test_the_session_script_loads_no_module_and_needs_nothing_from_the_user(site):
    install(site, "--proxy-path", "/node/")
    script = (site / "sys" / "bhaskera" / "template" / "script.sh.erb").read_text()
    assert "module load" not in script and "module purge" not in script
    assert 'BHASKERA_EXTERNAL_PREFIX="/node/${host}/${port}"' in script          # the trailing slash of the option is dropped
    for needed in ("BHASKERA_DIR=", "BHASKERA_KIT=", "BHASKERA_CLUSTER_PROFILE", "MLFLOW_ALLOW_FILE_STORE=true", "sacctmgr", "bhaskera-runs", "BHASKERA_HUB_TOKEN"):
        assert needed in script, needed
    r = sh("-n", site / "sys" / "bhaskera" / "template" / "script.sh.erb")
    assert r.returncode == 0, r.stderr


def test_install_refuses_what_it_cannot_decide(site):
    assert sh(OOD / "install.sh", "--clusters-dir", site / "clusters.d", "--dest", site / "x", "--skip-slurm-check").returncode == 1            # no partition
    (site / "clusters.d" / "habana.yml").write_text("v2: {}\n")
    r = install(site)
    assert r.returncode == 1 and "--cluster" in r.stderr                              # two clusters: say which
    assert install(site, "--cluster", "habana").returncode == 0
    assert sh(OOD / "install.sh", "--partition", "p", "--hours", "x", "--dest", site / "y", "--clusters-dir", site / "clusters.d", "--cluster", "grace").returncode == 1


def test_check_accepts_matching_node_names_and_rejects_others(site):
    install(site)
    conf = site / "ood_portal.yml"
    conf.write_text("host_regex: 'grace[0-9]+|hopper[0-9]+'\nrnode_uri: '/rnode'\n")
    ok = sh(OOD / "check.sh", "--dest", site / "sys" / "bhaskera", "--portal-conf", conf, "--nodes", "grace03 hopper05")
    assert "host_regex accepts grace03" in ok.stdout and "rnode_uri is /rnode" in ok.stdout
    bad = sh(OOD / "check.sh", "--dest", site / "sys" / "bhaskera", "--portal-conf", conf, "--nodes", "grace03 cbcn0176")
    assert bad.returncode == 1 and "REJECTS cbcn0176" in bad.stdout


@pytest.mark.skipif(shutil.which("ruby") is None, reason="needs ruby to render the templates")
def test_the_templates_render_and_parse(site):
    install(site)
    dest = site / "sys" / "bhaskera"
    code = ('require "erb"; require "yaml"; bc_queue="interactive"; bc_num_hours="6"; host="n1"; port=30000; password="p"*24\n'
            'B = binding\ndef r(f) ERB.new(File.read(f), trim_mode: "-").result(B) end\n'
            'y = YAML.safe_load(r("submit.yml.erb")); abort "bad submit" unless y["script"]["wall_time"] == "21600" && y["script"]["queue_name"] == "interactive"\n'
            'YAML.safe_load(File.read("form.yml.erb")); puts r("view.html.erb").include?("/rnode/n1/30000/")')
    r = subprocess.run(["ruby", "-e", code], cwd=dest, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "true", r.stderr
