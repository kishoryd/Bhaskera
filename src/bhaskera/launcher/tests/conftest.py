"""Shared fixtures: a cluster profile in use for the duration of a test."""
from __future__ import annotations

import os

import pytest

from bhaskera.launcher import cluster_profile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PARAM_RUDRA = os.path.join(REPO, "slurm", "clusters", "param-rudra.yaml")


@pytest.fixture
def use_profile(monkeypatch, tmp_path):
    """use_profile(path_or_None): make that profile (or none at all) the one in use."""
    def apply(path):
        if path is None:
            monkeypatch.delenv("BHASKERA_CLUSTER_PROFILE", raising=False)
            monkeypatch.setenv("HOME", str(tmp_path))          # no ~/.bhaskera/cluster.yaml either
        else:
            monkeypatch.setenv("BHASKERA_CLUSTER_PROFILE", str(path))
        for var in ("BHASKERA_PARTITION", "BHASKERA_MAX_HOURS", "BHASKERA_GPUS_PER_NODE", "BHASKERA_NET_INTERFACE"):
            monkeypatch.delenv(var, raising=False)
        cluster_profile.reset()
    yield apply
    cluster_profile.reset()


@pytest.fixture
def param_rudra(use_profile):
    use_profile(PARAM_RUDRA)
