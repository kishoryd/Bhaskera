"""Run identity (tags and params) must reach the MLflow store; the orphan-run cleanup relies on `slurm_job`."""
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow")

from bhaskera.utils.loggers.mlflow_logger import MLflowLogger  # noqa: E402


def test_identity_tags_and_params_are_written(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("identity-test")
    run = mlflow.start_run(run_name="r")
    try:
        lg = MLflowLogger.__new__(MLflowLogger)
        lg._mlflow, lg._run_id = mlflow, run.info.run_id
        lg._hostname, lg._rank, lg._world_size = "node1", 0, 2
        lg._log_run_identity(SimpleNamespace(logging=SimpleNamespace(group=None, tags=None),
                                             as_dict=lambda: {"training": {"lr": 0.1}}))
    finally:
        mlflow.end_run()
    data = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data
    assert data.tags["slurm_job"] == "4242" and data.tags["framework"] == "bhaskera"
    assert data.params["slurm_job"] == "4242" and data.params["training.lr"] == "0.1"
