"""Only the curves users need reach MLflow by default; the level can be raised."""
import pytest

from bhaskera.config import Config
from bhaskera.utils.loggers import mlflow_metrics as mm

RAW = ["loss", "lr", "grad_norm", "eval_loss", "eval_accuracy", "eval_perplexity", "throughput_tokens_per_sec",
       "throughput_mfu_pct", "throughput_hfu_pct", "time_step_s", "time_compute_s", "cluster_gpu_util_pct_mean",
       "cluster_gpu_power_w_sum", "gpu_0_util_pct", "sys_cpu_pct", "proc_rss_mib", "node_cbgpu0001_gpu0_util_pct"]


def _kept(level):
    return {n for n in RAW if mm.wanted(n, level)}


def test_essential_keeps_what_a_user_reads():
    kept = _kept("essential")
    assert {"loss", "lr", "grad_norm", "eval_loss", "eval_accuracy", "eval_perplexity",
            "throughput_tokens_per_sec", "throughput_mfu_pct", "cluster_gpu_util_pct_mean",
            "cluster_gpu_power_w_sum"} <= kept   # the energy a run used is worked out from this one
    assert not kept & {"gpu_0_util_pct", "sys_cpu_pct", "proc_rss_mib", "node_cbgpu0001_gpu0_util_pct",
                       "throughput_hfu_pct", "time_compute_s"}
    assert len(mm.ESSENTIAL) < 25


def test_detailed_adds_job_wide_families_but_not_per_node_readings():
    kept = _kept("detailed")
    assert {"throughput_hfu_pct", "time_compute_s", "cluster_gpu_power_w_sum"} <= kept
    assert not kept & {"gpu_0_util_pct", "sys_cpu_pct", "proc_rss_mib", "node_cbgpu0001_gpu0_util_pct"}


def test_full_keeps_everything():
    assert _kept("full") == set(RAW)


def test_level_names_fall_back_to_the_default():
    assert mm.normalize_level("Detailed") == "detailed"
    assert mm.normalize_level("everything") == "essential"
    assert mm.normalize_level(None) == "essential"


def test_the_recipe_option_reaches_the_config():
    assert Config.from_dict({}).logging.mlflow_metrics == "essential"
    assert Config.from_dict({"logging": {"mlflow_metrics": "full"}}).logging.mlflow_metrics == "full"


def test_logger_writes_only_the_chosen_level(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from bhaskera.utils.loggers.mlflow_logger import MLflowLogger
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("levels")
    run = mlflow.start_run()
    try:
        lg = MLflowLogger.__new__(MLflowLogger)
        lg._mlflow, lg._run_id, lg._rank, lg._hostname, lg._metrics_level = mlflow, run.info.run_id, 0, "n1", "essential"
        lg._flush_one({n: 1.0 for n in RAW}, step=1)
    finally:
        mlflow.end_run()
    written = set(mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.metrics)
    assert written == _kept("essential")


def test_adapter_options_reach_the_config():
    log = Config.from_dict({}).logging
    assert log.mlflow_log_adapter is False and log.mlflow_register_model is None
    log = Config.from_dict({"logging": {"mlflow_log_adapter": True, "mlflow_register_model": "qwen-sft"}}).logging
    assert log.mlflow_log_adapter is True and log.mlflow_register_model == "qwen-sft"
