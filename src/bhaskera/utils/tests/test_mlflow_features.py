"""MLflow tracing, the evaluation report and evaluation samples, with the real logger on a file store."""
from __future__ import annotations

import json

import pytest
import torch

mlflow = pytest.importorskip("mlflow")

from bhaskera.config import Config  # noqa: E402
from bhaskera.trainer.eval_samples import capture_rows, render_rows  # noqa: E402
from bhaskera.utils.loggers.base import NullSpan, phase_span  # noqa: E402
from bhaskera.utils.loggers.mlflow_logger import MLflowLogger  # noqa: E402


def _logger(tmp_path, monkeypatch, **options):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'store'}")
    cfg = Config.from_dict({"logging": {"tracker": "mlflow", "project": "feat", "run_name": "r",
                                        "mlflow_tracking_uri": f"file://{tmp_path / 'store'}", **options}})
    lg = MLflowLogger(cfg, rank=0, world_size=1)
    assert lg._available
    return lg


def _client(tmp_path):
    return mlflow.tracking.MlflowClient(tracking_uri=f"file://{tmp_path / 'store'}")


def _artifacts(tmp_path, run_id, path):
    return {a.path for a in _client(tmp_path).list_artifacts(run_id, path)}


def test_no_tracker_means_a_do_nothing_span():
    with phase_span(None, "evaluation", step=1) as span:
        assert isinstance(span, NullSpan)
        span.set_outputs({"a": 1})


def test_tracing_is_off_by_default_and_on_when_asked(tmp_path, monkeypatch):
    lg = _logger(tmp_path, monkeypatch)
    with lg.span("evaluation", step=1) as sp:
        sp.set_outputs({"eval_loss": 1.0})
    lg.finish()
    exp = mlflow.get_experiment_by_name("feat").experiment_id
    assert mlflow.search_traces(locations=[exp], return_type="list") == []

    lg2 = _logger(tmp_path, monkeypatch, mlflow_tracing=True, run_name="traced")
    with lg2.span("evaluation", step=5) as sp:
        sp.set_outputs({"eval_loss": 1.2})
    with pytest.raises(ValueError):                       # an error in the phase still propagates
        with lg2.span("checkpoint", step=5):
            raise ValueError("disk full")
    lg2.finish()
    traces = mlflow.search_traces(locations=[exp], return_type="list")
    assert sorted(t.data.spans[0].name for t in traces) == ["checkpoint", "evaluation"]
    assert {t.info.state.value for t in traces} == {"OK", "ERROR"}


def test_evaluation_report_is_stored_with_the_run(tmp_path, monkeypatch):
    lg = _logger(tmp_path, monkeypatch)
    for step, loss in ((10, 2.0), (20, 1.5), (30, 1.7)):
        lg.log({"loss": 1.0, "eval/loss": loss, "eval/accuracy": 0.5, "eval/perplexity": 2.718 ** loss}, step=step)
    run_id = lg._run_id
    lg.finish()
    assert {"evaluation/history.json", "evaluation/summary.json"} <= _artifacts(tmp_path, run_id, "evaluation")
    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="evaluation/summary.json",
                                                tracking_uri=f"file://{tmp_path / 'store'}")
    summary = json.load(open(local))
    assert summary["evaluations"] == 3 and summary["best"]["step"] == 20 and summary["final"]["step"] == 30
    assert set(_client(tmp_path).get_run(run_id).data.metrics) == {"loss", "eval_loss", "eval_accuracy", "eval_perplexity"}


def test_the_report_can_be_switched_off(tmp_path, monkeypatch):
    lg = _logger(tmp_path, monkeypatch, mlflow_eval_report=False)
    lg.log({"eval/loss": 1.0}, step=1)
    run_id = lg._run_id
    lg.finish()
    assert _artifacts(tmp_path, run_id, "evaluation") == set()


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"t{i}" for i in ids)


def test_sample_rows_show_prompt_reference_and_prediction(tmp_path, monkeypatch):
    vocab = 10
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[-100, -100, 3, 4, 5, -100]])
    logits = torch.zeros(1, 6, vocab)
    for pos, tok in ((1, 3), (2, 4), (3, 9)):             # position t predicts token t+1: right, right, wrong
        logits[0, pos, tok] = 5.0
    rows = render_rows(capture_rows(ids, labels, logits, count=3), _Tok())
    assert rows == [{"prompt": "t1 t2", "reference": "t3 t4 t5", "prediction": "t3 t4 t9", "match": 0.6667}]
    assert render_rows(capture_rows(ids, torch.full_like(labels, -100), logits, 1), _Tok()) == []   # no answer: no row
    # labels that start at the very first token (no masked prompt): that token has no prediction and is left out
    unmasked = render_rows(capture_rows(ids, ids.clone(), logits, 1), _Tok())
    assert unmasked[0]["prompt"] == "t1" and unmasked[0]["reference"].startswith("t2")

    lg = _logger(tmp_path, monkeypatch)
    lg.log_eval_samples(rows, step=40)
    run_id = lg._run_id
    lg.finish()
    assert "evaluation/samples_step40.json" in _artifacts(tmp_path, run_id, "evaluation")


def test_the_new_options_reach_the_config():
    log = Config.from_dict({}).logging
    assert (log.mlflow_tracing, log.mlflow_eval_report, log.mlflow_eval_samples) == (False, True, 0)
    log = Config.from_dict({"logging": {"mlflow_tracing": True, "mlflow_eval_samples": 4}}).logging
    assert (log.mlflow_tracing, log.mlflow_eval_samples) == (True, 4)
