"""A recipe key the parser never reads is reported, with the closest real name; the shipped recipes are clean."""
from __future__ import annotations

import glob
import os

import pytest
import yaml

from bhaskera.config import find_unknown_keys, load_config, load_config_checked, _Tracked, _dict_to_config


def _check(raw: dict) -> list[str]:
    tracked = _Tracked(raw)
    _dict_to_config(tracked)
    return find_unknown_keys(tracked)


def test_a_typo_is_reported_with_the_closest_name():
    problems = _check({"training": {"max_stepz": 10, "lr": 1e-4}})
    assert problems == ["'training.max_stepz' is not an option and has no effect (did you mean 'max_steps'?)"]


def test_a_misplaced_option_is_reported_at_its_place():
    problems = _check({"logging": {"tracker": "mlflow"}, "mlflow_metrics": "full"})
    assert len(problems) == 1 and problems[0].startswith("'mlflow_metrics' is not an option")


def test_nested_sections_are_checked():
    problems = _check({"training": {"distributed": {"fsdp": {"sharding_stratgy": "FULL_SHARD"}}}})
    assert problems and "training.distributed.fsdp.sharding_stratgy" in problems[0] and "sharding_strategy" in problems[0]


def test_correct_options_are_not_reported():
    assert _check({"training": {"max_steps": 5, "lr": 1e-4}, "logging": {"tracker": "mlflow", "mlflow_metrics": "full",
                                                                           "mlflow_log_all_ranks": True}}) == []


def test_documented_mlflow_options_are_now_read():
    cfg = _dict_to_config({"logging": {"mlflow_experiment": "mine", "mlflow_log_all_ranks": True,
                                       "mlflow_log_artifacts_every": 10}})
    assert (cfg.logging.mlflow_experiment, cfg.logging.mlflow_log_all_ranks, cfg.logging.mlflow_log_artifacts_every) == ("mine", True, 10)


def test_load_config_warns_and_strict_mode_fails(tmp_path, monkeypatch, caplog):
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump({"training": {"max_stepz": 3}}))
    with caplog.at_level("WARNING"):
        load_config(str(path))
    assert "max_stepz" in caplog.text
    monkeypatch.setenv("BHASKERA_STRICT_CONFIG", "1")
    with pytest.raises(ValueError, match="do nothing"):
        load_config(str(path))


def test_an_empty_file_is_fine(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    cfg, problems = load_config_checked(str(path))
    assert problems == [] and cfg.training.max_steps


RECIPES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "slurm", "recipes", "*.yaml")))


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: os.path.basename(p))
def test_shipped_recipes_have_no_ignored_options(path):
    if "tokenize" in os.path.basename(path):
        pytest.skip("the tokenize recipe is read by the tokenizer, not the training parser")
    assert load_config_checked(path)[1] == []
