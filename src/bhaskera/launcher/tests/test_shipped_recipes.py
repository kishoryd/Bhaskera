"""Every recipe shipped in slurm/recipes loads with the real config loader, names no key that does nothing, and matches its file name."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
from bhaskera.config import load_config_checked  # noqa: E402

RECIPES = sorted((Path(__file__).resolve().parents[4] / "slurm" / "recipes").glob("*.yaml"))
TRAINING = [p for p in RECIPES if not p.name.startswith("tokenize")]


def test_the_benchmark_matrix_is_all_there():
    """Five models x five methods, except the 20 benchmark runs that ran out of memory (a recipe for those would only fail)."""
    names = {p.stem for p in TRAINING}
    too_big = {("llama-3.3-70b", "ddp"), ("mistral-large-123b", "ddp"), ("qwen3-235b-a22b", "ddp"), ("qwen3-235b-a22b", "qlora")}
    for model in ("qwen3-14b", "qwen3-30b-a3b", "llama-3.3-70b", "mistral-large-123b", "qwen3-235b-a22b"):
        for method in ("ddp", "qlora", "pipeline", "hsdp", "fsdp"):
            assert (f"{model}-{method}" in names) == ((model, method) not in too_big), f"{model}-{method}"


@pytest.mark.parametrize("path", TRAINING, ids=lambda p: p.stem)
def test_recipe_loads_cleanly(path):
    cfg, unknown = load_config_checked(str(path))
    assert unknown == [], unknown
    assert cfg.logging.run_name == path.stem

