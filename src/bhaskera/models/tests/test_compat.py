"""patch_transformers_for_remote_code: re-creates removed names only when missing. CPU only."""
from __future__ import annotations

import transformers.utils as tu

from bhaskera.models.compat import patch_transformers_for_remote_code


def test_loss_kwargs_created_when_missing(monkeypatch):
    monkeypatch.delattr(tu, "LossKwargs", raising=False)
    patch_transformers_for_remote_code()

    class Kw(tu.LossKwargs):          # the way model code uses it
        pass

    assert "num_items_in_batch" in tu.LossKwargs.__annotations__


def test_existing_loss_kwargs_left_alone(monkeypatch):
    sentinel = type("Sentinel", (), {})
    monkeypatch.setattr(tu, "LossKwargs", sentinel, raising=False)
    patch_transformers_for_remote_code()
    assert tu.LossKwargs is sentinel
