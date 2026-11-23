"""The exported adapter must load into a fresh copy of the base model and give the same outputs."""
from __future__ import annotations

import copy
import os
import tempfile

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from bhaskera.utils.adapter_export import export_adapter  # noqa: E402


def _base():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


def _trained(base):
    model = get_peft_model(copy.deepcopy(base), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                           task_type="CAUSAL_LM"))
    for name, p in model.named_parameters():          # as if trained: LoRA B starts at zero
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.05)
    return model.eval()


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_export_loads_back_with_identical_outputs(tmp_path):
    base = _base()
    trained = _trained(base)
    ids = torch.randint(0, 64, (2, 8))
    out = export_adapter(trained, tmp_path / "adapter", rank=0, base_model="tiny-llama", steps=7)
    assert out is not None
    assert {p.name for p in out.iterdir()} == {"adapter_model.safetensors", "adapter_config.json", "README.md"}
    reloaded = PeftModel.from_pretrained(copy.deepcopy(base), str(out)).eval()
    assert torch.allclose(_logits(trained, ids), _logits(reloaded, ids), atol=1e-6)
    assert not torch.allclose(_logits(base, ids), _logits(reloaded, ids))      # the adapter really changes things


def test_export_works_through_a_ddp_wrapper(tmp_path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    base = _base()
    trained = _trained(base)
    ids = torch.randint(0, 64, (2, 8))
    expected = _logits(trained, ids)
    with tempfile.TemporaryDirectory() as d:
        dist.init_process_group("gloo", init_method=f"file://{d}/rendezvous", rank=0, world_size=1)
        try:
            out = export_adapter(DistributedDataParallel(trained), tmp_path / "ddp", rank=0)
        finally:
            dist.destroy_process_group()
    reloaded = PeftModel.from_pretrained(copy.deepcopy(base), str(out)).eval()
    assert torch.allclose(expected, _logits(reloaded, ids), atol=1e-6)


def test_a_model_without_an_adapter_is_skipped(tmp_path):
    assert export_adapter(_base(), tmp_path / "none", rank=0) is None
    assert not (tmp_path / "none").exists()


def test_other_ranks_write_nothing(tmp_path):
    assert export_adapter(_trained(_base()), tmp_path / "r1", rank=1) is None
    assert not (tmp_path / "r1").exists()


def test_adapter_is_stored_with_the_run_and_registered(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from bhaskera.utils.loggers.mlflow_logger import MLflowLogger
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path / 'store'}")
    mlflow.set_experiment("registry")
    base = _base()
    out = export_adapter(_trained(base), tmp_path / "adapter", rank=0, base_model="tiny-llama")
    run = mlflow.start_run()
    try:
        lg = MLflowLogger.__new__(MLflowLogger)
        lg._mlflow, lg._run_id, lg._available = mlflow, run.info.run_id, True
        assert lg.log_adapter(out, register_as="my-adapter", tags={"steps": 7}) == "1"
        assert lg.log_adapter(out, register_as="my-adapter") == "2"          # a second run adds a version
        assert lg.log_adapter(out) is None                                    # stored only, not registered
    finally:
        mlflow.end_run()
    client = mlflow.tracking.MlflowClient()
    files = {a.path for a in client.list_artifacts(run.info.run_id, "adapter")}
    assert files == {"adapter/adapter_model.safetensors", "adapter/adapter_config.json", "adapter/README.md"}
    versions = client.search_model_versions("name='my-adapter'")
    assert sorted(str(v.version) for v in versions) == ["1", "2"]
    assert versions[0].tags["format"] == "peft-lora" and versions[0].run_id == run.info.run_id
    # what the registry points at loads like any PEFT adapter
    local = mlflow.artifacts.download_artifacts(artifact_uri=versions[0].source, dst_path=str(tmp_path / "dl"))
    ids = torch.randint(0, 64, (2, 8))
    reloaded = PeftModel.from_pretrained(copy.deepcopy(base), local).eval()
    assert _logits(reloaded, ids).shape == (2, 8, 64)


def test_publish_adapter_follows_the_recipe_options(tmp_path):
    from types import SimpleNamespace
    from bhaskera.trainer.loop import _publish_adapter

    class Tracker:
        def __init__(self):
            self.calls = []

        def log_adapter(self, path, register_as=None, tags=None):
            self.calls.append((path.name, register_as, tags["base_model"]))
            return "3"

    def cfg(log_adapter, register):
        return SimpleNamespace(logging=SimpleNamespace(mlflow_log_adapter=log_adapter, mlflow_register_model=register),
                               checkpoint=SimpleNamespace(save_dir=str(tmp_path)), model=SimpleNamespace(name="tiny"))

    model, tracker = _trained(_base()), Tracker()
    _publish_adapter(model, cfg(False, None), tracker, 0, 5)
    assert tracker.calls == [] and not (tmp_path / "adapter").exists()      # off by default
    _publish_adapter(model, cfg(True, None), tracker, 0, 5)
    _publish_adapter(model, cfg(False, "name"), tracker, 0, 5)              # a model name alone also turns it on
    assert tracker.calls == [("adapter", None, "tiny"), ("adapter", "name", "tiny")]
    _publish_adapter(object(), cfg(True, None), tracker, 0, 5)              # a model without adapter: no failure
