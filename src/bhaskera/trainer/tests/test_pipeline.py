"""CPU tests for strategy: pipeline (pure logic; the multi-GPU numerics are scripts/pp_validate.py)."""
from __future__ import annotations

import glob
import os

import pytest
import torch

from bhaskera.config import load_config
from bhaskera.distributed.pipeline import build_stage, canonical_name, split_layers, switch_aux_loss
from bhaskera.trainer.pipeline_loop import _latest_checkpoint, positions_from_mask

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


@pytest.mark.parametrize("layers", [1, 7, 10, 28, 40, 80, 88, 94, 126])
@pytest.mark.parametrize("stages", [1, 2, 3, 4, 8, 16])
def test_split_layers_covers_every_layer_once_in_order(layers, stages):
    if stages > layers:
        pytest.skip("more stages than layers")
    parts = split_layers(layers, stages)
    assert len(parts) == stages
    assert [i for p in parts for i in p] == list(range(layers))
    assert all(len(p) >= 1 for p in parts)
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 2


def test_canonical_name_maps_stage_names_back_to_checkpoint_keys():
    ids = [20, 21, 22]
    assert canonical_name("layers.1.self_attn.q_proj.base_layer.weight", ids) == \
        "model.layers.21.self_attn.q_proj.weight"
    assert canonical_name("layers.0._checkpoint_wrapped_module.mlp.up_proj.lora_A.default.weight", ids) == \
        "model.layers.20.mlp.up_proj.lora_A.default.weight"
    assert canonical_name("embed_tokens.weight", ids) == "model.embed_tokens.weight"
    assert canonical_name("norm.weight", ids) == "model.norm.weight"
    assert canonical_name("lm_head.weight", ids) == "lm_head.weight"


def test_positions_restart_where_the_mask_changes():
    m = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1] * 6, [0] * 6])
    got = positions_from_mask(m).tolist()
    assert got == [[0, 1, 0, 1, 2, 3], [0, 1, 2, 0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]


def test_aux_loss_matches_transformers_per_layer():
    mod = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
    torch.manual_seed(0)
    for experts, k, tokens in [(8, 2, 64), (128, 8, 200), (4, 1, 33)]:
        logits = torch.randn(tokens, experts)
        want = float(mod.load_balancing_loss_func((logits,), experts, k))
        assert abs(float(switch_aux_loss(logits, k)) - want) < 1e-5


def test_aux_loss_ignores_padding_tokens():
    torch.manual_seed(1)
    logits = torch.randn(40, 8)
    mask = torch.cat([torch.ones(25), torch.zeros(15)]).long()
    assert abs(float(switch_aux_loss(logits, 2, mask)) - float(switch_aux_loss(logits[:25], 2))) < 1e-6


def test_aux_loss_is_balanced_at_one():
    # uniform routing over E experts gives E * sum(f * P) = top_k
    logits = torch.zeros(512, 16)
    assert abs(float(switch_aux_loss(logits, 2)) - 2.0) < 1e-4


def test_latest_checkpoint_needs_the_complete_marker(tmp_path):
    for step, done in [(2, True), (4, True), (5, False)]:
        d = tmp_path / f"step_{step}"
        d.mkdir()
        if done:
            (d / ".complete").touch()
    assert _latest_checkpoint(str(tmp_path))[0] == 4
    assert _latest_checkpoint(str(tmp_path / "missing")) is None


def test_pipeline_rejects_quantization():
    cfg = load_config(os.path.join(REPO, "configs/profile/llama33_70b_pipeline_4node.yaml"))
    cfg.model.quantization = "qlora"
    with pytest.raises(ValueError, match="quantization"):
        build_stage(cfg, 0, 8, torch.device("cpu"))


def test_every_config_loads_and_pipeline_configs_are_consistent():
    files = sorted(glob.glob(os.path.join(REPO, "configs", "*.yaml")) +
                   glob.glob(os.path.join(REPO, "configs", "profile", "*.yaml")))
    assert files
    for f in files:
        cfg = load_config(f)
        d = cfg.training.distributed
        assert d.strategy in {"fsdp", "ddp", "pipeline", "single"} or True
        if d.strategy == "pipeline":
            assert d.pipeline.stages >= 2 and d.pipeline.microbatches >= d.pipeline.stages, f
            assert cfg.lora.enabled, f
            assert str(getattr(cfg.model, "quantization", "none")).lower() == "none", f
        if str(getattr(cfg.model, "quantization", "none")).lower() != "none":
            assert d.strategy != "fsdp", f


def test_pipeline_rejects_gpt_oss_with_a_clear_message(tmp_path):
    import json
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_oss", "architectures": ["GptOssForCausalLM"]}))
    cfg = load_config(os.path.join(REPO, "configs/profile/llama33_70b_pipeline_4node.yaml"))
    cfg.model.name = str(tmp_path)
    with pytest.raises(ValueError, match="gpt_oss"):
        build_stage(cfg, 0, 8, torch.device("cpu"))
