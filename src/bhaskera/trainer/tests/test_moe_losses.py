"""MoE loss plumbing: router matching, HF's built-in aux term, router z-loss. CPU only."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from bhaskera.models.lora import is_router_param, router_frozen
from bhaskera.trainer.moe import disable_builtin_aux_loss, router_z_loss, router_z_loss_from_output


def _tiny(**kw):
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, moe_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        decoder_sparse_step=1, mlp_only_layers=kw.pop("mlp_only_layers", []), router_aux_loss_coef=0.02, **kw))


def _profile(names=()):
    return SimpleNamespace(router_module_names=list(names), is_moe=True, num_experts=8,
                           num_shared_experts=0, experts_per_token=2)


def test_router_param_matching_is_parent_scoped():
    prof = _profile(["model.layers.0.mlp.gate"])
    assert is_router_param("model.layers.0.mlp.gate.weight", prof)
    assert is_router_param("layers.3.mlp.gate.weight", prof)              # pipeline stage-local name
    assert is_router_param("model.layers.1.mlp.shared_expert_gate.weight", prof)
    for n in ("model.layers.0.mlp.gate_proj.weight",                       # dense / shared MLP
              "model.layers.0.mlp.gate_proj.lora_A.default.weight",
              "model.layers.0.mlp.experts.3.gate_proj.weight",
              "model.layers.0.mlp.gate.lora_A.default.weight",             # LoRA on a router stays LoRA
              "model.layers.0.self_attn.q_proj.weight"):
        assert not is_router_param(n, prof), n


def test_dense_mlp_gate_proj_lora_not_frozen_count():
    m = _tiny(mlp_only_layers=[0])                                         # layer 0 dense SwiGLU
    prof = _profile()
    routers = [n for n, _ in m.named_parameters() if is_router_param(n, prof)]
    assert routers == ["model.layers.1.mlp.gate.weight"]


def test_router_frozen_either_key():
    c = lambda a, b: SimpleNamespace(lora=SimpleNamespace(freeze_router=a), moe=SimpleNamespace(freeze_router=b))
    assert router_frozen(c(True, True))
    assert not router_frozen(c(False, True))
    assert not router_frozen(c(True, False))


def test_builtin_aux_term_removed_from_loss():
    torch.manual_seed(0)
    m = _tiny().eval()
    ids = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        plain = m(input_ids=ids, labels=ids).loss
        with_aux = m(input_ids=ids, labels=ids, output_router_logits=True)
        assert with_aux.loss > plain                                       # HF adds 0.02 x aux
        assert disable_builtin_aux_loss(m) == 1
        after = m(input_ids=ids, labels=ids, output_router_logits=True)
    torch.testing.assert_close(after.loss, plain)
    assert after.aux_loss is not None and float(after.aux_loss) > 0       # still reported for our own weight


def test_router_z_loss():
    lg = torch.randn(10, 8)
    ref = torch.logsumexp(lg, -1).pow(2).mean()
    torch.testing.assert_close(router_z_loss(lg), ref)
    mask = torch.tensor([1] * 6 + [0] * 4)
    torch.testing.assert_close(router_z_loss(lg, mask), torch.logsumexp(lg[:6], -1).pow(2).mean())
    out = SimpleNamespace(router_logits=(lg, lg * 2))
    want = (ref + torch.logsumexp(lg * 2, -1).pow(2).mean()) / 2
    torch.testing.assert_close(router_z_loss_from_output(out, _profile()), want)
