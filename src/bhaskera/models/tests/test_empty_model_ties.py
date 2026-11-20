"""_build_empty_model keeps tied embeddings tied (rank 0 and the empty-built ranks must match). CPU only."""
from __future__ import annotations

import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from bhaskera.models import loader


def _cfg(tie):
    return LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                       num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=tie)


def _untied_from_config(cfg, **_):
    """Mimics a custom-code model whose from_config leaves the embeddings separate."""
    m = LlamaForCausalLM(cfg)
    m.lm_head.weight = nn.Parameter(m.lm_head.weight.detach().clone())
    return m


def _is_tied(m):
    return m.get_output_embeddings().weight is m.get_input_embeddings().weight


def test_tied_config_is_retied(monkeypatch):
    monkeypatch.setattr(AutoModelForCausalLM, "from_config", staticmethod(_untied_from_config))
    m = loader._build_empty_model(_cfg(tie=True), {})
    assert _is_tied(m)
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in LlamaForCausalLM(_cfg(True)).parameters())


def test_untied_config_stays_untied(monkeypatch):
    monkeypatch.setattr(AutoModelForCausalLM, "from_config", staticmethod(_untied_from_config))
    m = loader._build_empty_model(_cfg(tie=False), {})
    assert not _is_tied(m)
