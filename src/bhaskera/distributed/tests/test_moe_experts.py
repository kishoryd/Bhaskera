"""Stacked frozen experts (moe_experts.py) == HF's per-expert loop, for Qwen3-MoE and Mixtral,
in every mode, with and without activation checkpointing. CPU, fp32, tiny models.

    PYTHONPATH=src python src/bhaskera/distributed/tests/test_moe_experts.py
"""
from __future__ import annotations

import copy
import sys

import torch
from torch.utils.checkpoint import checkpoint
from transformers import MixtralConfig, MixtralForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from bhaskera.distributed import moe_routing as mr
from bhaskera.distributed.moe_experts import stack_frozen_experts

SMALL = dict(vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=2, max_position_embeddings=64)


def _models():
    return [
        ("qwen3moe", lambda: Qwen3MoeForCausalLM(Qwen3MoeConfig(
            **SMALL, intermediate_size=64, moe_intermediate_size=16, num_experts=8,
            num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True,
            router_aux_loss_coef=0.0))),
        ("mixtral", lambda: MixtralForCausalLM(MixtralConfig(
            **SMALL, intermediate_size=16, num_local_experts=8, num_experts_per_tok=2,
            router_aux_loss_coef=0.0))),
    ]


def _freeze_experts(m):
    for n, p in m.named_parameters():
        p.requires_grad = not any(f".{a}." in n for a in mr._EXPERT_ATTRS)


def _run(m, ids, ckpt):
    emb = m.model.embed_tokens(ids).detach().requires_grad_(True)
    if ckpt:
        for layer in m.model.layers:
            layer.forward = (lambda f: lambda *a, **k: checkpoint(f, *a, use_reentrant=False,
                                                                  context_fn=mr.recompute_contexts, **k))(layer.forward)
    out = m(inputs_embeds=emb, labels=ids, use_cache=False)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return out.loss.detach(), emb.grad.clone(), grads


def check(name, make) -> bool:
    torch.manual_seed(0)
    base = make().float().train()
    _freeze_experts(base)
    ids = torch.randint(0, 128, (2, 24))
    ref_loss, ref_gx, ref_g = _run(copy.deepcopy(base), ids, ckpt=False)
    ok = True
    for mode in ("loop", "bmm", "auto"):
        for ckpt in (False, True):
            m = copy.deepcopy(base)
            mr.make_routing_recompute_stable(m)
            blocks = mr.routed_moe_blocks(m)[0]
            n = sum(stack_frozen_experts(b, a, mode) for _, b, a in blocks)
            loss, gx, g = _run(m, ids, ckpt)
            dl = float((loss - ref_loss).abs())
            dx = float((gx - ref_gx).abs().max() / ref_gx.abs().max())
            dg = max(float((g[k] - ref_g[k]).abs().max() / ref_g[k].abs().max().clamp_min(1e-12)) for k in ref_g)
            same = n == len(blocks) == 2 and set(g) == set(ref_g) and dl < 1e-5 and dx < 1e-4 and dg < 1e-4
            print(f"[{name}] mode={mode:4s} ckpt={ckpt!s:5s} stacked={n}  loss diff {dl:.1e}  "
                  f"input-grad rel {dx:.1e}  trainable-grad rel {dg:.1e}  {'OK' if same else 'FAIL'}")
            ok &= same
    m = copy.deepcopy(base)                      # trainable experts must not be stacked
    for p in m.parameters():
        p.requires_grad = True
    blk = mr.routed_moe_blocks(m)[0]
    kept = not any(stack_frozen_experts(b, a) for _, b, a in blk)
    print(f"[{name}] trainable experts left as modules: {kept}  {'OK' if kept else 'FAIL'}")
    return ok and kept


def main() -> int:
    results = [check(n, m) for n, m in _models()]
    ok = all(results)
    print(f"=== MOE STACKED EXPERTS: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


def test_moe_experts():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
