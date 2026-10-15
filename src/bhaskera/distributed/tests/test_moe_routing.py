"""
Decoder-layer activation checkpointing on MoE must survive router
nondeterminism between forward and recompute (CPU, fp32):
    python -m bhaskera.distributed.tests.test_moe_routing

Forces the cluster failure by adding noise to router logits during the
recompute only, large enough to flip top-k:
  A. unpatched HF blocks + decoder AC -> torch CheckpointError (the bug)
  B. routing pinned (moe_routing) + same noise -> trains, no stale cache
  C. pinned, no noise -> gradients match training without checkpointing
  D. the patched block's output equals HF's forward exactly
for Qwen3Moe and Mixtral.
"""
from __future__ import annotations

import copy
import sys

import torch
from torch.distributed._composable import checkpoint
from torch.utils.checkpoint import CheckpointError

from bhaskera.distributed import moe_routing as mr


def _models():
    from transformers import MixtralConfig, MixtralForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

    small = {"vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 2,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
             "attn_implementation": "eager"}
    return [
        ("qwen3moe", lambda: Qwen3MoeForCausalLM(Qwen3MoeConfig(
            **small, intermediate_size=64, moe_intermediate_size=16, num_experts=8,
            num_experts_per_tok=2, norm_topk_prob=True))),
        ("mixtral", lambda: MixtralForCausalLM(MixtralConfig(
            **small, intermediate_size=16, num_local_experts=8, num_experts_per_tok=2))),
    ]


def _layers(model):
    return list(model.model.layers)


def _add_recompute_noise(model):
    def hook(_mod, _inp, out):
        if getattr(mr._state, "phase", None) == "recompute":
            return out + 3.0 * torch.randn_like(out)
        return out
    for _, block, _ in mr.routed_moe_blocks(model)[0]:
        block.gate.register_forward_hook(hook)


def _step(model, ids):
    model.zero_grad(set_to_none=True)
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def check(name, make) -> bool:
    torch.manual_seed(0)
    base = make().float().train()
    ids = torch.randint(0, 128, (2, 24))
    ok = True

    m = copy.deepcopy(base)          # A
    _add_recompute_noise(m)
    for layer in _layers(m):
        checkpoint(layer, context_fn=mr.recompute_contexts)
    try:
        _step(m, ids)
        print(f"[{name}] A. unpatched + recompute noise: no error  FAIL (expected CheckpointError)")
        ok = False
    except CheckpointError:
        print(f"[{name}] A. unpatched + recompute noise: CheckpointError (reproduces the bug)  OK")

    m = copy.deepcopy(base)          # B
    mr.make_routing_recompute_stable(m)
    _add_recompute_noise(m)
    for layer in _layers(m):
        checkpoint(layer, context_fn=mr.recompute_contexts)
    _step(m, ids)
    stale = sum(len(b.__dict__.get("_bhaskera_route_cache", ())) for _, b, _ in mr.routed_moe_blocks(m)[0])
    print(f"[{name}] B. pinned + recompute noise: trained, stale cache entries={stale}  "
          f"{'OK' if stale == 0 else 'FAIL'}")
    ok &= stale == 0

    ref = _step(copy.deepcopy(base), ids)   # C
    m = copy.deepcopy(base)
    mr.make_routing_recompute_stable(m)
    for layer in _layers(m):
        checkpoint(layer, context_fn=mr.recompute_contexts)
    got = _step(m, ids)
    err = max((got[k] - ref[k]).abs().max().item() for k in ref)
    same = set(got) == set(ref) and err < 1e-5
    print(f"[{name}] C. pinned AC grads vs no-AC grads: max |diff| = {err:.2e}  {'OK' if same else 'FAIL'}")
    ok &= same

    hf = copy.deepcopy(base).eval()        # D
    pinned = copy.deepcopy(base).eval()
    mr.make_routing_recompute_stable(pinned)
    with torch.no_grad():
        exact = torch.equal(hf(input_ids=ids, use_cache=False).logits, pinned(input_ids=ids, use_cache=False).logits)
    print(f"[{name}] D. patched forward == HF forward (bitwise): {exact}  {'OK' if exact else 'FAIL'}")
    ok &= exact
    return ok


def main() -> int:
    results = [check(n, m) for n, m in _models()]  # run every model, no short-circuit
    ok = all(results)
    print("=== MOE ROUTING PIN:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
