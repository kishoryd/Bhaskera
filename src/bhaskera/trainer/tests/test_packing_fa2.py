"""
Packed training rows must be equivalent to running each example alone.
Single GPU with flash-attn; runs under torchrun too (each rank repeats it):
    python -m bhaskera.trainer.tests.test_packing_fa2

Checks, on a tiny randomly initialised Qwen3 model in bf16:
  1. each example's logits inside a packed, flattened [1, B*S] row match the
     example run alone
  2. isolation: changing an earlier example leaves later examples unchanged
  3. negative control: the same packed rows at batch size 2 (not flattened)
     DO leak across examples -- which is why trainer/loop.py flattens
  4. the packed loss equals the token-weighted loss of the examples alone
"""
from __future__ import annotations

import sys
from itertools import pairwise

import numpy as np
import torch

from bhaskera.data.packing import pack_batch

SEQ = 64


def _model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        max_position_embeddings=256, attn_implementation="flash_attention_2",
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def _padded(examples):
    ids, mask = [], []
    for e in examples:
        pad = SEQ - len(e)
        ids.append(np.concatenate([e, np.zeros(pad, np.int64)]))
        mask.append(np.concatenate([np.ones(len(e), np.int64), np.zeros(pad, np.int64)]))
    ids = np.stack(ids)
    return {"input_ids": ids, "attention_mask": np.stack(mask), "labels": np.where(np.stack(mask) == 1, ids, -100)}


def _segment_logits(logits, pos, mask):
    starts = [i for i in range(len(pos)) if pos[i] == 0] + [len(pos)]
    return [logits[a:b] for a, b in pairwise(starts) if mask[a:b].all()]


@torch.no_grad()
def main() -> int:
    model = _model()
    rng = np.random.default_rng(0)
    examples = [rng.integers(1, 512, size=n) for n in (29, 17, 11)]  # packs into one 64-token row
    out = pack_batch(_padded(examples), SEQ)
    assert out["input_ids"].shape[0] == 1
    t = {k: torch.as_tensor(v, device="cuda") for k, v in out.items()}

    def packed_forward(ids, pos, batch2=False):
        if batch2:  # NOT flattened: two copies of the row at batch size 2
            res = model(input_ids=ids.repeat(2, 1), position_ids=pos.repeat(2, 1)).logits[0]
        else:
            res = model(input_ids=ids.reshape(1, -1), position_ids=pos.reshape(1, -1)).logits[0]
        return _segment_logits(res.float(), out["position_ids"][0], out["attention_mask"][0])

    alone = {}
    for e in examples:
        ids = torch.as_tensor(e, device="cuda")[None]
        alone[len(e)] = model(input_ids=ids).logits[0].float()

    ok = True
    segs = packed_forward(t["input_ids"], t["position_ids"])
    err = max((s - alone[len(s)]).abs().max().item() for s in segs)
    ok &= err < 0.05
    print(f"1. packed vs alone: max |logit diff| = {err:.4f}  {'OK' if err < 0.05 else 'FAIL'}")

    first_len = len(segs[0])
    changed = t["input_ids"].clone()
    changed[0, :first_len] = torch.as_tensor(rng.integers(1, 512, size=first_len), device="cuda")
    segs2 = packed_forward(changed, t["position_ids"])
    leak = max((a - b).abs().max().item() for a, b in zip(segs[1:], segs2[1:]))
    ok &= leak < 1e-3
    print(f"2. isolation: later examples change by {leak:.2e} when the first changes  {'OK' if leak < 1e-3 else 'FAIL'}")

    segs_b2 = packed_forward(t["input_ids"], t["position_ids"], batch2=True)
    err_b2 = max((s - alone[len(s)]).abs().max().item() for s in segs_b2[1:])
    leaks = err_b2 > 0.05
    ok &= leaks
    print(f"3. negative control, batch 2 unflattened: max |diff| = {err_b2:.3f}  "
          f"{'OK (leaks, as expected)' if leaks else 'FAIL (expected leakage)'}")

    lab = t["labels"].reshape(1, -1)
    packed_loss = model(input_ids=t["input_ids"].reshape(1, -1),
                        position_ids=t["position_ids"].reshape(1, -1), labels=lab).loss.item()
    num = den = 0.0
    for e in examples:
        ids = torch.as_tensor(e, device="cuda")[None]
        el = ids.clone()
        el[0, 0] = -100
        n = (el[0, 1:] != -100).sum().item()
        num += model(input_ids=ids, labels=el).loss.item() * n
        den += n
    ref = num / den
    ok &= abs(packed_loss - ref) < 0.02
    print(f"4. loss: packed {packed_loss:.4f} vs token-weighted alone {ref:.4f}  "
          f"{'OK' if abs(packed_loss - ref) < 0.02 else 'FAIL'}")

    print("=== PACKING FA2:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
