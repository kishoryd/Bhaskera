# Branch: `perf/grouped-experts`

## Summary

Replaces Hugging Face's per-expert loop in routed MoE blocks with one pass that sorts
tokens by expert, gathers once, runs each expert's three GEMMs on its own contiguous
slice, and scatters back once. Output is bitwise identical to HF on deterministic
backends — this is a kernel-launch optimization, not an algorithm change.

Depends on `fsdp-moe-fixes`; merge that branch first.

## The problem

HF's routed MoE block loops over experts and, per expert, calls `nonzero`/`where` (a
GPU→CPU sync) and gathers/scatters its tokens with about 10 small kernels. Profiling
Qwen3-30B-A3B under FSDP with decoder-level activation checkpointing (128 experts × 48
layers, forward + recompute + backward) at batch 8: **~24k syncs and ~342k kernel
launches per micro-step**, and the step was CPU-bound — NCCL kernels mostly sat waiting
on the CPU to issue the next one.

## How it works

For each MoE block's forward (`distributed/moe_routing.py::_grouped_experts`):

1. Sort the flattened top-k expert assignments once (`torch.argsort`, stable).
2. One sync: `torch.bincount` gets each expert's token count.
3. Gather all tokens once via the sort order, then `.split()` into contiguous
   per-expert slices — no Python-level gather/scatter per expert.
4. Run each expert's three GEMMs (gate/up/down) on exactly its own rows. No capacity
   padding: an earlier padded-batched-GEMM version sized every expert to the busiest
   one, and since padding tokens all route to the same few experts, that computed
   ~8x the needed rows.
5. Scatter results back with a single `index_add`.

Row order matches HF's loop exactly (stable sort keeps token order within an expert,
experts accumulate in ascending order), so output is bitwise identical to HF on
deterministic backends. Only applies when every expert is a bias-free SiLU SwiGLU MLP
of plain `nn.Linear` layers (`_grouped_layout()` checks this and falls back to HF's
loop otherwise — e.g. when LoRA adapters are on the experts, since PEFT replaces the
`nn.Linear` with its own wrapper this path would otherwise bypass).

## Usage

No config flag — applies automatically whenever `activation_checkpointing: true` is
set on a supported MoE model (routing goes through the same recompute-stable path from
`fsdp-moe-fixes`). No code changes needed to adopt it.

## Measured effect

Factorial comparison, Qwen3-30B-A3B, single step, forward+backward+optimizer, same
batch size, HF's loop vs. this path:

| Batch size | HF per-expert loop | Grouped (this branch) | Speedup |
|---|---|---|---|
| 1 | 24.92s, 19,380 syncs | 10.64s, 484 syncs | 2.3x |
| 8 | 28.93s, 23,628 syncs | 14.55s, 484 syncs | 2.0x |

**Caveat found later (see `feature/megablocks-moe`):** at production batch sizes
(batch=8-12/GPU, seq_len 4096, ~2,048 tokens/expert on average), the remaining
sequential-GEMM-launch cost is small relative to each GEMM's own compute time — this
fix matters most at low tokens-per-expert (small batch, or many experts relative to
batch size), not as a multiplier on top of this at every scale. It also does not
touch expert-parallel communication cost, which can dominate separately — see
`feature/expert-parallel`'s README.

## Changes

| File | Change |
|---|---|
| `distributed/moe_routing.py` | Adds `_grouped_layout()` (detects a supported SwiGLU expert shape) and `_grouped_experts()` (the sorted-slice forward). `_stable_forward()` dispatches to it when the layout matches. |
| `distributed/tests/test_moe_routing.py` | Extended with grouped-path coverage. |

## Validation

```bash
python -m bhaskera.distributed.tests.test_moe_routing
```

Output matches `_grouped_experts` against HF's own loop exactly (bitwise, deterministic
backends) across random routing assignments, including the capacity-imbalance case
(all tokens routed to one expert).
