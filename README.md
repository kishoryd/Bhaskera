# Branch: `feature/megablocks-moe`

## Summary

Integrates Megablocks' block-sparse grouped-GEMM kernels (`grouped_gemm.ops.gmm`) as
an opt-in replacement for `_grouped_experts`' per-expert Python loop, gated behind a
new `moe.use_megablocks` config flag. **Benchmarked at production batch sizes and found
not to help** — the real bottleneck turned out to be elsewhere. Kept as a working,
numerically-verified integration in case it's useful for a different workload (very
small batch, or many more experts relative to tokens).

Built on `fsdp-moe-fixes` + `feature/expert-parallel` + `feature/robustness` (runtime
packing, startup watchdog, step timing, preflight, node health). Depends on all three;
merge those first if this branch is ever merged.

## Motivation

Combined-fixes MFU for Qwen3-30B-A3B on this cluster (A100 80GB PCIe, 2 GPUs/node, no
NVLink) plateaued at ~4-6.4% across 8-64 nodes despite grouped-expert kernels, expert
parallelism, sequence packing, and activation-checkpointing tuning. Megablocks' kernels
are the standard production fix for MoE expert-compute overhead (used in
Mixtral/DeepSpeed-MoE-style stacks), so the hypothesis was that the remaining gap was
still compute-bound.

## What it does

`_grouped_experts_megablocks()` (`distributed/moe_routing.py`) takes the same
expert-sorted token layout `_grouped_experts` already builds (see
`perf/grouped-experts`'s README) and runs each of the three GEMMs (gate/up/down) as
**one grouped-GEMM kernel call across all experts**, instead of one `F.linear` call
per expert in a Python loop. Wired through `fsdp.py` → `activation_ckpt.py` →
`make_routing_recompute_stable()`, which sets `_bhaskera_use_megablocks` per MoE block
from `cfg.moe.use_megablocks`.

```yaml
moe:
  use_megablocks: true
```

## Result: no measurable speedup at production batch sizes

| Scenario | Result |
|---|---|
| Correctness (synthetic data, 8 experts, forward+backward) | Bitwise identical to `_grouped_experts` — forward, input gradient, and expert-weight gradient all match exactly. |
| Benchmark at real dims (128 experts, hidden=2048, batch=8×seq_len=4096) | **1.01x** — no meaningful speedup. |
| Full training run, 16/32/64 nodes, batch=12/GPU | MFU within noise of the non-Megablocks baseline at every node count (e.g. 32n: 4.0-4.2% either way). |

**Why it doesn't help here:** at batch=8-12/GPU with 128 experts and top-8 routing,
each expert averages ~2,048 tokens per micro-step — large enough that each expert's
GEMM already saturates the GPU. The bottleneck Megablocks' kernel fusion targets
(many small, sequential kernel *launches*) isn't present at this scale; it would matter
more at low tokens-per-expert (very small batch, or many more experts relative to
batch size). `perf/grouped-experts`' README has more on this scaling point.

**The actual bottleneck, found by profiling instead of guessing again:** a
`torch.profiler` trace showed 68.7% of GPU time in expert-parallel all-to-all
communication, not any matmul — see `feature/expert-parallel`'s README for the full
finding and `nvidia-smi topo` evidence.

## Build notes (if reviving this)

Requires `torch>=2.7.0` (megablocks' build pin) and `grouped_gemm`/`megablocks`'
native CUDA extensions, which do not install cleanly in the shared Bhaskera venv
(pinned to torch 2.6). This branch was developed against an **isolated venv**
(`.venv/` in this worktree, torch 2.7.1+cu128, torchvision 0.22.1+cu128 — the matching
torchvision build matters, a mismatched one breaks the `transformers`/`peft` import
chain via a stale `torchvision::nms` op registration) — not the shared one other
branches use. The compute nodes on this cluster are also missing standard glibc/libc++
headers at their system paths (`features.h`, `c++/11/*`); building `grouped_gemm`'s and
`megablocks`' CUDA extensions needs `CUDA_HOME` set to a spack-provided CUDA toolkit and
`C_INCLUDE_PATH`/`CPLUS_INCLUDE_PATH` pointed at
`/scratch/nsmapplication/dlapp/wk-kishor/sysroot/usr/include`, both at build time and
at runtime (triton JIT-compiles a small driver file on first import).

## Changes

| File | Change |
|---|---|
| `distributed/moe_routing.py` | Adds `_grouped_experts_megablocks()`; `_stable_forward()` dispatches to it when `block._bhaskera_use_megablocks` is set. |
| `distributed/activation_ckpt.py` | Threads `use_megablocks` through `apply_activation_checkpointing()` → `_moe_context_fn()` → `make_routing_recompute_stable()`. |
| `distributed/fsdp.py` | Passes `cfg.moe.use_megablocks` at the call site. |
| `config.py` | Adds `moe.use_megablocks` (default `false`). |

## Validation

Numerical check (synthetic, `n_tokens=37` deliberately not a multiple of anything,
8 experts, bf16): forward, input gradient, expert-weight gradient all bitwise
identical to `_grouped_experts`. Benchmark and full-scale results above.
