# Bhaskera — branch `feature/expert-parallel`

This branch adds **expert parallelism (EP)** for routed Mixture-of-Experts (MoE) models under FSDP2. It is built on `fsdp-moe-fixes`; see that branch for the MoE fixes and `main` for the full documentation.

## What it does

**In plain terms.** An MoE layer is like a hospital with 128 specialists, where each patient (token) sees only the few specialists the receptionist (router) picks for them.

- **Plain FSDP:** every GPU is a small clinic that owns a sliver of every specialist. For each layer, each clinic calls in *all 128 specialists* from across the network, even though its patients need only a handful of them. For 128-expert models, that moving of specialists dominates the step time.
- **Expert parallelism:** the specialists stay put. Each GPU permanently hosts a fixed group of them (with 2 GPUs, experts 0–63 on one and 64–127 on the other). Patients are sent to whichever GPU hosts their specialist, and the results are sent back. Moving patients (a few kilobytes per token) is far cheaper than moving specialists (all expert weights).

Everything that isn't an expert (attention, routers, LoRA adapters) is still handled by FSDP as before.

**Technically.**
- **Ownership.** Rank `p` in an EP group of size `P` owns experts `[p·L, (p+1)·L)`, where `L = num_experts / P`. They are loaded from rank 0 by a bucketed broadcast.
- **Hidden from FSDP.** Experts are removed from the registered module tree, so FSDP, the optimizer and `state_dict()` never see them. Non-owned experts are on `meta` and use no memory.
- **New forward.** Each MoE block's forward is replaced. Routing is computed exactly as HF does it: softmax top-k, renormalised unless `norm_topk_prob` is `False`. Tokens are dispatched to their experts' owners with a differentiable `all_to_all_single`, and results come back with a second one.
  - Communication per layer is `tokens × top_k × hidden` each way, instead of all-gathering every expert's weights.
- **Recompute-safe routing.** Top-k goes through `select_experts()` from `moe_routing.py`, so the expert choice stays pinned under activation-checkpoint recompute (see fix 2 on `fsdp-moe-fixes`).
- **Per-expert sharding.** Per-expert FSDP sharding (`shard_experts_individually`) is skipped.

## Usage

```yaml
training:
  distributed:
    strategy: fsdp
    fsdp:
      expert_parallel_size: 2   # 1 = off (default)
lora:
  include_experts: false        # required: experts must be frozen
```

**Supported:** softmax top-k routed blocks with an `nn.Linear` router at `gate` and an `nn.ModuleList` of experts (Qwen3-MoE, Mixtral).

**Rejected with a clear error:**
- shared-expert blocks
- trainable experts (`lora.include_experts: true`)
- `num_experts` or `world_size` not divisible by `P`

**Checkpoints** don't contain experts. They are frozen pretrained weights, so a resume reloads them from the base model.

## Changes

| File | Change |
|---|---|
| `distributed/expert_parallel.py` (new) | `apply_expert_parallel()`: builds the EP groups, broadcasts expert weights in buckets, drops non-owned experts, and replaces each MoE block's forward with dispatch → local experts → combine. |
| `distributed/fsdp.py` | Calls `apply_expert_parallel()` before `fully_shard()` and skips per-expert sharding. |
| `distributed/moe_routing.py` | Top-k goes through `select_experts()`. |
| `trainer/loop.py` | Adds the non-resident expert parameters back into the MFU count (`model._bhaskera_ep_expert_params`), so MFU stays comparable with plain FSDP. |
| `config.py` | New `fsdp.expert_parallel_size` (default 1). |
| `distributed/README.md` | Expert-parallelism section. |

## Test

```bash
torchrun --nproc_per_node=2 -m bhaskera.distributed.tests.test_ep_equivalence
```

The test runs each MoE block both ways, the original HF block and the expert-parallel version split across 2 GPUs, and compares the results. Result on A100 (job 80849): **PASS**
- **Same results:** for Qwen3-MoE (with and without `norm_topk_prob`) and Mixtral in fp32, the output, router logits, input gradients and router gradients agree within ≤ 7e-7, i.e. fp32 rounding.
- **Checkpoint-safe:** with activation checkpointing and deliberate noise injected into the recompute, results match the run without checkpointing exactly (error 0).
