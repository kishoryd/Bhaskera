# Bhaskera — branch `feature/expert-parallel`

This branch adds **expert parallelism (EP)** for routed MoE models under FSDP2. It is built on `fsdp-moe-fixes`; see that branch for the MoE fixes and `main` for the full documentation.

## What it does

Under plain FSDP, every MoE layer all-gathers **all** experts' weights on every GPU, even though each token uses only its top-k experts. For 128-expert models this dominates step time.

With EP, each GPU keeps a fixed slice of the experts resident and sends tokens to whichever GPU holds their experts:

- Rank `p` in an EP group of size `P` owns experts `[p·L, (p+1)·L)`, where `L = num_experts / P`.
- Tokens go out and come back via two differentiable `all_to_all_single` calls. The traffic is `tokens × top_k × hidden` per direction, instead of every expert's weights.
- Experts are removed from FSDP's view. Non-owned experts sit on `meta` and use no memory.
- Attention, routers and LoRA adapters are still sharded by FSDP as usual.

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
| `distributed/moe_routing.py` | Top-k goes through `select_experts()`, so routing stays pinned under activation-checkpoint recompute. |
| `trainer/loop.py` | Adds the expert parameters back into the MFU count (`model._bhaskera_ep_expert_params`). |
| `config.py` | New `fsdp.expert_parallel_size` (default 1). |
| `distributed/README.md` | Expert-parallelism section. |

## Test

```bash
torchrun --nproc_per_node=2 -m bhaskera.distributed.tests.test_ep_equivalence
```

Result on A100 (job 80849): **PASS**
- Compared against the unmodified HF MoE block in fp32 for Qwen3-MoE (with and without `norm_topk_prob`) and Mixtral: output, router logits, input gradients and router gradients all agree within ≤ 7e-7.
- With activation checkpointing and noise injected into the recompute, results match the run without checkpointing exactly (error 0).
