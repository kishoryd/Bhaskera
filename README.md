# Branch: `feature/expert-parallel`

## Summary

Adds expert parallelism (EP) for Mixture-of-Experts models under FSDP2. Each GPU permanently holds a subset of the experts, and tokens are sent to the GPU that owns their selected expert. Without this, every MoE layer gathers all expert weights onto every GPU, even though each token uses only a few experts.

Depends on `fsdp-moe-fixes`; merge that branch first.

## How it works

| | FSDP only | FSDP + expert parallelism |
|---|---|---|
| Expert weights | All experts all-gathered on every GPU, every layer | Each GPU owns `num_experts / P` experts, kept resident |
| Data moved per MoE layer | Every expert's weights | Routed token activations: `tokens × top_k × hidden`, each direction |
| Non-expert weights (attention, router, LoRA) | Sharded by FSDP | Sharded by FSDP (unchanged) |

For each MoE layer:
1. The router selects the top-k experts per token. This matches the Hugging Face implementation exactly.
2. Tokens are dispatched to the owning GPUs with a differentiable `all_to_all_single`.
3. Each GPU runs its local experts.
4. Results are returned with a second `all_to_all_single`.

Experts that a GPU doesn't own are placed on the `meta` device and use no memory. Expert selection goes through the routing helper from `fsdp-moe-fixes`, so it remains stable under activation-checkpoint recompute.

## Usage

```yaml
training:
  distributed:
    strategy: fsdp
    fsdp:
      expert_parallel_size: 2   # P; 1 disables expert parallelism
lora:
  include_experts: false        # required: experts must be frozen
```

- **Supported:** softmax top-k routed MoE blocks with an `nn.Linear` router and an `nn.ModuleList` of experts (Qwen3-MoE, Mixtral).
- **Rejected at startup with a clear error:**
  - shared-expert blocks
  - trainable experts
  - `num_experts` or the world size not divisible by `P`
- **Checkpoints do not include expert weights.** The experts are frozen pretrained weights and are reloaded from the base model on resume.

## Changes

| File | Change |
|---|---|
| `distributed/expert_parallel.py` (new) | `apply_expert_parallel()`: forms EP groups, broadcasts expert weights, removes experts a GPU doesn't own, and replaces each MoE block's forward with dispatch → local experts → combine. |
| `distributed/fsdp.py` | Applies expert parallelism before `fully_shard()`. Skips per-expert FSDP units. |
| `distributed/moe_routing.py` | Adds `select_experts()` for recompute-stable top-k selection. |
| `trainer/loop.py` | Counts all experts in MFU, so figures stay comparable with FSDP-only runs. |
| `config.py` | Adds `fsdp.expert_parallel_size` (default `1`). |
| `distributed/README.md` | Documents expert parallelism. |

## Validation

```bash
torchrun --nproc_per_node=2 -m bhaskera.distributed.tests.test_ep_equivalence
```

Tested on 2 A100 GPUs against the unmodified Hugging Face MoE block in fp32, for Qwen3-MoE (with and without `norm_topk_prob`) and Mixtral:

| Check | Result |
|---|---|
| Outputs, router logits, input gradients, router gradients | Match within 7e-7 (fp32 rounding) |
| With activation checkpointing and injected recompute noise | Exact match with the non-checkpointed run |
