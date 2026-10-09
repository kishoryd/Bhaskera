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

## Measured communication cost (important before enabling this on PCIe nodes)

Correctness was validated above, but a later `torch.profiler` trace on a real training
run (Qwen3-30B-A3B, `expert_parallel_size: 2`, 8 nodes × 2 A100 80GB PCIe GPUs, no
NVLink between the two GPUs on a node) found that **the EP all-to-all consumed 68.7% of
total GPU time for that step** — dominating over every matmul combined. `nvidia-smi
topo -m` showed the two GPUs on a node connected as `NODE` class (crossing separate
PCIe host bridges, no NVLink), which is the likely reason: EP groups here are formed
from consecutive ranks (`_new_ep_group()`), which land on the same node for
`expert_parallel_size: 2` matched to 2 GPUs/node — so this all-to-all is intra-node,
and its cost comes from the lack of a fast GPU-GPU interconnect, not cross-node network
latency.

**Practical implication:** on nodes without NVLink between GPUs, `expert_parallel_size`
greater than 1 can cost far more in communication than it saves in memory —
`expert_parallel_size: 1` (EP disabled) routes experts through the normal
`fully_shard(layer)` path instead, which shards them across the *entire* world rather
than replicating a 2-way split per EP group, and may cost less memory too, not just
less communication. Benchmark both before committing to EP on PCIe-only clusters.
