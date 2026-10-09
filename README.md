# Branch: `feature/hsdp`

## Summary

Adds hybrid-sharded data parallelism (HSDP) to FSDP2 training. Model weights are sharded across the GPUs of one node and replicated across nodes. Each layer's weight all-gather then stays inside the node, and only gradients cross the network.

On `main`, `sharding_strategy: HYBRID_SHARD` has no effect and runs as `FULL_SHARD`. This branch implements it.

Depends on `fsdp-moe-fixes`; merge that branch first.

## How it works

| Strategy | Weights on each GPU | Per-layer all-gather spans | Cross-node traffic |
|---|---|---|---|
| `FULL_SHARD` | 1 / total GPUs | Every GPU in the job | Weights and gradients, every layer |
| `HYBRID_SHARD` | 1 / GPUs per group | GPUs in the same group (default: one node) | One gradient all-reduce per step |

Under LoRA, the gradient all-reduce covers only the adapter parameters, so cross-node traffic is small. The cost is memory: each GPU holds a larger share of the model. HSDP therefore suits models whose sharded weights fit within one group, with room left for activations.

FSDP2 expresses the strategy as a `DeviceMesh`. `HYBRID_SHARD` builds a 2D mesh `(replicate, shard)`, and parameters are placed as `(Replicate, Shard(0))`.

## Usage

```yaml
training:
  distributed:
    strategy: fsdp
    fsdp:
      sharding_strategy: HYBRID_SHARD
      hsdp_shard_size: 0      # GPUs per shard group; 0 = GPUs per node
```

- `hsdp_shard_size` must divide the total number of GPUs.
- On 2 × A100 80 GB nodes, one-node groups suit models up to about 30B parameters. Larger models, such as Qwen3-235B-A22B, need larger groups or `FULL_SHARD`.
- At startup, rank 0 logs the mesh shape and how many nodes each shard group spans.

## Changes

| File | Change |
|---|---|
| `distributed/fsdp.py` | `_build_device_mesh()` maps `sharding_strategy` to a device mesh. The mesh is passed to every `fully_shard()` call. Group placement is logged. |
| `config.py` | Adds `fsdp.hsdp_shard_size` (default `0`). |
| `distributed/README.md` | Documents the sharding strategies. |
| `distributed/tests/test_hsdp.py` | New test. |

## Validation

```bash
torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=<head>:29500 \
    -m bhaskera.distributed.tests.test_hsdp
```

Tested on 2 nodes × 2 A100 GPUs, training the same model one step under each strategy:

| Check | Result |
|---|---|
| Shard placement | `FULL_SHARD` holds 1/4 of each parameter per GPU. HSDP holds 1/2, with mesh `(2, 2)`. |
| Numerical equivalence | Identical loss. Weights after the optimizer step differ by at most 3.7e-9. |
| Replica consistency | Both replica groups hold identical weights after the step. |

## Measured throughput at scale

A later apples-to-apples comparison on Qwen3-30B-A3B at 64 nodes (128 A100 80GB PCIe
GPUs) found HSDP's median throughput about **1.9x higher** than `FULL_SHARD` (HSDP
~6,309 tok/s vs. FULL_SHARD ~3,268 tok/s). This differs from an earlier small-scale
test on the same model, which found no HSDP advantage — the benefit appears to grow
with scale, likely because `FULL_SHARD`'s weight all-gather spans the whole job at 64
nodes while HSDP's stays within each node's shard group. Both runs were otherwise
unoptimized (no expert parallelism, grouped-expert kernel, or packing), so absolute MFU
was low (well under 1%) for both — this is a sharding-strategy comparison, not a
throughput ceiling.
