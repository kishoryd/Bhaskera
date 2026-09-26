# Bhaskera — branch `feature/hsdp`

This branch adds **HSDP (hybrid-sharded data parallelism)** to FSDP2 training. It is built on `fsdp-moe-fixes`; see that branch for the MoE fixes and `main` for the full documentation.

## What it does

Under `FULL_SHARD`, every layer's weights are all-gathered from **every** GPU in the job, so at 64 nodes each layer's all-gather crosses the network to all 128 GPUs.

HSDP shards the weights within a small group, one node by default, and keeps a full copy of the model in each group:

- The per-layer all-gathers stay inside a node, over NVLink/PCIe rather than InfiniBand.
- The only traffic between groups is one gradient all-reduce per step. Under LoRA this covers only the adapter parameters, so it is tiny.
- The cost is memory: each GPU holds `1 / hsdp_shard_size` of the model instead of `1 / world_size`.

## Usage

```yaml
training:
  distributed:
    strategy: fsdp
    fsdp:
      sharding_strategy: HYBRID_SHARD
      hsdp_shard_size: 0      # GPUs per shard group; 0 = one node (LOCAL_WORLD_SIZE)
```

| `sharding_strategy` | Device mesh | Behaviour |
|---|---|---|
| `FULL_SHARD` (default) | 1D over all ranks | ZeRO-3, unchanged |
| `HYBRID_SHARD` | 2D `(replicate, shard)` | shard within a group, replicate across groups |
| anything else | 1D, with a warning | runs as `FULL_SHARD` |

**Constraints**
- `hsdp_shard_size` must divide `world_size`.
- `model_bytes / hsdp_shard_size` must leave room for activations on each GPU. On 2-GPU A100 80 GB nodes, a one-node group suits models up to ~30B. Qwen3-235B needs larger groups or `FULL_SHARD`.

At startup, rank 0 logs the mesh and how many nodes each shard group spans.

## Changes

| File | Change |
|---|---|
| `distributed/fsdp.py` | `_build_device_mesh()` turns `sharding_strategy` into a mesh. `_log_shard_group_placement()` logs where each group sits. `mesh=` is passed to every `fully_shard()` call. |
| `config.py` | New `fsdp.hsdp_shard_size` (default 0). |
| `distributed/README.md` | Section on sharding strategy and mesh. |
| `distributed/tests/test_hsdp.py` | New unit test. |

## Test

```bash
sbatch .claude_tmp/hsdp_unittest.sbatch   # 2 nodes x 2 GPUs; runs python -m bhaskera.distributed.tests.test_hsdp
```

Result on A100 (job 80857): **PASS**
1. Placement: `FULL_SHARD` holds 0.25 of each parameter per GPU; HSDP holds 0.5 with mesh `(2, 2)` and placements `(Replicate, Shard)`.
2. Loss is identical to `FULL_SHARD` (5.564554), and post-step weights differ by at most 3.7e-9.
3. Both replica groups hold identical weights after the optimizer step.
