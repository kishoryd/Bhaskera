# Bhaskera — branch `feature/hsdp`

This branch adds **HSDP (hybrid-sharded data parallelism)** to FSDP2 training. It is built on `fsdp-moe-fixes`; see that branch for the MoE fixes and `main` for the full documentation.

## What it does

**In plain terms.** Think of the model as a large reference book that every GPU needs to read, one chapter (layer) at a time.

- **`FULL_SHARD` (current default):** the book is torn into as many pieces as there are GPUs in the job, and each GPU keeps one piece. To read a chapter, a GPU must collect that chapter's pieces from *every other GPU*. At 64 nodes that means 128 GPUs, most of them in other nodes across the network, for every chapter, every step. Memory use is as low as possible, but the network becomes the bottleneck.
- **HSDP:** each node gets its own complete copy of the book, torn into pieces only among that node's GPUs. To read a chapter, a GPU collects pieces only from the other GPU in its own node. Nodes talk to each other just once per step, to agree on what was learned (the gradients). Under LoRA that is only the small adapter weights, so it's cheap.

The trade-off is memory: each GPU holds a bigger piece of the book (1/2 of the model on a 2-GPU node instead of 1/128), so HSDP suits models that fit comfortably in one node's GPU memory.

**Technically.** FSDP2 has no `ShardingStrategy` enum; the strategy is the `DeviceMesh` passed to `fully_shard()`.
- `FULL_SHARD` uses a 1D mesh over all ranks (ZeRO-3): every per-layer all-gather and reduce-scatter spans the whole job.
- `HYBRID_SHARD` now builds a 2D mesh `(replicate, shard)`. Parameters are `Shard(0)` within a shard group of `hsdp_shard_size` GPUs and `Replicate` across groups:
  - Per-layer all-gathers and reduce-scatters run only inside the shard group, which is intra-node with the default group size.
  - The cross-group collective is one gradient all-reduce over the replicate dimension per optimizer step, covering only the trainable (LoRA) parameters.

Before this branch, `HYBRID_SHARD` had no effect: it ran as `FULL_SHARD` with only a warning, so configs asking for it got full sharding.

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

At startup, rank 0 logs the mesh and how many nodes each shard group spans. A group that stays inside one node keeps weight traffic off the network entirely.

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

The test trains the same tiny model one step under `FULL_SHARD` and under HSDP, from identical weights and data. Result on A100 (job 80857): **PASS**
1. **Placement:** under `FULL_SHARD` each GPU holds 0.25 of each parameter. Under HSDP it holds 0.5, with mesh `(2, 2)` and placements `(Replicate, Shard)`.
2. **Same result:** the loss is identical (5.564554), and post-step weights differ by at most 3.7e-9. HSDP changes where data lives, not what is computed.
3. **Replicas agree:** both nodes' copies are identical after the optimizer step.
