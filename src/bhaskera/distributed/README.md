# distributed

How the model is spread over GPUs, and how checkpoints are saved and resumed.

## What you control

Under `training.distributed` in a recipe:

| Key | Meaning |
|---|---|
| `strategy` | `ddp`, `fsdp` or `pipeline`. QLoRA is `model.quantization: qlora` with `ddp` |
| `fsdp.sharding_strategy` | `FULL_SHARD` (default) shards over every GPU; `HYBRID_SHARD` is HSDP |
| `fsdp.hsdp_shard_size` | GPUs per shard group for HSDP. `0` means one node. It must divide the GPU count |
| `fsdp.param_dtype`, `reduce_dtype`, `buffer_dtype` | Mixed precision. `auto` takes the model's dtype, and float32 gradient reduction for MoE |
| `fsdp.shard_experts_individually` | MoE: each expert is its own unit, so only the used experts are gathered |
| `fsdp.activation_checkpointing`, `ddp.activation_checkpointing` | Trade compute for memory |
| `ddp.find_unused_parameters`, `static_graph`, `gradient_as_bucket_view`, `broadcast_buffers` | DDP settings |
| `pipeline.stages` | Pipeline: layers are split into this many stages |

And in the `checkpoint` section: `save_dir`, `save_interval` (counted in epochs for DDP, QLoRA and FSDP, and in steps for Pipeline), and `keep_last_n`. A relative `save_dir` is resolved against the folder you submitted from.

Which strategy to choose is in the main README.

## What Bhaskera does itself

- **FSDP2.** Sharding goes in three passes: experts (if asked), decoder layers, then the root. HSDP builds a 2-D device mesh of (replicate, shard), and rank 0 logs how many nodes a shard group spans; a group inside one node keeps weight gathers off the network. Mixed precision lives in FSDP's policy, so the loop does not add autocast.
- **DDP.** MoE models always get `find_unused_parameters=True`, because routing touches different parameters each step, and `static_graph` is switched off with it. Activation checkpointing is applied before the wrap.
- **Activation checkpointing.** One unit per decoder layer. Only a MoE model with no detectable layer class falls back to one per expert, with a warning.
- **Start-up guards.** A rank waits for `torch.distributed` to come up, and a probe all-reduce confirms every rank is alive, so a missing rank fails loudly instead of hanging later.
- **Checkpoints** (`checkpoint.py`). Sharded with torch's distributed checkpoint. All ranks write to `step_N.tmp`; after a barrier rank 0 renames it, writes `meta.json`, and writes an empty `.complete` file last. A folder without `.complete` is ignored.
- **Keeping the newest.** After a save the newest `keep_last_n` completed checkpoints stay (`0` keeps all); the newest is never removed, and checkpoints left by an earlier run of the same job count.
- **Resuming.** The latest complete checkpoint is loaded into the model and optimizer, and the learning-rate schedule is replayed to that step. The data iterator starts again from the beginning of the training set.
- **Pipeline** (`pipeline.py`) splits the frozen model by layers; only activations cross a stage boundary, and only adapter gradients are averaged across copies.
