# Branch: `fsdp-moe-fixes`

## Summary

Fixes that make Mixture-of-Experts (MoE) models train correctly under FSDP2, plus a six-model dense-vs-MoE benchmark suite. Before these fixes, Qwen3-235B-A22B ran out of GPU memory at every node count, and Qwen3-30B-A3B failed on its first step.

`feature/hsdp`, `feature/expert-parallel` and `feature/robustness` are built on this branch. Merge this branch first.

## Changes

### MoE training

| Change | Problem solved | Files |
|---|---|---|
| **Checkpoint whole decoder layers for MoE models.** Activation checkpointing now wraps each full decoder layer, as it already did for dense models. | Only the expert MLPs were checkpointed, so attention, LoRA, router and token-dispatch activations were kept for every layer. On Qwen3-235B-A22B this used 65–90 GB per GPU, and adding nodes did not reduce it. | `distributed/activation_ckpt.py` |
| **Reuse the forward pass's expert selection during recompute.** The recompute uses the same top-k experts that the forward pass chose. Routing weights are still recomputed, so gradients are unchanged. | bf16 router scores can differ slightly between the forward pass and the recompute. When two experts score almost equally, the top-k choice can flip, and PyTorch raises `CheckpointError`. | `distributed/moe_routing.py` (new) |
| **Honour `lora.include_experts: false`.** Expert modules are excluded with PEFT `exclude_modules`. | PEFT matches modules by name. Expert layers share the names `gate_proj`, `up_proj` and `down_proj`, so every expert received adapters (about 36,000 on Qwen3-235B-A22B). | `models/lora.py` |
| **Load the model once per job.** Under FSDP, rank 0 loads the checkpoint; the other ranks build the model on the `meta` device. | Every GPU process loaded a full copy into host memory: two copies per node, each about 470 GB for Qwen3-235B-A22B. | `models/loader.py` |
| **Broadcast weights in buckets after sharding.** Rank 0 sends fixed-size buckets, and each rank keeps only its own shard. Peak extra memory is one bucket. | DCP `broadcast_from_rank0` held the full unsharded model on every GPU. | `distributed/fsdp.py` |

### Metrics, data and tooling

| Change | Files |
|---|---|
| **MFU and tokens/s are no longer divided by `world_size` twice.** Reported MFU was too low by a factor of the GPU count: a 16-GPU run showed 0.18% instead of 2.87%. | `utils/throughput.py` |
| **Tokenisation spills to scratch instead of the node-local `/tmp`**, which ran out of space. | `launcher/tokenize.py` |
| **Rows rejected by a strict chat template (Mistral, gpt-oss) fall back to ChatML** instead of aborting the whole corpus. | `data/formats/builtins.py` |
| **The validation set is sharded only when it has at least as many rows as workers.** | `launcher/train.py` |
| **`BHASKERA_PROFILE_OOM=1` prints a CUDA allocator breakdown on out-of-memory.** MLflow now honours `log_gpu_every_n_steps`. | `trainer/loop.py`, `utils/loggers/mlflow_logger.py` |
| **`.gitignore` patterns are anchored to the repository root** (`/models/`, `/data1/`). The old `models/` pattern also excluded `src/bhaskera/models/`. | `.gitignore` |

### Benchmark suite

`configs/*_8node_nemotron.yaml` define six models trained under identical conditions: Nemotron-Cascade-2-SFT-Data (chat), `seq_len` 4096, LoRA r16/α32, 8 nodes × 2 GPUs.

| Model | Parameters | Architecture | Strategy |
|---|---|---|---|
| Qwen3-14B | 14.8B | Dense | DDP |
| Llama-3.3-70B-Instruct | 70B | Dense | FSDP |
| Mistral-Large-Instruct-2411 | 123B | Dense | FSDP |
| Qwen3-30B-A3B-Instruct-2507 | 30B (3B active) | MoE | FSDP |
| gpt-oss-120b | 120B (5.1B active) | MoE | FSDP |
| Qwen3-235B-A22B-Instruct-2507 | 235B (22B active) | MoE | FSDP |

The three Qwen3 models share a tokenizer, so they form a controlled dense-vs-MoE comparison. Each model has a tokenization config (`configs/tokenize_<model>_nemotron.yaml`) and a submit script (`scripts/submit_<model>_<strategy>_8_nemotron.sh`). `scripts/submit_nemotron_pipeline.sh` runs all six.

## Configuration guidance

- `data.seq_len` is part of the tokenization cache key. If you change it, re-tokenize and update `tokenized_path` and `val_tokenized_path`.
- Set `seq_len` to fit the data, not the model's maximum context. At 16384, about 80% of each Nemotron sequence was padding.
- Keep `fsdp.shard_experts_individually: false` for models with 128 experts. Enabling it creates thousands of small FSDP units, and the NCCL collectives hang.

## Validation

```bash
python -m bhaskera.distributed.tests.test_moe_routing
```

CPU, fp32, Qwen3-MoE and Mixtral:
- With injected recompute noise, the unpatched model raises `CheckpointError` and the patched model trains.
- Gradients with activation checkpointing exactly match gradients without it.
- The patched forward pass is bitwise identical to the Hugging Face implementation.

Qwen3-235B-A22B trains at 8, 32 and 64 nodes with these changes.
