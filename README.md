# Bhaskera — branch `fsdp-moe-fixes`

This branch fixes MoE training under FSDP2 and adds a six-model dense-vs-MoE benchmark. It lists only what changed relative to `main`; see `main` for the full framework documentation.

`feature/hsdp`, `feature/expert-parallel` and `feature/robustness` are all built on this branch.

## Fixes

| Area | Problem | Fix | File |
|---|---|---|---|
| Activation checkpointing | Qwen3-235B-A22B OOM'd at every node count. Only expert MLPs were checkpointed, so attention, LoRA, router and dispatch activations stayed resident for every layer (~65–90 GB/GPU). | MoE models now checkpoint whole decoder layers. Per-expert checkpointing is kept only as a fallback. | `distributed/activation_ckpt.py` |
| Routing under recompute | With decoder-layer checkpointing, Qwen3-30B-A3B failed on step 1 with `CheckpointError`. The recomputed bf16 router logits are not bitwise identical to the forward's, so a near-tied top-k can pick different experts on recompute. | The recompute reuses the forward's top-k expert indices. Routing weights are still recomputed, so gradients are unchanged. | `distributed/moe_routing.py` (new) |
| LoRA on experts | `lora.include_experts: false` was ignored. PEFT matches by module name, so `gate_proj`/`up_proj`/`down_proj` matched every expert (~36k adapters on 235B). | Experts are excluded with PEFT `exclude_modules`. | `models/lora.py` |
| Model loading | Under FSDP every rank loaded a full copy of the model into CPU RAM. | Only rank 0 loads; other ranks build on `meta`. | `models/loader.py` |
| Weight broadcast | DCP `broadcast_from_rank0` kept the full unsharded model pinned on each GPU. | A bucketed rank-0 broadcast runs after sharding. | `distributed/fsdp.py` |
| MFU / throughput | These were divided by `world_size` twice, so MFU read low by exactly `world_size` (0.18% shown for a true 2.87%). | Corrected. | `utils/throughput.py` |
| Tokenisation | Ray spilled to the node's small `/tmp` (ENOSPC). Strict chat templates (Mistral, gpt-oss) aborted whole corpora. | Spilling now goes to scratch. Rows a template rejects fall back to ChatML instead of aborting. | `launcher/tokenize.py`, `data/formats/builtins.py` |
| Validation | Sharding the val set crashed when it had fewer rows than workers. | The val set is sharded only when rows ≥ workers. | `launcher/train.py` |
| `.gitignore` | `models/` also matched `src/bhaskera/models/`, so new source files there were silently dropped from commits. | The pattern is anchored to the repo root (`/models/`, `/data1/`). | `.gitignore` |
| Diagnostics | – | `BHASKERA_PROFILE_OOM=1` prints an allocator breakdown on OOM. MLflow now honours `log_gpu_every_n_steps`. | `trainer/loop.py`, `utils/loggers/mlflow_logger.py` |

## Tests

```bash
python -m bhaskera.distributed.tests.test_moe_routing   # CPU, fp32, Qwen3-MoE + Mixtral
```

Without the fix, the test reproduces the `CheckpointError`; with it, training runs. Gradients with checkpointing equal gradients without it exactly, and the patched forward is bitwise equal to HF's.

## Benchmark matrix

`configs/*_8node_nemotron.yaml` define six models trained on Nemotron-Cascade-2-SFT-Data (chat) with the same dataset, `seq_len` 4096, LoRA r16/α32 and 8 nodes × 2 GPUs:

| Model | Params | Arch | Strategy |
|---|---|---|---|
| Qwen3-14B | 14.8B | dense | DDP |
| Llama-3.3-70B-Instruct | 70B | dense | FSDP |
| Mistral-Large-Instruct-2411 | 123B | dense | FSDP |
| Qwen3-30B-A3B-Instruct-2507 | 30B / 3B active | MoE | FSDP |
| gpt-oss-120b | 120B / 5.1B active | MoE | FSDP |
| Qwen3-235B-A22B-Instruct-2507 | 235B / 22B active | MoE | FSDP |

The three Qwen3 models share a tokeniser, so they form the controlled dense-vs-MoE comparison. Each model has a `configs/tokenize_<model>_nemotron.yaml` and a `scripts/submit_<model>_<strategy>_8_nemotron.sh`. `scripts/submit_nemotron_pipeline.sh` chains all six.

## Config notes

- `data.seq_len` is part of the tokenised-cache key. Changing it means re-tokenising and updating `tokenized_path` / `val_tokenized_path`.
- Size `seq_len` to the data, not to the model's maximum length. At 16384, about 80% of every Nemotron sequence was padding.
- Keep `fsdp.shard_experts_individually: false` for models with 128 experts. With it on, thousands of tiny FSDP units hung all three MoE models in NCCL collectives.
