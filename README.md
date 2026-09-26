# Bhaskera — branch `fsdp-moe-fixes`

This branch fixes Mixture-of-Experts (MoE) training under FSDP2 and adds a six-model dense-vs-MoE benchmark. It covers only what changed relative to `main`; see `main` for the full framework documentation.

`feature/hsdp`, `feature/expert-parallel` and `feature/robustness` are all built on this branch.

## Background: what an MoE model does differently

A dense model sends every token through the same feed-forward network in each layer. An MoE model has many feed-forward networks ("experts") per layer: 128 in Qwen3-30B-A3B and Qwen3-235B-A22B, 8 in Mixtral. A small **router** picks the best few experts for each token (the top-k, e.g. 8 of 128), and only those run. That's why Qwen3-235B-A22B has 235B parameters but uses only about 22B per token.

Most of the fixes below come from this difference. Code written with dense models in mind silently misbehaves when layers contain hundreds of experts and a routing decision.

## Summary

| # | Fix | Symptom before | File |
|---|---|---|---|
| 1 | Checkpoint whole decoder layers for MoE | Qwen3-235B-A22B out of memory at every node count | `distributed/activation_ckpt.py` |
| 2 | Pin expert routing during recompute | Qwen3-30B-A3B crashed on step 1 with `CheckpointError` | `distributed/moe_routing.py` (new) |
| 3 | Keep LoRA off experts when `include_experts: false` | ~36k unintended adapters on 235B | `models/lora.py` |
| 4 | Load the model once per job, not once per GPU | Every GPU process read a full copy into CPU RAM | `models/loader.py` |
| 5 | Broadcast weights in buckets after sharding | Broadcast kept the full model on each GPU | `distributed/fsdp.py` |
| 6 | Correct MFU and throughput | MFU shown `world_size` times too low | `utils/throughput.py` |
| 7 | Tokenisation robustness | Disk-full crashes, whole corpus aborted by one bad row | `launcher/tokenize.py`, `data/formats/builtins.py` |
| 8 | Validation-set sharding | Crash when val set had fewer rows than workers | `launcher/train.py` |
| 9 | `.gitignore` anchoring | New files in `src/bhaskera/models/` silently not committed | `.gitignore` |
| 10 | Diagnostics | No way to see what filled GPU memory on OOM | `trainer/loop.py`, `utils/loggers/mlflow_logger.py` |

## The fixes

### 1. Activation checkpointing: checkpoint whole decoder layers for MoE

**In plain terms.** Training has to remember intermediate results from the forward pass to compute gradients in the backward pass. Activation checkpointing saves memory by keeping only each layer's input and redoing that layer's work when the backward pass needs it. It's like keeping just the question for each exercise and re-solving it when marking, instead of keeping all the rough work. For MoE models, Bhaskera applied this only to the experts. Everything else in every layer (attention, LoRA, the router, the step that sends tokens to experts) kept all its rough work. On Qwen3-235B-A22B that alone filled the GPU. Adding more nodes didn't help, because this memory is per GPU and doesn't shrink as the job grows.

**Technically.** With expert-level checkpointing, attention, LoRA, router and dispatch activations stayed resident for all 94 layers, about 65–90 GB per GPU on Qwen3-235B-A22B. That is why the OOM didn't change with world size. MoE models now checkpoint at the `decoder_layer_cls` level, like dense models. Per-expert checkpointing remains only as a fallback when no decoder layer class is found. With this change, Qwen3-235B-A22B trained at 8, 32 and 64 nodes.

### 2. Pin expert routing during recompute

**In plain terms.** Fix 1 means each layer is run twice: once in the forward pass and again during backward. For the second run to be valid it must do exactly what the first run did. In an MoE layer the router picks experts by score. When two experts score almost exactly the same, a tiny rounding difference between the two runs can make the router pick a different expert the second time. The redone work then no longer matches what was saved, and PyTorch stops training with an error. The fix records which experts the first run picked and makes the rerun use the same ones.

**Technically.** HF routed MoE blocks only execute experts that received tokens, so the number of tensors saved for backward depends on the top-k result. Router logits come from bf16 GEMMs, which aren't bitwise reproducible between forward and recompute. A near-tied top-k therefore flipped, and torch raised `CheckpointError: a different number of tensors was saved` on step 1 of Qwen3-30B-A3B.

`moe_routing.py` replaces the routed block's forward with an exact reproduction of HF's (Qwen3-MoE and Mixtral: softmax, top-k, optional `norm_topk_prob` renormalisation). It uses the checkpoint `context_fn` to mark forward and recompute phases: the forward caches the top-k expert indices and the recompute reuses them. Expert choice is discrete and has no gradient, so reusing it is exact. The routing weights are still recomputed through a differentiable `gather`, so gradients are unchanged. Top-k runs on detached probabilities in both phases, so it never adds a saved tensor.

### 3. Keep LoRA off experts when `lora.include_experts: false`

**In plain terms.** LoRA fine-tunes a model by attaching small trainable add-ons ("adapters") to chosen layers, found by name. The attention and feed-forward layers are named `gate_proj`, `up_proj`, `down_proj`, and every one of the 128 experts in each layer uses the same names. So the setting that says "don't put adapters on experts" was ignored, and adapters were attached to every expert. That meant tens of thousands of unintended adapters, more memory and optimizer state, and a different experiment from the one configured.

**Technically.** PEFT matches `target_modules` by module-name suffix, so auto targets also matched every expert's linears. `include_experts: false` only stopped expert names from being *added* to the target list. Experts are now excluded with PEFT `exclude_modules`, using a full-match regex over the expert container names found by introspection. On a tiny Qwen3-MoE, 48 of 56 adapters had been inside experts; with the fix there are 0 (48 with `include_experts: true`). Dense models are unchanged (14).

### 4. Load the model once per job, not once per GPU

**In plain terms.** Previously every GPU process read the whole model from disk into CPU memory, so a 2-GPU node held two full copies. For Qwen3-235B-A22B one copy in bf16 is about 470 GB. Now only the first process loads the real weights. The others build an empty skeleton of the model and receive their share of the weights later (fix 5).

**Technically.** Under FSDP, non-zero ranks now build the model on the `meta` device. Rank 0 loads it from the checkpoint. The loader also warns loudly when an MXFP4 checkpoint would be silently dequantised to bf16.

### 5. Broadcast weights in buckets after sharding

**In plain terms.** Once each GPU has its skeleton, the real weights must be sent out from the first GPU. The previous method sent the whole model to every GPU in one go, which doesn't fit on an 80 GB GPU for the large models. The weights are now sent in fixed-size batches, and each GPU keeps only its own share.

**Technically.** DCP's `broadcast_from_rank0` held the full unsharded model on every GPU through views. After `fully_shard`, `fsdp.py` now broadcasts from rank 0 in fixed-size buckets, and each rank copies out only its local shard of each `DTensor`. Peak extra memory is one bucket instead of the whole model.

### 6. Correct MFU and throughput

**In plain terms.** MFU (model FLOPs utilisation) is the share of the GPUs' theoretical peak speed that training actually uses. A calculation bug divided the result by the number of GPUs twice, so every multi-GPU run looked far slower than it was.

**Technically.** The loop passes this rank's token count, which is already per GPU, and the meter divided by `world_size` again. MFU read low by exactly `world_size`: a 16-GPU run showed 0.18% for a true 2.87%. Job-wide figures now scale up from the per-rank rate instead of all-reducing every step.

### 7. Tokenisation robustness

**In plain terms.** Two things broke long tokenisation jobs. Ray's overflow files went to the compute node's small local disk, which filled up. And some models' chat templates (Mistral, gpt-oss) reject certain conversation shapes by raising an error, which aborted the whole dataset. Overflow now goes to the large scratch filesystem. A row a template rejects is formatted with the standard ChatML format instead, so every model trains on the same rows.

**Technically.** Ray object spilling is redirected via `object_spilling_config`. `_temp_dir` isn't used because it also moves Ray's Unix sockets and exceeds the 107-byte `AF_UNIX` path limit on a scratch path. In `data/formats/builtins.py`, a Jinja `raise_exception()` from a strict template falls back to manual ChatML for that row. Dropping the row instead would give each model a different training set and invalidate the comparison.

### 8. Validation-set sharding

**In plain terms.** Each GPU gets its own slice of the validation set. With a very small validation set and many GPUs, some slices came out empty and training crashed.

**Technically.** `launcher/train.py` now shards the val dataset only when its row count is at least the number of workers.

### 9. `.gitignore` anchoring

**In plain terms.** The rule meant to ignore the large model-weights folder `models/` also matched the source-code folder `src/bhaskera/models/`. So new code files there were silently left out of every commit.

**Technically.** The patterns are anchored to the repository root (`/models/`, `/data1/`). The weights and datasets at the root are still ignored.

### 10. Diagnostics

- `BHASKERA_PROFILE_OOM=1` prints an allocator breakdown when a GPU runs out of memory. This is how fix 1's root cause was found.
- MLflow runs now honour `log_gpu_every_n_steps`.

## Tests

```bash
python -m bhaskera.distributed.tests.test_moe_routing   # CPU, fp32, Qwen3-MoE + Mixtral
```

The test forces noise into the recompute to simulate the rounding difference in fix 2:
- Without the fix it reproduces the `CheckpointError`; with it, training runs.
- Gradients with checkpointing exactly equal gradients without it.
- The patched forward is bitwise equal to HF's original.

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
