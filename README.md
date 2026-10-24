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

### Profiling with Nsight Systems (branch `profile/qwen3-14b-nsys`)

Adds a way to record a timeline of a training step with NVIDIA Nsight Systems (`nsys`), to see how much of a step the GPU spends computing, waiting on communication, or idle. Everything is off by default; a normal run is unchanged.

| Change | What it does | Files |
|---|---|---|
| **Labels for each phase of a step.** The training loop marks `data_wait`, `forward_loss`, `backward`, `grad_clip`, `optimizer_step`, `loss_sync`, `throughput_tracker`, `metrics_logging` and `system_stats` with NVTX ranges. The last four cover the code after the optimizer step, which a first test run showed takes a large share of wall time. | The Nsight Systems timeline shows these names, so each part of a step can be timed separately. Gradient communication overlaps with `backward`, so it shows inside that range. | `trainer/loop.py` |
| **A capture window.** Recording starts and stops inside the program (`cudaProfilerStart`/`Stop`) at chosen steps. | Only a couple of steady-state steps are recorded, which keeps the report small and skips start-up and warm-up. | `utils/nvtx_prof.py` (new) |
| **A profiling launch script.** `nsys profile` wraps the Ray start-up, so the training worker processes are traced too. It writes one report per node. | Ray starts the workers, so `nsys` cannot simply wrap the Python command. | `scripts/nsys_train.sbatch` (new) |
| **A short Qwen3-14B config** (8 nodes, DDP, 16 steps). | A run that matches the benchmark setup and finishes quickly. | `configs/profile/qwen3_14b_ddp_8node_nsys.yaml` (new) |
| **Short configs for other setups.** Qwen3-14B at 64 nodes (DDP, 16 steps), and Llama-3.3-70B at 8 nodes (FSDP, 5 steps, capturing only the 4th step because one step takes about 5 minutes). | Profile how communication scales (DDP at 64 nodes) and how FSDP's all-gather and reduce-scatter behave (Llama-70B). | `configs/profile/qwen3_14b_ddp_64node_nsys.yaml`, `configs/profile/llama33_70b_fsdp_8node_nsys.yaml` (new) |
| **Start-up timestamps.** The worker logs one line per milestone (`dist_ready`, `model_loaded`, `model_wrapped`, `first_step_done`) with the time since the previous milestone and since the worker started. | Shows how long start-up took and where, without a profiler. It is always on and costs one GPU sync per milestone. Ray start-up and the driver's data-set build happen before the worker exists, so read those from the driver's own log timestamps. | `utils/startup_timer.py` (new), `launcher/worker.py`, `trainer/loop.py` |

Settings, all environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `BHASKERA_NSYS` | off | Set to `1` to turn on the labels and the capture window. The launch script sets it. |
| `BHASKERA_NSYS_START_STEP` | `12` | First recorded step (counting from 0, so the 13th step). |
| `BHASKERA_NSYS_STOP_STEP` | `14` | Recording stops when this step begins, so two steps are recorded. |
| `BHASKERA_NSYS_FLUSH_SECONDS` | `10` | Pause after the capture ends so every node finishes writing its report. |

Run it:

```bash
sbatch --nodes=8 --partition=gpu-small --job-name=nsys-q14b \
  --export=ALL,SRC_ROOT=$PWD scripts/nsys_train.sbatch configs/profile/qwen3_14b_ddp_8node_nsys.yaml
```

The launch script uses `--capture-range-end=stop-shutdown`: when the capture ends, Nsight Systems writes its report and ends the run, so the job stops after the captured steps. (With plain `stop`, the first test run was killed after the capture, and two of the eight nodes did not write a complete report.) Reports are written to `nsys_reports/<jobid>/<hostname>.nsys-rep`. Open one in the Nsight Systems GUI, or print a summary on the cluster with `nsys stats <file>.nsys-rep`.

Tested once on the cluster (job 96499, Qwen3-14B, 8 nodes) with plain `stop`: the capture opened and closed and six nodes wrote full reports. `stop-shutdown`, the flush delay and the start-up timestamps have not been run yet.

### Kernel-level profiling with Nsight Compute (branch `profile/qwen3-14b-nsys`)

Nsight Systems (above) shows when each kernel runs. Nsight Compute shows how well each kernel uses the GPU: tensor-core use, memory bandwidth, occupancy and stalls. It re-runs every profiled kernel several times, so it is slow and is limited to a small number of kernels in one step.

| File | What it does |
|---|---|
| `scripts/ncu_smoke.sbatch` | A 2-minute check that GPU performance counters can be read by a normal user on this cluster. Run it first. |
| `scripts/ncu_train.sbatch` | Profiles Qwen3-14B DDP on one node (2 GPUs, partition `gpu-debug`). Only matrix-multiply and attention kernels are profiled (`KERNEL_REGEX`, default `gemm|flash|fmha`), at most 40 per process, only in the 13th step. NCCL kernels are deliberately excluded because replaying a collective on one rank would hang the others. |
| `configs/profile/qwen3_14b_ddp_1node_ncu.yaml` | The 1-node DDP config: 1 micro-batch per step, 14 steps. |

Reports are written to `ncu_reports/<jobid>/*.ncu-rep` and opened in the Nsight Compute GUI.

### MFU calculation and sequence packing (branch `profile/qwen3-14b-nsys`)

The reported MFU was wrong in three ways for the Qwen3-14B LoRA benchmark; it is now computed from what the GPU actually does.

| Change | Problem solved | Files |
|---|---|---|
| **FLOPs per token follow the real cost of the backward pass.** Forward is 2 × parameters, backward adds 2 × parameters for activation gradients plus 2 × *trainable* parameters for weight gradients, plus a causal attention term (6 × layers × hidden × sequence length). Full fine-tuning gives the usual 6 × parameters; LoRA with a frozen base gives about 4 × parameters. | The old formula used 6 × parameters for LoRA runs, which overstated the model FLOPs by about 1.5×, and ignored attention, which is about 8% for Qwen3-14B at 4096 tokens. | `utils/throughput.py`, `trainer/loop.py` |
| **MFU counts the tokens the GPU processed** (rows × sequence length, padding included). Tokens per second still counts real tokens only. | Padding costs the same compute as real text. About 43% of each 4096-token row of the Nemotron cache is padding, so leaving it out understated MFU by that share. | `utils/throughput.py`, `trainer/loop.py` |
| **Two new metrics.** `throughput/hfu_pct` also counts the forward pass that activation checkpointing recomputes during backward. `throughput/padding_pct` is the padding share. The step log line adds `HFU=` and `pad=`. | Separates "useful model FLOPs" from "FLOPs the hardware executed", and shows how much a run wastes on padding. | `utils/throughput.py`, `trainer/loop.py` |
| **Runtime sequence packing is available on this branch** (`data.runtime_packing`, from `feature/robustness`). Several examples share one 4096-token row, and flash attention keeps them from attending to each other. | Cuts the padding from about 43% to about 0.1%, so the same compute processes about 1.74× more real tokens. | `data/packing.py`, `launcher/train.py`, `config.py`, `trainer/loop.py` |

Two configs compare the effect on Qwen3-14B DDP at 8 nodes, 20 steps each: `configs/profile/qwen3_14b_ddp_8node_mfu_unpacked.yaml` and `configs/profile/qwen3_14b_ddp_8node_mfu_packed.yaml`.

The attention term uses the full row length, so with packing it slightly overstates the attention work (at most about 4% of the total for Qwen3-14B). The new formula is covered by `utils/tests/test_throughput.py`; the packed training path was not run on this branch yet.

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
