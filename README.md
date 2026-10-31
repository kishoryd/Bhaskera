# Robustness and profiling additions

## Summary

Features for large multi-node runs. They catch failures before or early in a job, show where step time is spent, and remove padding waste from training data.

## Changes

### Startup watchdog (`utils/watchdog.py`)

Stops a job that hangs before training starts, instead of leaving it idle for its full allocation.

- Each worker logs its startup phase: distributed init, model build, model wrap, checkpoint resume, first batch.
- If no training step completes within `training.startup_timeout_s` (default 3600 s; `0` disables it), the worker writes the stack trace of every thread and exits with code 86.

### Step time breakdown and straggler detection (`trainer/step_timing.py`)

Splits each optimizer step into four metrics:

| Metric | Measures |
|---|---|
| `time/data_wait_s` | Time blocked on the data loader |
| `time/compute_s` | Forward and backward pass, including FSDP communication |
| `time/optimizer_s` | Optimizer step |
| `time/other_s` | Remainder |

- GPU times are measured with CUDA events at an existing synchronization point, so the measurement adds no overhead.
- Every `monitoring.metrics.straggler_every_n_steps` steps (default 10), all ranks share their timings.
- Rank 0 logs the slowest rank and its host, and warns when that rank is more than 25% and more than 1 s slower than the median.

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

### FSDP communication: findings and fixes (branch `profile/qwen3-14b-nsys`)

Measured on Llama-3.3-70B (LoRA, FSDP2, 2 x A100 80GB PCIe per node):

* **Network:** one 100 Gb/s InfiniBand card per node, attached to the other CPU socket from both GPUs. NCCL uses InfiniBand, but GPUDirect RDMA stays disabled even with `NCCL_NET_GDR_LEVEL=SYS`, so every byte is staged through host memory. Slurm runs `topology/flat`, so `--switches` placement is not available.
* **Weight all-gathers dominate at scale:** at 128 nodes they take 80% of GPU time (161 per GPU per step, ~308 ms each). Each GPU receives the whole model twice per step whatever the node count, and the per-link rate falls from ~10 GB/s at 16 ranks to ~5.7 GB/s at 256.
* **NCCL settings do not help:** `Simple` (default) is fastest; `LL`/`LL128` are 1.5-2x slower; more channels, connections per link, buffer size and CPU/memory placement make no measurable difference. Run-to-run variation from shared-fabric load is up to ~1.7x (`scripts/allgather_bench.sbatch`).
* **Communication did not overlap compute at all** (0% in the 8-node trace): `CUDA_DEVICE_MAX_CONNECTIONS=1`, a Megatron tensor-parallel convention set by every launcher, serializes FSDP's all-gather stream behind compute. DDP was unaffected (51-67% of its all-reduce hidden).

| Change | Files | Measured effect |
|---|---|---|
| `CUDA_DEVICE_MAX_CONNECTIONS=32` for FSDP (DDP keeps 1), chosen from the config's strategy; override by exporting it | `scripts/submit.sh`, `scripts/submit_*fsdp*.sh`, profiling launchers | Llama-70B, 8 nodes: 125 s -> 107 s per step (4 seq/GPU); 102 s -> 91 s (8 seq/GPU + packing, 34.6% MFU) |
| `fsdp.reshard_after_forward_group: K` keeps a 1/K shard after forward so backward gathers inside blocks of K ranks; `fsdp.keep_gathered_layers: N` keeps the last N layers whole (no backward gather) | `distributed/fsdp.py`, `config.py` | 128 nodes, 1 seq/GPU, together with the line above: 62 s -> 53 s per step |
| One micro-batch per step, as large as fits (8 seq/GPU Llama-70B, 4 Mistral-123B at 8 nodes) plus packing | `configs/profile/*_v2.yaml` | Llama-70B, 8 nodes: 280 s -> 91 s per step; ~1,000 -> 5,760 real tokens/s |
| Bigger global batch at large scale (4 seq/GPU at 128 nodes = 512 sequences/step) | `configs/profile/llama33_70b_fsdp_128node_bs4*.yaml` | 128 nodes: 62 s -> 90 s per step for 4x the tokens (16,700 -> 46,800 tokens/s). Changes the training recipe. |

What is left is the per-step weight volume over a 100 Gb/s link. Beyond the above, reducing it needs a larger batch per step, fewer bytes per weight, or a parallelism that moves activations instead of weights.

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

### Pre-submission checks (`bhaskera-preflight`)

```bash
bhaskera-preflight --config <config.yaml>
```

Runs on the login node in seconds, with no GPU required. It checks:
- the model directory and tokenizer
- flash-attention and Liger imports
- that the tokenized caches exist and match the config
- for config combinations known to fail

It also estimates per-GPU memory and reports the smallest GPU count that fits. It exits with code 1 if any check fails.

### Node health check (`scripts/node_healthcheck.py`, `scripts/sbatch_x.sh`)

- `scripts/submit.sh` runs a one-minute GPU test on every node before training. The test checks GPU count, bf16 matmul correctness and throughput, a cross-GPU checksum, and peer-to-peer copy.
- A node that fails, hangs, or runs below 70% of the fastest node is recorded in `.bad_nodes`, and the job exits with code 3.
- `scripts/sbatch_x.sh` resubmits a job, excluding nodes recorded in the last `BAD_NODE_TTL_DAYS` days (default 7).
- Set `BHASKERA_PREFLIGHT=0` to skip the check.

### Tokenized cache validation (`data/tokenize.py`)

Training checks the cache metadata against the config:
- A `seq_len` mismatch is an error.
- A different model name is a warning, because models that share a tokenizer can share a cache.

### Runtime sequence packing (`data/packing.py`)

`data.runtime_packing: true` packs several training examples into each `seq_len` row.

- **Effect:** on Nemotron at 4096 tokens, padding drops from 42.7% to 0.1%, and the same data needs 1.74× fewer rows.
- **Isolation:** examples in a row never attend to each other.
  - `position_ids` restart at each example boundary.
  - Each micro-batch is flattened to batch size 1, which makes flash-attention treat each example as a separate sequence (variable-length mode).
  - The first label of each example is masked, so no example is trained to predict across a boundary.
- **Requirements:** `model.attn_impl: flash_attention_2`. Only the training split is packed; evaluation is unchanged.

## New configuration keys

| Key | Default |
|---|---|
| `training.startup_timeout_s` | `3600` |
| `monitoring.metrics.step_timing` | `true` |
| `monitoring.metrics.straggler_every_n_steps` | `10` |
| `data.runtime_packing` | `false` |
| `data.packing_window_rows` | `2048` |

## Validation

```bash
python -m bhaskera.data.tests.test_packing
python -m bhaskera.trainer.tests.test_step_timing
python -m bhaskera.trainer.tests.test_packing_fa2      # requires GPU and flash-attention 2
```

Tested on A100:

| Check | Result |
|---|---|
| Step timing | Pass |
| Packed example vs. same example alone | Outputs match within bf16 precision (0.0039) |
| Isolation between packed examples | Changing one example leaves the others exactly unchanged |
| Loss | Identical packed and unpacked (6.2670) |
| Negative control (batch size 2, not flattened) | Leak detected, confirming the test catches it |

More detail is in the READMEs under `src/bhaskera/data/`, `launcher/` and `trainer/`.
