# Branch: `feature/robustness`

## Summary

Features for large multi-node runs. They catch failures before or early in a job, show where step time is spent, and remove padding waste from training data.

Depends on `fsdp-moe-fixes`; merge that branch first.

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
