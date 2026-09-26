# Bhaskera — branch `feature/robustness`

This branch adds features that catch failures early and show where step time goes on large multi-node runs. It is built on `fsdp-moe-fixes`; see that branch for the MoE fixes and `main` for the full documentation.

## Features

### 1. Startup watchdog — `utils/watchdog.py`

**In plain terms.** Two earlier runs sat at 0 training steps for 3.5 hours and 1.4 hours, burning their node allocations without any error message. Now each worker announces which startup stage it's in. If training hasn't taken its first step within a time limit, the worker records exactly where every part of the program was stuck and stops, instead of hanging silently.

**Technically.**
- Workers log phases: distributed init → build_model → wrap_model → resume checkpoint → first batch.
- A heartbeat names the phase each rank is in.
- If no forward+backward window completes within `training.startup_timeout_s` (default 3600, 0 disables), the worker dumps every thread's Python stack via `faulthandler` and exits with code 86.
- Ray Data's `split(N)` progress bar can sit at 0% while training runs normally, so it is not a stall signal. The first completed step is.

### 2. Per-step time breakdown and straggler detection — `trainer/step_timing.py`

**In plain terms.** When a run is slow, the question is *where* the time goes: waiting for data, computing, updating weights, or waiting on one slow machine. Every step is now split into those parts. Every few steps, all GPUs compare their timings and the slowest machine is named. In a synchronised job one bad node slows everyone down, so it shows up as the one the others wait for.

**Technically.**
- Each optimizer step is split into four metrics:
  - `time/data_wait_s`: blocked on the loader
  - `time/compute_s`: forward+backward, including the FSDP collectives
  - `time/optimizer_s`
  - `time/other_s`
- GPU times come from CUDA events read after the loop's existing `loss.item()` sync, so no extra synchronisation is added.
- Every `monitoring.metrics.straggler_every_n_steps` (default 10) steps, per-rank times are all-gathered. Rank 0 logs the slowest rank and its host, and warns when it is more than 1.25× the median and more than 1 s slower.
- `monitoring.metrics.step_timing: false` turns this off.

### 3. `bhaskera-preflight` — `launcher/preflight.py`

**In plain terms.** Many failures only appeared after a job had waited in the queue and started: a missing Python package, a tokenised dataset that didn't match the config, a model too big for the GPUs requested. This command checks all of that on the login node in seconds, before you submit.

**Technically.** CPU-only checks, exit code 1 on any FAIL:
- model directory and tokenizer load (catches missing `sentencepiece` and similar)
- flash-attn / Liger import against the installed torch
- tokenised caches: they exist and match the config's `seq_len` and model
- config combinations known to fail here (`shard_experts_individually` with ≥ 64 experts)
- per-GPU memory estimate, plus the smallest GPU count that fits. The estimate adds up sharded weights, two gathered layers, decoder-level activation checkpoints, one layer's recompute, logits and 3.5 GiB overhead.

```bash
bhaskera-preflight --config configs/qwen3_235b_a22b_fsdp_8node_nemotron.yaml
```

### 4. Node health check and bad-node exclusion — `scripts/`

**In plain terms.** On a large cluster, one faulty or slow node can ruin a 64-node job. Before training starts, every node now runs a one-minute GPU test. A bad node is recorded and the job stops early. The next submission automatically avoids the recorded nodes.

**Technically.**
- `submit.sh` runs `node_healthcheck.py` on every node before training. It checks GPU count, bf16 matmul finiteness, matmul throughput, a cross-GPU checksum and a local P2P copy.
- A node that errors, hangs past `PREFLIGHT_TIMEOUT_S`, or runs below 70% of the *fastest* node is appended to `.bad_nodes`, and the job exits with code 3.
  - The fastest node is the reference, not the median. With a median reference, a slow node in a 2-node allocation would pass.
- `scripts/sbatch_x.sh <sbatch args>` resubmits with recent bad nodes (newer than `BAD_NODE_TTL_DAYS`, default 7) merged into the script's own `#SBATCH --exclude`. A plain command-line `--exclude` would replace that list rather than add to it.
- `BHASKERA_PREFLIGHT=0` skips the check.

### 5. Tokenised-cache validation — `data/tokenize.py`

**In plain terms.** Training reads a pre-tokenised dataset from disk. If that dataset was built with a different sequence length than the config says, training would run but report the wrong numbers. It now refuses to start in that case.

**Technically.** `load_tokenized()` checks the cache's `metadata.json` against the config.
- A `seq_len` mismatch is an error, because rows are never re-padded.
- A different model is only a warning, because models that share a tokeniser can share a cache.

### 6. Runtime sequence packing — `data/packing.py`

**In plain terms.** Each training row has a fixed length (4096 tokens), but most conversations are shorter, so the rest of the row was padding that the GPU still processes. On Nemotron, 42.7% of all compute went to padding. Packing fits several conversations into each row, like packing a suitcase properly. Padding drops to 0.1%, and the same data needs 1.74× fewer rows. The conversations in a row are kept strictly separate, so none of them can see another.

**Technically.**
- `data.runtime_packing: true` packs the train split first-fit-decreasing, within windows of `data.packing_window_rows` rows (default 2048).
- Example boundaries are `position_ids` restarting at 0. The first label of each example is -100, so the causal shift never predicts across examples.
- `attention_mask` is kept only to count real tokens for throughput/MFU; the model never receives it.
- `trainer/loop.py` flattens each packed micro-batch to `[1, B × S]` and passes `position_ids`. transformers only switches flash-attention to its per-example (varlen) path at batch size 1. At batch size 2, packed examples attend to each other.
- Requires `model.attn_impl: flash_attention_2`. Eval stays unpacked.
- Distinct from `data.pack_sequences`, which only builds masks for caches that are already packed.

## New config keys

| Key | Default |
|---|---|
| `training.startup_timeout_s` | 3600 |
| `monitoring.metrics.step_timing` | true |
| `monitoring.metrics.straggler_every_n_steps` | 10 |
| `data.runtime_packing` | false |
| `data.packing_window_rows` | 2048 |

## Tests

```bash
python -m bhaskera.data.tests.test_packing
python -m bhaskera.trainer.tests.test_step_timing
python -m bhaskera.trainer.tests.test_packing_fa2      # GPU, flash-attn 2
```

Result on A100 (job 80848): **PASS**
- **Step timing:** PASS.
- **Packing:**
  - A packed example gives the same output as running it alone, within bf16 precision (0.0039).
  - Changing one example leaves the others in its row exactly unchanged (difference 0).
  - The loss is identical (6.2670).
  - Negative control: batch size 2 without flattening does leak between examples (difference 1.1). This confirms the test would catch the leak the flattening prevents.

Each feature is documented further in the READMEs under `src/bhaskera/data/`, `launcher/` and `trainer/`.
