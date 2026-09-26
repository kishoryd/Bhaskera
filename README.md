# Bhaskera — branch `feature/robustness`

This branch adds features for catching failures early and seeing where step time goes on large multi-node runs. It is built on `fsdp-moe-fixes`; see that branch for the MoE fixes and `main` for the full documentation.

## Features

### 1. Startup watchdog — `utils/watchdog.py`
Two earlier runs sat at 0 steps for 3.5 h and 1.4 h without any error. Now each worker logs its startup phase (distributed init → build_model → wrap_model → resume → first batch). If no training step completes within `training.startup_timeout_s` (default 3600, 0 disables), the worker dumps every thread's stack trace and exits with code 86.

### 2. Per-step time breakdown and straggler detection — `trainer/step_timing.py`
Each optimizer step is split into `time/data_wait_s`, `time/compute_s` (forward+backward, including FSDP collectives), `time/optimizer_s` and `time/other_s`, measured with CUDA events and no extra syncs. Every `monitoring.metrics.straggler_every_n_steps` (default 10) steps, rank 0 names the slowest rank and its host. It warns when that rank is more than 1.25× the median and more than 1 s slower. `monitoring.metrics.step_timing: false` turns this off.

### 3. `bhaskera-preflight` — `launcher/preflight.py`
Runs quick CPU-only checks on the login node before you submit:
- model directory and tokenizer load (catches missing `sentencepiece` and similar)
- flash-attn / Liger import
- tokenised caches: they exist and match the config's `seq_len` and model
- config combinations known to fail here
- per-GPU memory estimate, plus the smallest GPU count that fits

```bash
bhaskera-preflight --config configs/qwen3_235b_a22b_fsdp_8node_nemotron.yaml
```

### 4. Node health check and bad-node exclusion — `scripts/`
- Before training, `submit.sh` runs a ~1 minute GPU smoke test on every node (`node_healthcheck.py`: GPU count, bf16 matmul, throughput, cross-GPU checksum, P2P).
- Nodes that fail, hang, or run below 70% of the fastest node are recorded in `.bad_nodes`, and the job exits with code 3.
- `scripts/sbatch_x.sh <sbatch args>` resubmits with the bad nodes merged into the script's own `--exclude`.
- `BHASKERA_PREFLIGHT=0` skips the check.

### 5. Tokenised-cache validation — `data/tokenize.py`
`load_tokenized()` checks the cache's `metadata.json` against the config. A `seq_len` mismatch is an error; a different model is a warning, since models that share a tokeniser can share a cache.

### 6. Runtime sequence packing — `data/packing.py`
`data.runtime_packing: true` packs several examples into each `seq_len` row of the train split. On Nemotron at 4096 tokens, padding drops from 42.7% to 0.1%, so the same tokens fit in 1.74× fewer rows.
- Examples are kept apart by `position_ids` restarting at 0.
- `trainer/loop.py` flattens each micro-batch to `[1, B × S]`, because transformers only uses flash-attention's per-example (varlen) path at batch size 1.
- Requires `model.attn_impl: flash_attention_2`. Eval data stays unpacked.

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
- **Packing:** a packed example's output matches running it alone within bf16 precision (0.0039). Changing one example leaves the others in its pack exactly unchanged. The loss is identical (6.2670). A negative control (batch 2, not flattened) shows the cross-example leak that the flattening prevents.

Each feature is documented further in the READMEs under `src/bhaskera/data/`, `launcher/` and `trainer/`.
