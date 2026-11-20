# Changelog

All notable changes to Bhaskera are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/): **MAJOR** for config or API changes that break existing
runs, **MINOR** for new features that keep existing configs working, **PATCH** for fixes only.
How to cut a release: [RELEASING.md](RELEASING.md).

## [Unreleased]

## [1.0.0] - 2026-11-20

First release. Bhaskera was developed internally before this (its `pyproject.toml` read 2.1.0–2.2.0, never
tagged); numbering restarts here as the first supported version.

Validated on PARAM Rudra (2 × A100 80 GB per node) with five models from 14 B to 235 B parameters
(Qwen3-14B, Llama-3.3-70B, Mistral-Large-123B, Qwen3-30B-A3B, Qwen3-235B-A22B), every strategy, on
8–128 nodes (125 runs), and with three BharatGen models (Param-1-2.9B, Param-1-7B, Param2-17B).

### Added
- **Parallelism:** DDP, FSDP2, HSDP (`HYBRID_SHARD` on a 2-D mesh), **QLoRA + DDP** (4-bit NF4 frozen base,
  `model.quantization: qlora`) and a **Pipeline strategy** (1F1B / GPipe) for dense and MoE models.
- **MoE:** frozen routers, load-balancing auxiliary loss, router z-loss, active-parameter MFU; in the
  pipeline, frozen experts are stacked and computed with one host sync per layer
  (`pipeline.moe_experts: auto | loop | bmm | hf`) and routing is pinned across checkpoint recompute.
- **Data:** persistent tokenisation cache (`bhaskera-tokenize`), runtime sequence packing
  (`data.runtime_packing`), chat formats `chatml`, `alpaca`, `sharegpt` and custom ones.
- **Evaluation:** held-out loss and next-token accuracy; `training.eval_max_rows`.
- **Observability:** one log line per step with tokens/s, MFU, peak GPU memory, step-time breakdown and
  straggler detection; MLflow; startup watchdog; node-health preflight.
- **Preflight** (`python -m bhaskera.launcher.preflight`): login-node checks of model, libraries, data and
  GPU-memory fit before a job is queued.
- **`slurm/` kit:** `train.sbatch`, `tokenize.sbatch`, `submit.sh` (partition and time-limit picker,
  preflight), eleven tested recipes (eight for the five benchmark models, three for BharatGen models),
  `make-quickstart.py` (tokenises once and writes ready recipes for a shared model folder),
  `install-bhaskera.sh`, an Lmod modulefile and `requirements-paramrudra.lock` (the exact package versions
  of the benchmark environment).
- `scripts/convert_mxfp4_to_bf16.py`: convert an MXFP4 checkpoint (gpt-oss) to bf16 on disk.

### Fixed (found while preparing this release)
- **Resume never worked for DDP, QLoRA and FSDP runs.** The trainer passed the newest `step_N` folder to a
  function that expects the folder containing the `step_N` folders, so every resubmitted job restarted at
  step 0. It now resumes, and the learning-rate schedule continues from the resumed step. (Pipeline had its
  own resume, which worked.)
- **DDP, QLoRA and FSDP checkpoints stored the whole model**, including the frozen base weights: 10 GB for
  Qwen3-14B QLoRA, 40 GB for Llama-3.3-70B QLoRA, and Mistral-Large-123B ran the node out of host RAM while
  saving, leaving an empty folder. Checkpoints now hold only the trained weights and optimizer state
  (Qwen3-14B QLoRA: 10 GB → 942 MB; Mistral-Large-123B: saves, 5 GB).
- **A relative `checkpoint.save_dir` silently lost every checkpoint:** workers write into their own
  temporary working directory. The launcher now resolves it to an absolute path.
- **FSDP start-up deadlock for models with tied embeddings and custom code** (BharatGen Param2-17B): ranks
  that build the model empty did not tie the embeddings, so they held one more parameter than rank 0 and the
  weight broadcast waited forever.
- Tokenising a model whose tokenizer defines no special tokens (BharatGen Param-1-7B) failed with "tokenizer
  does not have a padding token"; the pad token now falls back to the id in `config.json`.
- Custom-code models importing `transformers.utils.LossKwargs` (removed in transformers 4.56) failed to
  load; a compatibility shim re-creates it when missing.
- Preflight estimated QLoRA memory at bf16 size and pipeline memory per GPU count instead of per stage;
  QLoRA recipes were rejected (Llama-3.3-70B: 147.7 GiB estimated vs 55 GB measured). Estimates are now
  within 13% of the measured peak on all seven recipes checked.
- MoE: FSDP counted the auxiliary loss twice; router matching treated `gate_proj` LoRA as a router;
  `freeze_router: false` was ignored by the pipeline; pipeline stage outputs were views (aux-loss crash);
  one routing store per checkpointed call (CheckpointError with micro-batches in flight).
- FSDP2: the gradient-sync toggle was a no-op; non-persistent buffers (RoPE `inv_freq`) were broadcast as
  garbage; dist-init and checkpoint rank races.

### Known limitations
- MoE models are slow outside the Pipeline strategy (0.2–1.7% MFU); BharatGen MoE models run under FSDP
  only (Pipeline is not validated for their custom code).
- DDP needs the whole model on each GPU; 70 B and larger need QLoRA or Pipeline. Qwen3-235B does not fit
  with QLoRA. Mistral-Large-123B QLoRA needs `batch_size: 1`.
- BharatGen Param-1-2.9B supports neither flash-attention nor PyTorch attention, so it runs with `eager`
  attention and without packing (74% of each batch is padding).
- gpt-oss (MXFP4) training is not validated.

[Unreleased]: https://github.com/kishoryd/Bhaskera/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kishoryd/Bhaskera/releases/tag/v1.0.0
