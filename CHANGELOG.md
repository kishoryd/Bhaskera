# Changelog

All notable changes to Bhaskera are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/): **MAJOR** for config or API changes that break existing
runs, **MINOR** for new features that keep existing configs working, **PATCH** for fixes only.
How to cut a release: [RELEASING.md](RELEASING.md).

## [Unreleased]

## [1.0.0] - 2026-11-20

First release. Earlier development builds carried version numbers 2.1.0 to 2.2.0 in `pyproject.toml` but
were never tagged, so numbering starts again at 1.0.0.

Benchmarked on PARAM Rudra (2 × A100 80 GB per node) with Qwen3-14B, Llama-3.3-70B, Mistral-Large-123B,
Qwen3-30B-A3B and Qwen3-235B-A22B on 8 to 128 nodes. BharatGen Param-1-2.9B, Param-1-7B and Param2-17B
are supported.

### Added
- **Parallelism:** DDP, FSDP2, HSDP (`HYBRID_SHARD` on a two-dimensional mesh), QLoRA + DDP (4-bit NF4
  frozen base, `model.quantization: qlora`) and a Pipeline strategy (1F1B and GPipe) for dense and MoE
  models.
- **MoE:** frozen routers, load-balancing auxiliary loss, router z-loss and active-parameter MFU. In the
  pipeline, frozen experts are stacked and computed with one host sync per layer
  (`pipeline.moe_experts: auto | loop | bmm | hf`), and routing is pinned across checkpoint recompute.
- **Data:** persistent tokenisation cache (`bhaskera-tokenize`), runtime sequence packing
  (`data.runtime_packing`), chat formats `chatml`, `alpaca` and `sharegpt`, and custom formats.
- **Evaluation:** held-out loss and next-token accuracy; `training.eval_max_rows`.
- **Monitoring:** one log line per step with tokens/s, MFU, peak GPU memory, step-time breakdown and
  straggler detection; MLflow; startup watchdog; node-health preflight.
- **Preflight** (`python -m bhaskera.launcher.preflight`): checks model, libraries, data and GPU-memory fit
  on the login node before a job is queued.
- **`slurm/` kit:** `train.sbatch`, `tokenize.sbatch`, `submit.sh` (picks the partition and time limit,
  runs the preflight), eleven recipes (eight for the five benchmark models, three for BharatGen models),
  `make-quickstart.py` (tokenises once and writes ready recipes for a shared model folder),
  `install-bhaskera.sh`, an Lmod modulefile, and `requirements-paramrudra.lock` with the exact package
  versions of the benchmark environment.
- `scripts/convert_mxfp4_to_bf16.py`: converts an MXFP4 checkpoint (gpt-oss) to bf16 on disk.

### Changed
- DDP, QLoRA and FSDP checkpoints hold only the trained weights and the optimizer state, not the frozen
  base model. Qwen3-14B QLoRA: 10 GB before, 942 MB now. Gathering the 4-bit weights of
  Mistral-Large-123B for saving exceeded the host memory of a node.
- A relative `checkpoint.save_dir` is resolved against the folder the job was submitted from.
- Preflight memory estimates account for 4-bit weights and for the per-stage split in pipeline runs, and
  are accurate to about 15% of the measured peak.
- The version written into tokenisation-cache metadata comes from the installed package.

### Fixed
- Resuming a DDP, QLoRA or FSDP run always restarted at step 0: the trainer passed the newest `step_N`
  folder to a function that expects the folder containing the `step_N` folders. The learning-rate schedule
  is now rebuilt at the resumed step, and optimizer settings that the state load leaves out (`betas`, `eps`,
  `lr`, `weight_decay`) are restored from the configured values. Pipeline resume was not affected.
- FSDP start-up hung for models with tied embeddings and custom code (BharatGen Param2-17B): the ranks that
  build the model empty did not tie the embeddings, so they held one more parameter than rank 0 and the
  weight broadcast never completed.
- Tokenising a model whose tokenizer defines no special tokens (BharatGen Param-1-7B) failed with "tokenizer
  does not have a padding token". The pad token now falls back to the id in `config.json`.
- Custom-code models that import `transformers.utils.LossKwargs` (removed in transformers 4.56) failed to
  load; the name is re-created when it is missing.
- MoE: FSDP counted the auxiliary loss twice; `gate_proj` LoRA weights were matched as routers;
  `freeze_router: false` was ignored by the pipeline; pipeline stage outputs were views, which broke the
  auxiliary-loss path; one routing store is now kept per checkpointed call.
- FSDP2: the gradient-sync toggle had no effect; non-persistent buffers (RoPE `inv_freq`) were broadcast
  with wrong values; races in distributed init and checkpoint saving.

### Known limitations
- After a resume the data iterator starts again from the beginning of the training set.
- MoE models are slow outside the Pipeline strategy (0.2–1.7% MFU). The BharatGen MoE models run under FSDP
  only; Pipeline does not support their custom code.
- DDP needs the whole model on each GPU, so 70 B and larger need QLoRA or Pipeline. Qwen3-235B does not fit
  with QLoRA. Mistral-Large-123B QLoRA needs `batch_size: 1`.
- BharatGen Param-1-2.9B supports neither flash-attention nor PyTorch attention, so it runs with `eager`
  attention and without packing.
- Training gpt-oss (MXFP4) models is not supported.

[Unreleased]: https://github.com/kishoryd/Bhaskera/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kishoryd/Bhaskera/releases/tag/v1.0.0
