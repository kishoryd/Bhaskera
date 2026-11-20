# Changelog

All notable changes to Bhaskera are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/): **MAJOR** for config or API changes that break existing
runs, **MINOR** for new features that keep existing configs working, **PATCH** for fixes only.
How to cut a release: [RELEASING.md](RELEASING.md).

## [Unreleased]

## [2.3.0] - 2026-11-19

Validated on PARAM Rudra (2 × A100 80 GB per node) with five models from 14 B to 235 B parameters,
every strategy, on 8–128 nodes (125 runs).

### Added
- **Pipeline strategy** (`training.distributed.strategy: pipeline`) for dense and MoE models: 1F1B and
  GPipe schedules, periodic evaluation, checkpoint resume, LoRA kept in sync across pipeline copies.
- **QLoRA + DDP**: 4-bit NF4 frozen base model (`model.quantization: qlora`) so a 70 B–123 B model fits
  on one 80 GB GPU; `qlora` extra declares bitsandbytes.
- **HSDP**: `fsdp.sharding_strategy: HYBRID_SHARD` mapped to a 2-D FSDP2 mesh (`hsdp_shard_size`).
- **MoE in the pipeline**: frozen experts stacked and computed with one host sync per layer
  (`pipeline.moe_experts: auto | loop | bmm | hf`); routing pinned across activation-checkpoint recompute;
  router z-loss; active-parameter MFU.
- **Evaluation accuracy**: next-token accuracy next to eval loss (FSDP/DDP and pipeline);
  `training.eval_max_rows` caps rows per evaluation.
- **Runtime sequence packing** (`data.runtime_packing`) — no compute spent on padding.
- **Preflight** (`python -m bhaskera.launcher.preflight`): login-node checks of model, libraries, data and
  GPU-memory fit before a job is queued.
- Per-step **peak GPU memory** (max over ranks), step-time breakdown, straggler detection, startup watchdog,
  node-health preflight.
- **`slurm/` kit**: `train.sbatch`, `tokenize.sbatch`, `submit.sh` (partition/time picker + preflight),
  one tested recipe per model, `install-bhaskera.sh`, an Lmod modulefile and
  `requirements-paramrudra.lock` (exact package versions of the benchmark environment).
- `scripts/convert_mxfp4_to_bf16.py`: convert an MXFP4 checkpoint (gpt-oss) to bf16 on disk.

### Changed
- README rewritten with a quick-run guide, recipe table and measured performance.
- The version written into tokenised-cache metadata now comes from the installed package
  (single source: `pyproject.toml`).

### Fixed
- Tokenising a model whose tokenizer defines no special tokens at all (BharatGen Param-1-7B) failed with
  "tokenizer does not have a padding token"; the pad token now falls back to the id in the model's `config.json`.
- Preflight estimated QLoRA memory at bf16 size and pipeline memory per GPU count instead of per stage;
  QLoRA recipes were rejected (Llama-3.3-70B: 147.7 GiB estimated vs 55 GB measured).
- MoE: FSDP counted the auxiliary loss twice; router matching treated `gate_proj` LoRA as a router;
  `freeze_router: false` was ignored by the pipeline.
- Pipeline: stage outputs must not be views (MoE aux-loss crash); routing store per checkpointed call
  (CheckpointError with micro-batches in flight).
- FSDP2: gradient-sync toggle was a no-op; non-persistent buffers (RoPE `inv_freq`) were broadcast as
  garbage; dist-init and checkpoint rank races.

## [2.2.0]

Baseline before this changelog: FSDP2 / DDP training, MoE-aware sharding, PEFT LoRA, persistent
tokenisation cache, chat formats (`chatml`, `alpaca`, `sharegpt`), DCP checkpoints, architecture
introspection, Liger kernels, MLflow / Ray Dashboard observability, inference CLI.

[Unreleased]: https://github.com/kishoryd/Bhaskera/compare/v2.3.0...HEAD
[2.3.0]: https://github.com/kishoryd/Bhaskera/releases/tag/v2.3.0
