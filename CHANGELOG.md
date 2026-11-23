# Changelog

All notable changes to Bhaskera are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/): **MAJOR** for config or API changes that break existing
runs, **MINOR** for new features that keep existing configs working, **PATCH** for fixes only.

## [Unreleased]

## [1.0.0] - 2026-11-23

The first release. Bhaskera fine-tunes language models, and models that also read images, video and audio, on Slurm GPU
clusters with Ray.

### Training
- LoRA, QLoRA (4-bit frozen base) and full fine-tuning, from one YAML file. Strategies: DDP, QLoRA + DDP, Pipeline (1F1B / GPipe),
  FSDP2 and HSDP. `target_modules: [auto]` finds the LoRA layers of any architecture; MoE models get frozen routers, a
  load-balancing loss and a router z-loss.
- **Multimodal:** images, video and audio next to text (Qwen3-VL, Qwen2.5-Omni thinker) with the same strategies. Encoders are found
  from the model config, each encoder block is its own FSDP unit and encoders are frozen by default; every rank runs every encoder,
  so FSDP never waits on a batch without a given modality. Media are loaded while training and never stored in the tokenised cache.
- Chat data in `chatml`, `alpaca` or `sharegpt` form (or your own), tokenised once into a Parquet cache; runtime sequence packing
  removes padding.
- Sharded checkpoints with the newest N kept, and resume with the learning-rate schedule in step.
- Recipe checks before queueing (`bhaskera-preflight`): model files, libraries, data, GPU-memory fit, and any option that does nothing.
- A node check before every job and a plain-words failure summary after it; faulty nodes are remembered and avoided.
- Hyperparameter sweeps (`python -m bhaskera.launcher.sweep`) and data tools (`inspect`, `dedupe`, `filter`, `split`).

### Home page and results
- `bhaskera-dashboard ray`: one web page with your jobs (state, progress, live loss, MFU and GPU load per node), a log window, links to
  the Ray Dashboard, GPU charts and MLflow, and your recent runs.
- Start a training job, a chat server or a benchmark from the page. *Check first* runs the limits and preflight without submitting.
- Chat with a tuned model, the base model, or both side by side, from the same weights (`./submit.sh serve`).
- Benchmark a tuned model against its base with the lm-evaluation-harness (`./submit.sh eval`); results are stored with the run.
  Your own tests: `evaluate init-task` turns a question/answer JSONL file into a test, `--save-samples` keeps every example's answer,
  and `evaluate review` shows what tuning fixed, broke and left wrong, with a file of examples to correct.
- MLflow: about 20 curated curves by default, the trained adapter in standard PEFT format, optional model registry entry, evaluation
  report and traces.

### Clusters
- Nothing is tied to one site: partitions, GPUs and CPUs per node, host memory and the network interface come from a cluster profile
  (`slurm/clusters/`). PARAM Rudra ships as one profile. Without a profile, `submit.sh` uses `$BHASKERA_PARTITION` and the hours you give.
- Slurm QoS limits are read from the cluster and checked before submitting.
- A shared install (`slurm/install-bhaskera.sh`, Lmod modulefile) and a quick-start kit of models, tokenised data and recipes.

### Security
- The home page and everything behind it are private to your user, the MLflow page and Ray Dashboard listen on the local machine only,
  and the Ray Dashboard is read-only through the home page. See [SECURITY.md](SECURITY.md).

### Configs
- `configs/` has the option reference, one starter per parallelism for text and for multimodal, and a tokenizer config for each
  model type. Tested recipes for the supported models are in `slurm/recipes/`.

[Unreleased]: https://github.com/kishoryd/Bhaskera/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kishoryd/Bhaskera/releases/tag/v1.0.0
