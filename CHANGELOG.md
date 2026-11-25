# Changelog

All notable changes to Bhaskera are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/): **MAJOR** for config or API changes that break existing
runs, **MINOR** for new features that keep existing configs working, **PATCH** for fixes only.

## [Unreleased]

### Energy, log out, and a plainer Home
- **Energy used:** the energy a run drew on its GPUs, worked out from the summed GPU power it records (kWh). It shows as a column in the runs table, as *Energy so far* on a running job, and as a total on Home. It counts GPUs only, and runs made before this release show nothing. The power reading is now part of the default metrics (`cluster_gpu_power_w_sum`); before, only the detailed level kept it.
- **Log out** in the account menu: it asks first, then stops the page (or, behind Open OnDemand, ends the portal session). Jobs already submitted keep running.
- Home shows a *Multimodal model* card in place of the text one, and the empty Jobs box no longer has buttons or a pointer sentence. The cluster name is no longer in the Home subtitle.
- **Open OnDemand app** (`ood/`): Launch, then Open, and the page opens in a tab with no setup for the user (`bhaskera-dashboard portal`, session token, MLflow under the portal prefix). `ood/install.sh` and `ood/check.sh` are for the administrators.

### Where your files go
- Everything a user creates stays in the run folder (the one with `submit.sh`): the MLflow store (`mlflow-runs/`, with the registry), small settings (`.bhaskera/`) and the benchmark data cache
  (`.cache/huggingface/`) now default to it. A copy already kept in the home folder is used until a copy or link exists in the run folder, so no run disappears. `src/bhaskera/paths.py` decides.

### Home page
- **Five pages** (Home, Train, Serve, Evaluate, Monitor) with a hover menu under each name in the header and under Docs, a new logo that returns to Home, a **New run** button, and one set of action cards per page.
  Home shows four numbers (running now, training runs, saved models, benchmarked) that lead to the pages. When a chat server finishes loading, the page opens Serve.
- Only runs and saved models that Bhaskera made are listed: a run carries the tag `framework: bhaskera`, or is filed in a `bhaskera*` experiment (older runs); other tools' runs in the same MLflow store are not shown.
- **Create recipe** (Start menu): a recipe from a few choices for text and multimodal models, with QLoRA + DDP, DDP, FSDP2, HSDP and Pipeline,
  *All settings* (every option, searchable, with its meaning), a *Custom* mode (paste or start from a starter) and *Prepare data* for raw JSONL.
  Recipes are checked with the training job's own loader before they are written. DDP and QLoRA recipes turn on activation checkpointing.
- **Benchmark** takes several models (a job each), registered names, your own tests (from a question/answer file), a few-shot count and
  *tuned model only*. **Compare models** puts their scores side by side.
- **Models** lists every registered version with Chat and Benchmark, without MLflow's own page. A registered version carries a hand-off note:
  your note, the base model, settings and scores (`register add --note`, `register show`). Aliases are typed, not preset.
- A guided path from data to a usable model, a **Guide** of the words used, plain explanations beside each choice, a Start menu, and a
  calmer, more consistent design.
- Train refuses an HSDP or Pipeline recipe whose shard size or stages do not divide the GPUs of the chosen nodes.

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
- Sharded checkpoints with the newest N kept, saved every `checkpoint.save_every_steps` steps (default 500) and when the run ends. A run
  that stopped is continued with `./submit.sh resume --run <id>` or the **Resume** button: the same recipe, nodes and time, the optimizer and
  learning-rate schedule at the saved step, and the training data already used skipped. `--step N` goes back to an earlier checkpoint (the
  later ones are set aside, not deleted), and the resumed run is tagged `resumed_from`.
- Recipe checks before queueing (`bhaskera-preflight`): model files, libraries, data, GPU-memory fit, and any option that does nothing.
- A node check before every job and a plain-words failure summary after it; faulty nodes are remembered and avoided.
- Hyperparameter sweeps (`python -m bhaskera.launcher.sweep`) and data tools (`inspect`, `dedupe`, `filter`, `split`).

### Home page and results
- `bhaskera-dashboard ray`: one web page with your jobs (state, progress, live loss, MFU and GPU load per node), a log window, links to
  the Ray Dashboard, GPU charts and MLflow, and your recent runs.
- Cancel a job, or close a chat server, from the page (two clicks; only your own Bhaskera jobs).
- Start a training job, a chat server or a benchmark from the page. *Check first* runs the limits and preflight without submitting.
- Chat with a tuned model, the base model, or both side by side, from the same weights (`./submit.sh serve`).
- Benchmark a tuned model against its base with the lm-evaluation-harness (`./submit.sh eval`); results are stored with the run.
  Your own tests: `evaluate init-task` turns a question/answer JSONL file into a test, `--save-samples` keeps every example's answer,
  and `evaluate review` shows what tuning fixed, broke and left wrong, with a file of examples to correct.
- MLflow: about 20 curated curves by default, the trained adapter in standard PEFT format, evaluation report and traces.
- Model registry: `register add` (or the **Register a model** card) saves a trained run's adapter under a name and version with its base
  model and results as tags, and an alias such as `champion`; `register list` and `resolve` find it again, and `serve` and `eval` accept
  `name`, `name:3` or `name@alias` wherever a run id goes.

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
