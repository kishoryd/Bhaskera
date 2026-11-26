# Changelog

All notable changes to Bhaskera are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/): **MAJOR** for config or API changes that break existing
runs, **MINOR** for new features that keep existing configs working, **PATCH** for fixes only.

## [Unreleased]

### Plain text, and a recipe for every benchmarked method
- **The starter configs are in the kit.** `configs/` (text and multimodal starters, for example `configs/multimodal/ddp.yaml`) is now copied into `share/kit/` by the installer, so it is in the run folder you make from the kit. Before, only `recipes/` was, and the commands in the multimodal guide failed with "no such file".
- **"Home page" is now "Bhaskera dashboard"** in the documentation, the README and the messages the commands print. The documentation page names MLflow, the Ray Dashboard and Grafana (the GPU charts) and says what each answers and where it is on the dashboard.
- **Five more data formats:** `dolly` (instruction, context, response), `qa` (question, answer, optional context), `scored_response` (keeps only answers scoring at least `min_score`: fine-tuning on good answers, not reward-model training), `tool_calling` (chat with `tool_calls`, `tool` turns and a `tools` column, laid out by the model's own chat template) and `reasoning` (`reasoning_content`, or prompt/reasoning/answer columns, written between `<think>` tags). Checked on the cluster with the real Qwen3 chat template; 20 tool-calling rows went through a real tokenise run.
- **A fourth data format, `prompt_completion`:** rows with a prompt and a completion, under any column names (`format_options: {prompt_field: question, completion_field: answer}`), for question-and-answer sets, Dolly-style files and OpenAI's older fine-tuning files. It is offered in Create recipe and by `data_tools`.
- **Tunnel commands that work.** The login nodes of some clusters hand out a different node on every login, and refuse `ssh -J`. `bhaskera-dashboard start|tunnel|status` now prints a command that ends on the node the page runs on (log in, then `ssh` on to that node), and `bhaskera-dashboard ray --info --node login01` prints the command that starts the home page on a login node of your choice (for example the one with internet). Both were run from outside the cluster and the page answered. Starting the home page no longer prints a second, confusing tunnel for MLflow.
- **Plain text tokenising.** `.txt`, `.text` and `.md` files (optionally `.gz`) are read as documents, with no `data.format` needed. `data.format_options.text_split` says what one example is (`file`, `paragraphs` or `lines`), and `chunk: true` cuts every document into `seq_len` rows ending with the end-of-text token, instead of keeping only the first `seq_len` tokens. Hugging Face datasets that hold only `.txt` files work too (a stray `readme.txt` next to real data files is ignored). Checked on PARAM Rudra with a real tokenizer.
- **Recipes for the whole benchmark matrix.** `slurm/recipes/` now has a recipe for every model and method that fitted in the benchmark: QLoRA, LoRA + DDP, Pipeline, HSDP and FSDP2 for Qwen3-14B, Qwen3-30B-A3B, Llama 3.3 70B, Mistral Large 123B and Qwen3-235B-A22B (21 recipes; the 4 combinations that ran out of memory have none). A test loads every shipped recipe with the real configuration loader.

### Full fine-tuning, more data formats, and Hugging Face names
- **Full fine-tuning in the page.** *What to train* in Create recipe: a small add-on (LoRA, the default) or the whole model. The whole model uses DDP, FSDP2 or HSDP (not QLoRA or Pipeline), a smaller learning rate (1e-5), always keeps its checkpoints (the result is in them), and does not offer Chat, Benchmark or Register, which need an adapter. The disk estimate is corrected: a LoRA or QLoRA checkpoint holds only the trained add-on and its optimizer state (measured about 1 GB for Qwen3-14B), a full fine-tune checkpoint about five times the size of the model.
- **More data formats for the tokenizer:** JSONL, JSON (lines or one array), CSV, TSV, Parquet, Arrow/Feather, any of them gzipped, and Hugging Face datasets saved to disk. A CSV or TSV cell that holds a list (`messages`, `conversations`) as JSON text is decoded.
- **A recipe can name Hugging Face models and datasets:** `model.name: Qwen/Qwen3-1.7B`, `data.train_path: hf://owner/dataset` (or `data.name: owner/dataset`). `./submit.sh` downloads them first on a machine with internet and runs a resolved copy of the recipe; yours is not changed. Train and validation parts are sorted automatically. `python -m bhaskera.downloads get|preview` does the same by hand.
- **Downloads work from any login node.** When the machine running the page has no internet (some login nodes have none), the preview and the download run over SSH on a node that has it (`download_hosts` in the cluster profile). If none is reachable, the message names the machine and what to set.

### Checkpoint choices and disk safety
- **Create recipe** has a *Checkpoints* row in *The training*: save them or not, how often (every N steps), how many to keep (the newest N), and **where**: a folder of your choice (the recipe gets its own folder inside it). Without a choice, they go to `checkpoints/` in the run folder, as before.
- **Disk safety, everywhere something large is written.** One shared check (`bhaskera.diskguard`) looks at the free space of the disk holding the folder, and stops with a plain message instead of failing half way: a new recipe whose checkpoints would not fit is refused; **Check first** shows how much room the checkpoints have; starting training re-checks the recipe's checkpoint folder; preparing data checks room for the result; downloads use the same rule; and no job (training, resume, chat server, benchmark, tokenising) is submitted when the run folder's disk has under 1 GB free. Home shows a warning when the run folder's disk has under 20 GB free. A quota on your account can be lower than the disk's free space, and is not visible to Bhaskera.

### Runs: ten at first, and clear what you do not need
- The runs table shows the latest 10, with *Show 10 more*, *Show all* and *Show fewer*. Each finished, failed or stalled run has a *Remove* button, and *Clear failed and stalled* takes all of those off at once; both ask first. Removing marks a run deleted in MLflow (it is not erased and can be restored with `mlflow runs restore --run-id <id>`). Only your Bhaskera runs can be removed, and never one that is still running.

### Setting up on another cluster, and a clean-up
- The cluster profile has `gpu_mem_gb` (memory of one GPU), and the check before a job uses it instead of assuming 80 GB. A new guide, *Setting Up on Another Cluster*, walks through a first install: requirements, the profile, the modulefile, shared models and data, a check as an ordinary user, and where it breaks.
- Removed: committed `.pyc` files, `requirements.txt` (a loose snapshot that said not to use it; the tested versions are in `slurm/requirements-paramrudra.lock`), and the design document for a central service, which lives on the `central-service` branch. The Open OnDemand app is on the `ood` branch.

### Downloads, sign-in and dataset versions
- The *Your path* strip (Prepare data to Register) is removed from Home; the cards and menus lead to the same places.
- **Downloads** (header): a model or a dataset from Hugging Face into a folder you choose. A size and licence preview comes first, then a download with progress, cancel and resume (starting it again continues where it stopped). Only Hugging Face is contacted, the folder must be one you can write to, and a download above 50 GB asks for confirmation. A finished model joins the model lists; a finished dataset is registered as a dataset version with its repository and revision as the source.
- **Hugging Face sign-in:** paste a key on the Downloads dialog. It is checked with Hugging Face, kept in a private file (`<hf home>/token`, readable by you only) and never shown, logged or put in a command line. Sign out deletes it.
- **Dataset versions** (`bhaskera.dataversions`, a *Data* section on Train): a name, a number and a content fingerprint for a data folder or file. The same content keeps its version, changed content becomes the next number. A run records the version it trained on (`dataset.name`, `dataset.version`), the runs table shows it, and training on data that changed after it was registered is refused until it is registered again. Command line: `python -m bhaskera.dataversions add|list|show|verify|remove`.
- **Run folder and Slurm account:** with no run folder given, `~/bhaskera-runs` is made from the shared kit (set `BHASKERA_RUNS` to change where); the shared kit is never used as a run folder. Jobs are charged to your login name; `SLURM_ACCOUNT` is only needed to use another account.

### Energy, log out, and a plainer Home
- **GPU energy used:** the energy a run drew on its GPUs, worked out from the summed GPU power it records (kWh). It shows as a *GPU energy* column in the runs table, as *GPU energy so far* on a running job, and as a total on Home. It counts GPUs only, and runs made before this release show nothing. The power reading is now part of the default metrics (`cluster_gpu_power_w_sum`); before, only the detailed level kept it.
- **Log out** in the account menu: it asks first, then stops the page (or, behind a portal, ends the portal session). Jobs already submitted keep running.
- Home shows a *Multimodal model* card in place of the text one, and the empty Jobs box no longer has buttons or a pointer sentence. The cluster name is no longer in the Home subtitle.

### Where your files go
- Everything a user creates stays in the run folder (the one with `submit.sh`): the MLflow store (`mlflow-runs/`, with the registry), small settings (`.bhaskera/`) and the benchmark data cache
  (`.cache/huggingface/`) now default to it. A copy already kept in the home folder is used until a copy or link exists in the run folder, so no run disappears. `src/bhaskera/paths.py` decides.

### Home page
- **Five pages** (Home, Train, Serve, Evaluate, Monitor) with a hover menu under each name in the header and under Docs, a new logo that returns to Home, a **New run** button, and one set of action cards per page.
  Home shows four numbers (running now, training runs, saved models, benchmarked) that lead to the pages. When a chat server finishes loading, the page opens Serve.
- Only runs and saved models that Bhaskera made are listed: a run carries the tag `framework: bhaskera`, or is filed in a `bhaskera*` experiment (older runs); other tools' runs in the same MLflow store are not shown.
- **Create recipe** (New run): a recipe from a few choices for text and multimodal models, with QLoRA + DDP, DDP, FSDP2, HSDP and Pipeline,
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
