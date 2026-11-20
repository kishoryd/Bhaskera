# Bhaskera

**Distributed LoRA / QLoRA fine-tuning of large language models on SLURM GPU clusters, built on Ray.**

Bhaskera fine-tunes dense and Mixture-of-Experts (MoE) causal language models from one YAML file:
the model, the data and how to spread the work across GPUs. It starts a Ray cluster on the nodes
SLURM gives your job, loads and splits the model, trains LoRA adapters, evaluates, checkpoints and
logs one line of metrics per step. It runs on **PARAM Rudra** (C-DAC, 2 × A100 80 GB per node) and has been
benchmarked there from 8 to 128 nodes with models from 14 B to 235 B parameters. BharatGen Param models are
supported too.

**Version 1.1.1** · Python ≥ 3.11 · PyTorch 2.6 (CUDA 12.4) · transformers 4.57 ·
[Changelog](CHANGELOG.md) · [Releasing](RELEASING.md) · [PARAM Rudra user manual](https://cdacindia.github.io/ParamRudra-20PF-UserManual/bhaskera/)

![Bhaskera architecture](docs/architecture.svg)

---

## Quick run guide

### A. Shared install (PARAM Rudra: nothing to download or install)

```bash
module load bhaskera/1.1.1
cp -r $BHASKERA_KIT /scratch/<project>/<username>/bhaskera-runs && cd /scratch/<project>/<username>/bhaskera-runs
cp $BHASKERA_QUICKSTART/recipes/*.yaml recipes/        # recipes for the models on shared storage
export SLURM_ACCOUNT=<your_account>

./submit.sh recipes/qwen3-14b-qlora.yaml 8 1           # recipe, nodes, hours
tail -f logs/qwen3-14b-qlora_<jobid>.out | grep --line-buffered -E '\[step|eval|exit code'
```

`submit.sh` checks the recipe on the login node (model files, libraries, data, GPU-memory fit), picks the
partition and time limit from the node count, and submits. Your own data and models: see
[Your own model and data](#your-own-model-and-data).

### B. Your own install

Install once on a login node, in your `/scratch` area (5 to 15 minutes, depending on file-system load). Login nodes have no GPU, so the
CUDA version must be named, otherwise `setup.sh` installs a CPU-only PyTorch.

```bash
git clone --branch v1.1.1 https://github.com/kishoryd/Bhaskera.git && cd Bhaskera
BHASKERA_CUDA=12.4 bash setup.sh
source .venv/bin/activate
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install -r slurm/requirements-paramrudra.lock      # the exact package versions used for the benchmarks
export BHASKERA_DIR=$PWD

cp -r slurm ../bhaskera-runs && cd ../bhaskera-runs
export SLURM_ACCOUNT=<your_account>
```

Nothing in the install is compiled: PyTorch, flash-attention and bitsandbytes come as prebuilt CUDA 12.4
wheels, and the Triton kernels compile on the fly. The lock file matters: newer Ray and grpcio releases
break multi-node start-up on PARAM Rudra.

### Your own model and data

Compute nodes have no internet, so download first, on a login node:

```bash
huggingface-cli download Qwen/Qwen3-14B --local-dir ../models/Qwen3-14B     # gated models: huggingface-cli login first
```

Data is JSONL, one conversation per line (see [Data formats](#data-formats)). Tokenise once, as a CPU job,
then paste the two printed cache paths into the recipe:

```bash
# edit recipes/tokenize-chatml.yaml: model.name, data.train_path, data.val_path, data.cache_dir
./submit.sh tokenize recipes/tokenize-chatml.yaml
./submit.sh recipes/qwen3-14b-qlora.yaml 8 12
```

## Recipes

`slurm/recipes/` holds one tested configuration per model. Each file states its measured speed and memory
in the header and needs only your paths. Recipes are set to 10 steps (`max_steps: 10`) so a first job
finishes in minutes. For a full run, set `max_steps` to *training-set tokens ÷ (`tok/s` × `step_s`)*, taken
from the log of that first job.

| Recipe | Model | Strategy | Measured |
|---|---|---|---|
| `qwen3-14b-qlora.yaml` | Qwen3-14B | QLoRA + DDP | one epoch of 1.24 B tokens: 12.5 h on 8 nodes · 49 min on 128 |
| `qwen3-14b-ddp.yaml` | Qwen3-14B | DDP | 14.1 h on 8 nodes · 54 min on 128 |
| `qwen3-30b-a3b-pipeline.yaml` | Qwen3-30B-A3B (MoE) | Pipeline, 4 stages | 12.3 h on 8 nodes · 53 min on 128 |
| `llama-3.3-70b-qlora.yaml` | Llama-3.3-70B | QLoRA + DDP | 2.2 days on 8 nodes · 3.3 h on 128 |
| `llama-3.3-70b-pipeline.yaml` | Llama-3.3-70B | Pipeline, 4 stages | 2.8 days on 8 nodes · 4.6 h on 128 |
| `mistral-large-123b-qlora.yaml` | Mistral-Large-123B | QLoRA + DDP, batch 1 | 4.1 days on 8 nodes · 6.4 h on 128 |
| `mistral-large-123b-pipeline.yaml` | Mistral-Large-123B | Pipeline, 8 stages | 2.6 days on 16 nodes · 7.8 h on 128 |
| `qwen3-235b-a22b-pipeline.yaml` | Qwen3-235B-A22B (MoE) | Pipeline, 16 stages | 2.6 days on 8 nodes · 4.3 h on 128 |
| `param-1-2.9b-ddp.yaml` | BharatGen Param-1-2.9B | DDP, eager attention | 5 nodes: 1.6 s/step, 24 GiB per GPU |
| `param-1-7b-moe-fsdp.yaml` | BharatGen Param-1-7B (MoE) | FSDP | 5 nodes: 10.1 s/step, 12 GiB per GPU |
| `param2-17b-moe-fsdp.yaml` | BharatGen Param2-17B (MoE) | FSDP | 5 nodes: 9.4 s/step, 17 GiB per GPU |

## What it does

| Feature | Detail |
|---|---|
| Parallelism | DDP, QLoRA + DDP (4-bit frozen base model), Pipeline (1F1B / GPipe), FSDP2, HSDP, chosen per run in YAML |
| LoRA | PEFT adapters; `target_modules: [auto]` finds the right layers for any architecture |
| MoE | Frozen routers, load-balancing auxiliary loss, router z-loss; fast stacked-expert compute in the Pipeline strategy |
| Data | `bhaskera-tokenize` tokenises once into a Parquet cache; runtime sequence packing removes padding |
| Chat formats | `chatml`, `alpaca`, `sharegpt`, or your own via `@register_format` |
| Evaluation | Held-out loss and next-token accuracy |
| Checkpoints | Sharded (DCP), trained LoRA weights and optimizer state only (Qwen3-14B QLoRA: 942 MB); a resubmitted job resumes from the latest checkpoint with the learning-rate schedule in step |
| Preflight | `python -m bhaskera.launcher.preflight` checks model, libraries, data and GPU-memory fit on the login node (accurate to about 15% of the measured peak) |
| Monitoring | One log line per step, MLflow curves for the model and for every node, Ray Dashboard with all nodes and GPUs |

### Choosing a strategy

| Strategy | Use it for | YAML |
|---|---|---|
| QLoRA + DDP | Dense models up to ~123 B: fastest at every node count | `model.quantization: qlora`, `strategy: ddp` |
| DDP | Models that fit on one GPU in bf16 (~14 B) | `strategy: ddp` |
| Pipeline | MoE models and very large dense models | `strategy: pipeline` + `pipeline.stages` |
| FSDP / HSDP | Only when nothing else fits; weights cross the network every step | `strategy: fsdp` (+ `HYBRID_SHARD`) |

With LoRA the base model is frozen, so DDP, QLoRA and Pipeline exchange only the small adapter gradients
between nodes and scale almost linearly (8 → 128 nodes: about 15× faster at constant MFU).

### What the optimisations save

Effect of switching each one off on its own: Qwen3-14B, QLoRA, 5 nodes (10 A100s), sequence length 4096.

| Setup | Step time | Peak GPU memory | Real tokens/s |
|---|---|---|---|
| All on (as shipped) | 4.78 s | **16.0 GB** | **17,156** |
| Triton kernels (Liger) off | 5.27 s | 29.9 GB | 15,551 |
| Sequence packing off | 4.73 s | 16.0 GB | 9,826 |
| Flash-attention off, packing off | 5.96 s | 16.2 GB | 8,176 |

The Triton kernels cut peak memory by 46% and steps by 10%; packing gives 1.75× more real tokens per
second (42.5% of an unpacked batch is padding); flash-attention makes steps 1.26× faster and is what makes
packing possible.

## Monitoring a job

| Where | What you see |
|---|---|
| Job log | One line per step; every 10 steps a summary of all nodes: `[step 30] nodes=5 gpus=10 gpu_util=94/98/99% (min/mean/max) gpu_mem_max=17.7GiB temp_max=58C` |
| MLflow (`bhaskera-dashboard start`) | One run per job: loss, accuracy, speed, step-time spread over the ranks, GPU curves over the whole job (`cluster_...`) and per node (`node_<host>_...`, up to 16 nodes) |
| Ray Dashboard (`bhaskera-dashboard ray`) | Every node of the running job with CPU, memory and per-GPU load, and the logs of each worker, at any number of nodes |

To open the Ray Dashboard, run this on a login node. It prints one command for your laptop that is always
the same: it finds your running job (and waits if the job is still queued), so you never need to know
which node it runs on.

```bash
module load bhaskera/1.1.1
bhaskera-dashboard ray                  # or --job <jobid> when several jobs run
bhaskera-dashboard ssh-config           # optional: an entry for ~/.ssh/config, then `ssh bhaskera-ray`
```

On the laptop, open <http://localhost:8265>: it lists all your running Bhaskera jobs, with a link to the Ray
Dashboard and the Grafana charts of each (`http://job-<id>.localhost:8265`, which browsers send to your laptop
by themselves). Jobs that start later appear on their own. MLflow is at <http://localhost:5055>. The command also starts MLflow on the login node and keeps the connection
open when the job ends, so the metrics of a finished or failed job stay available in MLflow (a failed run is
marked FAILED, a killed one KILLED). The Ray Dashboard and Grafana exist only while the job runs.
Ray numbers its jobs itself, so the IDs in the dashboard differ from the Slurm job ID, and a job shows
two entries: the cluster launcher (no tasks) and the training. The charts need Prometheus and Grafana
(`slurm/install-observability.sh`); without them the node list still works. The `monitoring.metrics`
section of a recipe controls the node readings: `node_stats_every_n_steps` (default 10, `0` turns them off)
and `node_stats_per_node_max` (default 16).

## Data formats

One JSON object per line:

```json
{"messages": [{"role": "user", "content": "What is HPC?"}, {"role": "assistant", "content": "High-performance computing is ..."}]}
```

`alpaca` (`instruction` / `input` / `output`) and `sharegpt` (`conversations` with `from` / `value`) are
also built in. Set `data.format` in the tokenize recipe.

## Measured performance (PARAM Rudra, November 2026)

Every model × strategy × node count (8, 16, 32, 64, 128) was run for 10 steps with sequence length 4096
on a 528,450-conversation ChatML dataset (1.24 B real tokens). Fastest setup per model at 128 nodes:

| Model | Strategy | Tokens/s | MFU | One epoch | Eval accuracy |
|---|---|---|---|---|---|
| Qwen3-14B | QLoRA + DDP | 424,899 | 34.4% | 49 min | 65.5% |
| Qwen3-30B-A3B (MoE) | Pipeline | 386,473 | 7.7% | 53 min | 67.1% |
| Llama-3.3-70B | QLoRA + DDP | 103,481 | 38.8% | 3.3 h | 62.8% |
| Mistral-Large-123B | QLoRA + DDP | 57,457 | 37.3% | 6.4 h | 71.9% |
| Qwen3-235B-A22B (MoE) | Pipeline | 80,110 | 9.9% | 4.3 h | 68.0% |

MFU is the share of the A100s' peak bf16 throughput spent on model work (active parameters for MoE).
Full tables for every strategy and node count: the PARAM Rudra user manual, *LLM Fine-tuning* section.

## Commands

| Command | Purpose |
|---|---|
| `./submit.sh <recipe> <nodes> <hours>` | Preflight, pick partition and time limit, submit (`slurm/`) |
| `./submit.sh tokenize <recipe>` | Tokenise data as a one-node CPU job |
| `python -m bhaskera.launcher.preflight --config <yaml> --num-gpus <N>` | Check a run before queueing it |
| `bhaskera-tokenize --config <yaml> --split both` | Tokenise train and validation data into the Parquet cache |
| `bhaskera-train --config <yaml> --num-workers <N>` | Training driver (started by `slurm/train.sbatch`) |
| `bhaskera-dashboard start --port <port>` | MLflow UI on the login node (`stop`, `status`, `tunnel`) |
| `bhaskera-dashboard ray [--job <id>]` | Tunnel commands for the Ray Dashboard of a running job |

Set `BHASKERA_EXCLUDE=cbgpuXXXX,...` before `submit.sh` to keep jobs off faulty nodes.

## Configuration

One YAML file per run, schema in `src/bhaskera/config.py`:

| Section | Holds |
|---|---|
| `model` | Path, dtype, attention implementation, `quantization: none \| qlora`, Liger kernels |
| `data` | Tokenised cache paths, sequence length, `runtime_packing` |
| `lora` | Rank, alpha, dropout, target modules, MoE options |
| `moe` | Auxiliary and z-loss weights, router freezing |
| `training` | Batch size, gradient accumulation, learning rate, `max_steps`, evaluation, `distributed.strategy` and its `fsdp` / `ddp` / `pipeline` blocks |
| `checkpoint` | Directory, interval, retention |
| `logging`, `monitoring` | MLflow / W&B tracking, per-step metrics |

> **Checkpoints:** with DDP, QLoRA and FSDP, `checkpoint.save_interval` counts **epochs** (a checkpoint is
> written when the run ends or an epoch ends); with Pipeline it counts **steps**. A relative
> `checkpoint.save_dir` is resolved against the folder the job was submitted from.

## Central installation (administrators)

`slurm/install-bhaskera.sh` builds `/home/apps/bhaskera/<version>` (Python environment, source and job
scripts) from a release tag, and `slurm/modulefiles/bhaskera/<version>.lua` is the matching Lmod modulefile.
`slurm/make-quickstart.py` prepares a shared folder of models, tokenised data and ready recipes, which is
what `$BHASKERA_QUICKSTART` points at. Details: the *Installation* page of the user manual.

## Repository layout

```text
Bhaskera/
├── slurm/                 # tested SLURM kit: train.sbatch, tokenize.sbatch, submit.sh, recipes/,
│                          #   install-bhaskera.sh, make-quickstart.py, Lmod modulefile, package lock file
├── docs/                  # architecture diagram
├── configs/               # further example configs
├── scripts/               # benchmarking, profiling and validation scripts
├── src/bhaskera/
│   ├── config.py          # dataclass schema + YAML loader
│   ├── introspect.py      # architecture auto-detection (layers, experts, routers, LoRA targets)
│   ├── data/              # Ray Data datasets, chat formats, tokenisation, packing
│   ├── distributed/       # FSDP2 / DDP / pipeline wrapping, MoE routing, checkpoints
│   ├── launcher/          # CLI entry points
│   ├── models/            # model loader, LoRA / QLoRA, compatibility shims
│   ├── trainer/           # training loops, MoE losses, evaluation, optimiser
│   └── utils/             # throughput / MFU, memory, trackers, watchdog
├── CHANGELOG.md           # what changed in each release
├── RELEASING.md           # branches, versioning, how to cut a release
├── ARCHITECTURE.md        # design notes
├── setup.sh               # installer (writes bhaskera-activate.sh)
└── pyproject.toml
```

## Versions and releases

Bhaskera follows [Semantic Versioning](https://semver.org/). Releases are tags (`v1.1.1`); every change
is listed in the [Changelog](CHANGELOG.md), and [RELEASING.md](RELEASING.md) describes the branch model and
how a release is cut. Use a tagged version for installs: `git clone --branch v1.1.1 ...`.

## Known limitations

- After a resume the data iterator starts again from the beginning of the training set.
- MoE models are slow outside the Pipeline strategy (0.2–1.7% MFU): use Pipeline for MoE. The BharatGen MoE
  models run under FSDP only; Pipeline does not support their custom code.
- DDP needs the whole model on each GPU: 70 B and larger need QLoRA or Pipeline. Qwen3-235B does not fit
  with QLoRA (about 118 GB of 4-bit weights per GPU). Mistral-Large-123B QLoRA needs `batch_size: 1`.
- BharatGen Param-1-2.9B supports neither flash-attention nor PyTorch attention, so it runs with `eager`
  attention and without packing (74% of each batch is padding).
- Training gpt-oss (MXFP4) models is not supported. `scripts/convert_mxfp4_to_bf16.py` converts one to bf16
  for A100 nodes.
