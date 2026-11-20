# Bhaskera

**Distributed LoRA / QLoRA fine-tuning of large language models on SLURM GPU clusters, built on Ray.**

Bhaskera fine-tunes dense and Mixture-of-Experts (MoE) causal language models from one YAML file:
the model, the data and how to spread the work across GPUs. It starts a Ray cluster on the nodes
SLURM gives your job, loads and splits the model, trains LoRA adapters, evaluates, checkpoints and
logs one line of metrics per step. It has been validated on **PARAM Rudra** (C-DAC, 2 × A100 80 GB per
node) from 8 to 128 nodes with models from 14 B to 235 B parameters.

Version 2.2.0 · Python ≥ 3.11 · PyTorch 2.6 (CUDA 12.4) · transformers 4.57

---

## Quick run guide

From a login node, in your `/scratch` area. Compute nodes have no internet, so the model and data
are downloaded first.

```bash
# 1. Install (once, about 5 minutes). Login nodes have no GPU: name the CUDA version explicitly,
#    otherwise setup.sh installs a CPU-only PyTorch.
git clone https://github.com/kishoryd/Bhaskera.git && cd Bhaskera
BHASKERA_CUDA=12.4 bash setup.sh
source .venv/bin/activate
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install -r slurm/requirements-paramrudra.lock      # exact versions used for the benchmarks
export BHASKERA_DIR=$PWD

# 2. A working folder with the job scripts and the tested recipes
cp -r slurm ../bhaskera-runs && cd ../bhaskera-runs
export SLURM_ACCOUNT=<your_account>

# 3. Model and data (JSONL, one conversation per line; see "Data formats")
huggingface-cli download Qwen/Qwen3-14B --local-dir ../models/Qwen3-14B

# 4. Tokenise once (edit paths in the recipe first), then paste the two printed paths into the training recipe
./submit.sh tokenize recipes/tokenize-chatml.yaml

# 5. Train: recipe, nodes, hours. submit.sh runs a preflight check, picks the partition and submits.
./submit.sh recipes/qwen3-14b-qlora.yaml 8 12
tail -f logs/qwen3-14b-qlora_<jobid>.out | grep --line-buffered -E '\[step|eval|exit code'
```

On PARAM Rudra a central install may also be available as a module: `module load bhaskera/2.2.0`, then
`cp -r $BHASKERA_KIT bhaskera-runs` and continue from step 3.

## Recipes

`slurm/recipes/` holds one tested configuration per model, each using the fastest strategy measured on
PARAM Rudra. Each file states the measured speed and memory in its header.

| Recipe | Model | Strategy | Time for one epoch (1.24 B tokens) |
|---|---|---|---|
| `qwen3-14b-qlora.yaml` | Qwen3-14B | QLoRA + DDP | 12.5 h on 8 nodes · 49 min on 128 nodes |
| `qwen3-14b-ddp.yaml` | Qwen3-14B | DDP | 14.1 h on 8 nodes · 54 min on 128 nodes |
| `qwen3-30b-a3b-pipeline.yaml` | Qwen3-30B-A3B (MoE) | Pipeline, 4 stages | 12.3 h on 8 nodes · 53 min on 128 nodes |
| `llama-3.3-70b-qlora.yaml` | Llama-3.3-70B | QLoRA + DDP | 2.2 days on 8 nodes · 3.3 h on 128 nodes |
| `llama-3.3-70b-pipeline.yaml` | Llama-3.3-70B | Pipeline, 4 stages | 2.8 days on 8 nodes · 4.6 h on 128 nodes |
| `mistral-large-123b-qlora.yaml` | Mistral-Large-123B | QLoRA + DDP (batch 1) | 4.1 days on 8 nodes · 6.4 h on 128 nodes |
| `mistral-large-123b-pipeline.yaml` | Mistral-Large-123B | Pipeline, 8 stages | 2.6 days on 16 nodes · 7.8 h on 128 nodes |
| `qwen3-235b-a22b-pipeline.yaml` | Qwen3-235B-A22B (MoE) | Pipeline, 16 stages | 2.6 days on 8 nodes · 4.3 h on 128 nodes |

The recipes run 10 steps (`max_steps: 10`) as a smoke test. For a full run, set `max_steps` from the
smoke test's log: training-set tokens ÷ (`tok/s` × `step_s`).

## What it does

| Feature | Detail |
|---|---|
| Parallelism | DDP, QLoRA + DDP (4-bit frozen base model), Pipeline (1F1B / GPipe), FSDP2, HSDP, chosen per run in YAML |
| LoRA | PEFT adapters; `target_modules: [auto]` finds the right layers for any architecture |
| MoE | Frozen routers, load-balancing auxiliary loss, router z-loss; fast stacked-expert compute in the Pipeline strategy |
| Data | `bhaskera-tokenize` tokenises once into a Parquet cache; runtime sequence packing removes padding |
| Chat formats | `chatml`, `alpaca`, `sharegpt`, or your own via `@register_format` |
| Evaluation | Held-out loss and next-token accuracy |
| Checkpoints | Sharded (DCP); a resubmitted job resumes from the last checkpoint |
| Preflight | `python -m bhaskera.launcher.preflight` checks model, libraries, data and GPU-memory fit on the login node |
| Metrics | One log line per step (loss, tok/s, MFU, peak GPU memory, step-time breakdown), MLflow |

### Choosing a strategy

| Strategy | Use it for | YAML |
|---|---|---|
| QLoRA + DDP | Dense models up to ~123 B: fastest at every node count | `model.quantization: qlora`, `strategy: ddp` |
| DDP | Models that fit on one GPU in bf16 (~14 B) | `strategy: ddp` |
| Pipeline | MoE models and very large dense models | `strategy: pipeline` + `pipeline.stages` |
| FSDP / HSDP | Only when nothing else fits; weights cross the network every step | `strategy: fsdp` (+ `HYBRID_SHARD`) |

With LoRA the base model is frozen, so DDP, QLoRA and Pipeline exchange only the small adapter
gradients between nodes and scale almost linearly (8 → 128 nodes: about 15× faster at constant MFU).

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
Full tables: the PARAM Rudra user manual, *LLM Fine-tuning* section.

## Commands

| Command | Purpose |
|---|---|
| `bhaskera-tokenize --config <yaml> --split both` | Tokenise train and validation data into the Parquet cache |
| `python -m bhaskera.launcher.preflight --config <yaml> --num-gpus <N>` | Check a run before queueing it |
| `bhaskera-train --config <yaml> --num-workers <N>` | Training driver (started by `slurm/train.sbatch`) |
| `bhaskera-dashboard start --port <port>` | MLflow UI on the login node (`stop`, `status`) |

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

> **Checkpoint interval:** with DDP, QLoRA and FSDP, `checkpoint.save_interval` counts **epochs** (a
> checkpoint is written when the run ends or an epoch ends); with Pipeline it counts **steps**.

## Repository layout

```text
Bhaskera/
├── slurm/                 # tested SLURM kit: train.sbatch, tokenize.sbatch, submit.sh, recipes/,
│                          #   install-bhaskera.sh and an Lmod modulefile for a central install
├── configs/               # further example configs
├── scripts/               # benchmarking, profiling and validation scripts
├── src/bhaskera/
│   ├── config.py          # dataclass schema + YAML loader
│   ├── introspect.py      # architecture auto-detection (layers, experts, routers, LoRA targets)
│   ├── data/              # Ray Data datasets, chat formats, tokenisation, packing
│   ├── distributed/       # FSDP2 / DDP / pipeline wrapping, MoE routing, checkpoints
│   ├── launcher/          # CLI entry points
│   ├── models/            # model loader, LoRA / QLoRA
│   ├── trainer/           # training loops, MoE losses, evaluation, optimiser
│   └── utils/             # throughput / MFU, memory, trackers, watchdog
├── ARCHITECTURE.md        # design notes
├── setup.sh               # installer (writes bhaskera-activate.sh)
└── pyproject.toml
```

## Known limitations

- MoE models are slow outside the Pipeline strategy (0.2–1.7% MFU): use Pipeline for MoE.
- DDP needs the whole model on each GPU: 70 B and larger need QLoRA or Pipeline.
- Qwen3-235B does not fit with QLoRA (about 118 GB of 4-bit weights per GPU).
- gpt-oss (MXFP4) checkpoints are not yet validated for training; `scripts/convert_mxfp4_to_bf16.py`
  converts one to bf16 for A100 nodes.
