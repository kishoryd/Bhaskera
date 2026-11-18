# Bhaskera

Ray-native distributed LLM fine-tuning framework, tested at scale on **PARAM Rudra** (8–128 nodes, up to 256 NVIDIA A100 80 GB GPUs).

Bhaskera wraps PyTorch FSDP2 / DDP / pipeline parallelism, Hugging Face `transformers`, PEFT LoRA and QLoRA, and a persistent Ray Data tokenisation pipeline behind a YAML-driven CLI. It targets supervised fine-tuning of dense and mixture-of-experts (MoE) causal language models on Slurm clusters, with MLflow and Ray Dashboard observability.

Version: 2.2.0 (Python >= 3.11).

---

## Quick start on PARAM Rudra

**1. Log in and set up** (once; compiling flash-attention is the slow part):

```bash
ssh -p 4422 <user>@paramrudra.cdacb.in
cd /scratch/<project>/<you>
git clone git@github.com:kishoryd/Bhaskera.git && cd Bhaskera
bash setup.sh                      # venv + PyTorch + Bhaskera + QLoRA/MLflow/W&B extras + flash-attention
source bhaskera-activate.sh        # run this in every new shell
```

**2. Point a config at your model and data.** Copy a tested config and edit `model.name`, `data.tokenized_path` and `data.val_tokenized_path` (see *Typical workflow* to tokenise your own data):

```bash
cp configs/release/rel-ddp-qwen14b.yaml configs/my_run.yaml
bhaskera-preflight --config configs/my_run.yaml --num-gpus 16    # seconds, on the login node: all PASS?
```

**3. Train** on 8 nodes (16 GPUs):

```bash
mkdir -p logs
sbatch --nodes=8 --partition=gpu-small --time=04:00:00 \
  --exclude=cbgpu0075,cbgpu0076,cbgpu0159,cbgpu0178,cbgpu0270,cbgpu0305,cbgpu0312 \
  scripts/train.sbatch configs/my_run.yaml
```

**4. Monitor:**

```bash
squeue -u $USER                                                   # queued / running
tail -f logs/bhaskera_<jobid>.out | grep --line-buffered -E '\[step|eval'   # first step after ~5–15 min
```

Charts in MLflow: on the login node run `bhaskera-dashboard start --port 5055` and `hostname -i`, then on your laptop:

```bash
ssh -J <user>@paramrudra.cdacb.in:4422 -L 5055:localhost:5055 -L 8265:localhost:8265 <user>@<login-node-address>
```

Open http://localhost:5055 (MLflow). For the Ray Dashboard of a running job, first run on the login node `ssh -N -L 8265:localhost:8265 <first node of the job> &`, then open http://localhost:8265. Details are in *Running on PARAM Rudra* and *Monitoring* below.

---

## What it does

- **Five ways to spread a run over GPUs**, chosen per run in YAML: DDP, FSDP2 (`fully_shard`), HSDP (FSDP2 with `HYBRID_SHARD`), QLoRA + DDP (4-bit frozen base model), and Pipeline (1F1B or GPipe schedule).
- **LoRA fine-tuning** via PEFT, with target modules found automatically (`target_modules: ["auto"]`).
- **MoE support:** frozen routers, Switch-Transformer-style auxiliary load-balancing loss, router z-loss, expert-utilisation logging. Under Pipeline: routing pinned across activation-checkpoint recompute, and frozen experts stacked for fast batched maths.
- **Persistent tokenisation cache:** `bhaskera-tokenize` writes Parquet shards keyed by model, sequence length, dataset and format; training reads the cache directly.
- **Runtime sequence packing** (`data.runtime_packing`): joins short conversations into full rows so no compute goes to padding.
- **Pluggable chat formats:** built-in `chatml`, `alpaca`, `sharegpt`; custom formats register with `@register_format`.
- **Evaluation** during training: held-out loss and next-token accuracy (`training.eval_steps`, `training.eval_max_rows`).
- **Preflight check** (`bhaskera-preflight`): on the login node, in seconds, checks model, tokenizer, kernels and data caches, and estimates memory per GPU.
- **Sharded checkpointing** with `torch.distributed.checkpoint` and an atomic `.complete` marker; resume from the last save.
- **Architecture-agnostic introspection** (`bhaskera.introspect`): one pass over the loaded model finds decoder layers, experts, routers and LoRA targets. No hardcoded model class names.
- **Liger Kernel** (fused RMSNorm / RoPE / SwiGLU / cross-entropy) and **flash-attention 2**.
- **Observability:** one log line per step (loss, tokens/s, MFU, padding, peak GPU memory, time breakdown, slowest GPU); MLflow on a shared `$HOME` file store; Ray Dashboard; optional W&B.
- **Reliability:** startup watchdog, straggler detection, node health check script.
- **Inference CLI** (`bhaskera-infer`) with TurboQuant KV cache and speculative decoding (not re-tested in this release).

---

## Running on PARAM Rudra

### Hardware (as measured, November 2026)

| | |
|---|---|
| GPU nodes | `cbgpuXXXX`: 2 × NVIDIA A100 80 GB PCIe, 48 CPU cores, about 187 GB RAM |
| Network | One 100 Gb/s InfiniBand card per node (`ib0`), attached to the other CPU socket from the GPUs. **GPUDirect RDMA does not work**, so all GPU traffic between nodes goes through host memory. Slurm uses a flat topology. |
| Internet | Login nodes: yes. **Compute nodes: no.** Download models and datasets on a login node first. |
| Storage | `/scratch` (Lustre) for code, models, data and logs. `$HOME` is shared with compute nodes; MLflow writes there (`~/mlflow-runs`). |

### Logging in

```bash
ssh -p 4422 <user>@paramrudra.cdacb.in
```

This lands on one of several login nodes (for example `login02`, `login08`). Check with `hostname`. Anything you start there, such as the MLflow UI, runs only on that login node. The login nodes block each other's ports except SSH, so tunnels must go **through SSH to that exact login node** (see *Monitoring*).

### Partitions and limits (per user)

| Partition | Max time | Running / submitted jobs | Used here for |
|---|---|---|---|
| `gpu-debug` | 1 h | 2 / 4 | Tests, tokenising, 1–2-node rehearsals |
| `gpu-small` | 24 h | 4 / 8 | 8–16 nodes |
| `gpu-large` | 12 h | 2 / 4 | 32–64 nodes |
| `gpu-massive` | 6 h | 2 / 2 | 128 nodes |

128-node jobs can wait hours in the queue. Startup (Ray plus model loading) takes about 5 min for Qwen3-14B, 10–15 min for 70B–123B and 15–20 min for Qwen3-235B, before the first step.

### Practical notes from the test campaign

- **Exclude known bad nodes.** These failed during November 2026 testing (broken GPU, network faults or a full local disk): `cbgpu0039–0042, 0055–0058, 0075, 0076, 0086, 0127, 0139, 0155, 0159, 0172, 0178, 0240, 0270, 0305, 0312`. Pass them with `sbatch --exclude=...`. The list changes over time.
- **Slurm can report a failed launch as COMPLETED.** Always check the job log for `[step` lines.
- **Tokenise on a compute node** (`gpu-debug`), not on a login node. Ray stalled on a busy shared login node.
- **MLflow UI port:** port 5000 was already in use on `login02`. Start the UI with `--port 5055` (or any free port).
- **A GPU that fails to start** shows `invalid device ordinal` in the log; a node whose local disk is full shows `No space left on device`. Exclude that node and resubmit.

---

## Installation

**Shared module (PARAM Rudra).** Where an administrator has installed Bhaskera centrally, no setup is needed:

```bash
module load bhaskera/2.2.0
cp $BHASKERA_HOME/configs/release/rel-ddp-qwen14b.yaml my_run.yaml    # tested configs ship with the module
sbatch -A <account> --nodes=8 $BHASKERA_HOME/scripts/train.sbatch my_run.yaml
```

Administrators install it with `bash scripts/install_system.sh` from a checkout. It builds the environment in
`/home/apps/miniconda/envs/bhaskera` from `scripts/requirements-lock.txt` (the exact tested versions), copies configs,
launcher and docs to `/home/apps/bhaskera/<version>`, and writes the modulefile `/home/apps/AIStack/bhaskera/<version>`.
`ENV_PREFIX`, `SHARE` and `MODDIR` override the three locations for a staging install.

**Your own copy:**

```bash
cd /scratch/<project>/<you>
git clone git@github.com:kishoryd/Bhaskera.git && cd Bhaskera
bash setup.sh
source bhaskera-activate.sh
```

`setup.sh` finds CUDA (on PARAM Rudra through `module load cuda/12.4`), maps it to a PyTorch wheel, creates a Python 3.11 venv with `uv`, installs PyTorch, then Bhaskera with the `[wandb,mlflow,qlora]` extras, then builds flash-attention (a failed build is not fatal). It writes `bhaskera-activate.sh` into the repository folder: it loads CUDA and activates the venv, and every job script sources it.

Overrides: `BHASKERA_CUDA=12.4 bash setup.sh`, `BHASKERA_VENV=/path bash setup.sh`.

---

## Typical workflow on PARAM Rudra

**1. Get the model and data onto `/scratch`** from a login node (compute nodes have no internet), for example with `huggingface-cli download <model> --local-dir /scratch/.../models/<model>`.

**2. Tokenise once**, as a small job. Start from `configs/local_data_tokenize_chatml.yaml`: set `model.name`, your `train_path` / `val_path`, `seq_len` and `cache_dir`.

```bash
sbatch --partition=gpu-debug --nodes=1 --gres=gpu:1 --cpus-per-task=48 --time=00:30:00 \
  --wrap "source bhaskera-activate.sh && bhaskera-tokenize --config my_tokenize.yaml --split both"
```

It prints the two lines (`tokenized_path`, `val_tokenized_path`) to paste into the training config.

**3. Choose a config** from `configs/release/` (table in [configs/README.md](configs/README.md)). Set `model.name` and the data paths. For a real run, raise `training.max_steps` and set `checkpoint.enabled: true`; the release configs are 6–10-step tests with checkpointing off.

**4. Preflight** on the login node:

```bash
bhaskera-preflight --config configs/release/rel-ddp-qwen14b.yaml --num-gpus 16
```

**5. Train:**

```bash
mkdir -p logs                     # Slurm does not create it
sbatch --nodes=8 --partition=gpu-small --time=04:00:00 --exclude=<bad nodes> \
  scripts/train.sbatch configs/release/rel-ddp-qwen14b.yaml
```

`scripts/train.sbatch` is the launcher used for the 125 test runs. It starts Ray on every node (`ray symmetric-run` over InfiniBand), sets the NCCL and memory settings used in testing, and runs `bhaskera-train` with one worker per GPU. It is sized for PARAM Rudra GPU nodes (2 GPUs, 48 cores, 170 GB requested); edit its `#SBATCH` lines and `NUM_GPUS` for other hardware.

**6. Monitor** (next section).

---

## Monitoring

**Step log:**

```bash
tail -f logs/bhaskera_<jobid>.out | grep --line-buffered -E '\[step|eval'
```

```
[step 15] loss=1.5586 lr=3.01e-05 grad_norm=0.3502 tok/s=6940 MFU=36.0% HFU=53.4% pad=0% step_s=4.68 MFU_step=36.2% mem=16.0/16.3GiB | step=4.59s data=0.00s fwd+bwd=4.58s opt=0.00s
[step 15] eval_loss=1.5579 eval_acc=0.6567
```

| Field | Meaning |
|---|---|
| `loss`, `eval_loss`, `eval_acc` | Training loss; loss and next-token accuracy on held-out data |
| `lr`, `grad_norm` | Learning rate; size of the update (a sudden jump is a warning sign) |
| `tok/s`, `MFU`, `pad` | Real tokens per second (all GPUs); share of the GPUs' peak maths speed used for model work; share of the batch that was padding |
| `HFU`, `MFU_step` | Like MFU, also counting activation-checkpoint recompute; MFU of this step alone |
| `mem` | Peak GPU memory on the busiest GPU (allocated / reserved) |
| `step`, `data`, `fwd+bwd`, `opt`, `slowest=rN` | Where the step's time went; the slowest GPU when one lags |

**MLflow** (charts for every run, kept for comparison). On your login node:

```bash
bhaskera-dashboard start --port 5055      # remembers the port next time
hostname -i                               # this login node's internal address, e.g. 172.40.0.22
```

On your laptop:

```bash
ssh -J <user>@paramrudra.cdacb.in:4422 -L 5055:localhost:5055 <user>@<login-node-address>
# browser: http://localhost:5055   (experiment "bhaskera")
```

**Ray Dashboard** (nodes, one worker per GPU, GPU use, logs), only while the job runs. On the same login node:

```bash
ssh -N -L 8265:localhost:8265 $(scontrol show hostnames $(squeue -j <jobid> -h -o %N) | head -1) &
```

On your laptop, add `-L 8265:localhost:8265` to the ssh command above, then open http://localhost:8265.

From VS Code Remote-SSH you can instead forward ports 5055 and 8265 in the **Ports** tab.

---

## Choosing a strategy

Set `training.distributed.strategy` (`ddp`, `fsdp` or `pipeline`). HSDP is `fsdp` with `fsdp.sharding_strategy: HYBRID_SHARD`; QLoRA is `ddp` with `model.quantization: qlora`.

| Strategy | In plain words | Use it for | Avoid when |
|---|---|---|---|
| DDP | Every GPU holds the whole model; only small LoRA updates cross the network | Models that fit one GPU in bf16 (about 14B) | Larger models (out of memory) |
| QLoRA + DDP | Frozen model stored in 4-bit so it fits one GPU | 70B–123B dense models: fastest at scale | MoE models; Qwen3-235B does not fit |
| Pipeline | Layers split into stages on different GPUs, like an assembly line | Very large models and **all MoE models** | Very small models |
| FSDP | Model cut into pieces over all GPUs, fetched every step | When nothing else fits | 128 nodes on PARAM Rudra (weight traffic dominates) |
| HSDP | FSDP inside small groups of GPUs | As FSDP; keep each group inside one node | Groups that span nodes (no gain here) |

In the 10-step tests, QLoRA's eval accuracy was within about half a point of the bf16 strategies (Llama-70B, 8 nodes: 62.99% vs 63.55%). A longer quality comparison has not been done.

---

## Tested results on PARAM Rudra

5 models × 5 strategies × 5 node counts (8, 16, 32, 64, 128) = **125 runs**, each 10 steps plus an evaluation: **105 passed, 20 failed** (all 20 are memory limits: DDP with models over 80 GB, and Qwen3-235B in QLoRA). Full table: [docs/scale-matrix-results.md](docs/scale-matrix-results.md).

Fastest setup per model at 128 nodes. "Time per epoch" is projected from measured speed for one pass over a 1.24-billion-token training set, startup excluded.

| Model | Strategy | Tokens/s | MFU | Time per epoch | Eval accuracy |
|---|---|---|---|---|---|
| Qwen3-14B | QLoRA + DDP | 424,899 | 34.4% | 49 min | 65.5% |
| Qwen3-30B-A3B (MoE) | Pipeline | 386,473 | 7.7% | 53 min | 67.1% |
| Llama-3.3-70B | QLoRA + DDP | 103,481 | 38.8% | 3.3 h | 62.8% |
| Mistral-Large-123B | QLoRA + DDP | 57,457 | 37.3% | 6.4 h | 71.9% |
| Qwen3-235B-A22B (MoE) | Pipeline | 80,110* | 9.9% | 4.3 h | 68.0% |

\* Some steps ran over 1.5× slower than usual (a slow node or network); included in the figure.

Scaling, Qwen3-14B with QLoRA (time per epoch): 8 nodes 12.5 h → 16 nodes 6.3 h → 32 nodes 3.2 h → 64 nodes 1.8 h → 128 nodes 49 min. QLoRA, DDP and Pipeline nearly halve the time each time nodes double. FSDP and HSDP fall to about 5–7% MFU at 128 nodes, because on PARAM Rudra every weight transfer goes through host memory.

---

## CLI entry points

| Command | Module | Purpose |
|---|---|---|
| `bhaskera-train` | `bhaskera.launcher.train` | Ray Train driver: one worker per GPU (launch through `scripts/train.sbatch` on Slurm) |
| `bhaskera-tokenize` | `bhaskera.launcher.tokenize` | Tokenise a dataset once into a Parquet cache |
| `bhaskera-preflight` | `bhaskera.launcher.preflight` | Pre-submission checks and memory estimate, on the login node |
| `bhaskera-dashboard` | `bhaskera.launcher.dashboard` | Start / stop / status of the MLflow UI on the login node |
| `bhaskera-infer` | `bhaskera.launcher.infer` | Generation from a model or checkpoint |
| `bhaskera-diag` | `bhaskera.launcher.diagnostics` | NCCL all-reduce and bandwidth check |

---

## Configuration

One YAML file per run (`--config`). The schema is the dataclasses in `src/bhaskera/config.py`.

| Section | Holds |
|---|---|
| `model` | `name` (local path on PARAM Rudra), `dtype`, `attn_impl` (`flash_attention_2`), `use_liger_kernel`, `quantization` (`none` or `qlora`) |
| `data` | `name: local`, `format`, `seq_len`, `tokenized_path`, `val_tokenized_path`, `runtime_packing`, `num_workers` (2 × nodes) |
| `lora` | `r`, `alpha`, `dropout`, `target_modules` (`["auto"]`), `include_experts`, `freeze_router` |
| `moe` | `aux_loss_weight`, `router_z_loss_weight`, `freeze_router`, expert-utilisation logging |
| `training` | `batch_size`, `grad_accum`, `lr`, `max_steps`, `warmup_steps`, `eval_steps`, `eval_max_rows`, `target_eval_loss` |
| `training.distributed` | `strategy` (`ddp` / `fsdp` / `pipeline`), plus `fsdp` (`sharding_strategy`, `hsdp_shard_size`, activation checkpointing), `ddp` and `pipeline` (`stages`, `microbatches`, `schedule`: `1f1b` / `gpipe`, `activation_checkpointing`) blocks |
| `checkpoint` | `enabled`, `save_dir`, `save_interval`, `keep_last_n` |
| `logging` | `tracker` (`mlflow`, `wandb`, `ray`, a list, or `none`), `project`, `run_name` |
| `monitoring` | Ray Dashboard host/port, metrics export port, per-step metric switches |
| `inference` | Generation settings, KV-cache mode, TurboQuant and speculative-decoding options |

Configs shipped: `configs/release/` (the 15 configs tested in this release), tokenise configs for the five tested models, a custom-data example, and inference examples. See [configs/README.md](configs/README.md).

---

## Validation status

| Done in this release | Result |
|---|---|
| Unit tests (`pytest src`, on a compute node) | 97 passed, 8 skipped. One timing test, `test_amortised_O1_per_step`, can fail on a heavily loaded login node. |
| Pipeline correctness harness (`scripts/pp_validate.py`): pipeline vs the unsplit model, including loss, router terms and eval accuracy | 56 checks passed, 0 failed |
| 125-run scale matrix on PARAM Rudra | 105 passed; 20 failed for memory limits |
| Resume from checkpoint | Tested with Pipeline (Llama-70B, Qwen3-30B) |
| Tokenise → preflight → train with DDP, FSDP, Pipeline and QLoRA on a fresh public dataset | Run end to end on 1–2 nodes; eval loss fell and accuracy rose in every strategy |

**Not done:** a long run (hundreds to thousands of steps) comparing final quality between strategies; the test runs are 10 steps, so accuracy shows the strategies agree, not which is best. Checkpoint resume was not re-tested with FSDP, HSDP or QLoRA.

---

## Known limitations

1. **MoE models are slow outside Pipeline:** 0.2–1.7% MFU with FSDP, HSDP, DDP and QLoRA, because Hugging Face's expert code loops over every expert. Use Pipeline (8–11% MFU).
2. **DDP needs the whole model on each GPU:** out of memory for Llama-70B, Mistral-123B and Qwen3-235B. Use QLoRA or Pipeline.
3. **Qwen3-235B with QLoRA does not fit** (about 118 GB of 4-bit weights against 80 GB per GPU). Mistral-123B with QLoRA fits only at batch size 1 with gradient accumulation 2 (78 of 80 GB).
4. **FSDP and HSDP lose speed at 128 nodes** on PARAM Rudra, because there is no GPUDirect RDMA.
5. **Checkpoints with FSDP, HSDP, DDP and QLoRA are written only when an epoch ends** (`save_interval` counts epochs for them), so a long single-epoch run has no mid-run save. Pipeline saves every `save_interval` steps. Until this changes, split long runs into shorter ones, or use Pipeline.
6. **gpt-oss-120b is not supported** (MXFP4 experts). `scripts/convert_mxfp4_to_bf16.py` writes a bf16 copy, but training on it was not validated.
7. **Inference and serving are outside this release.** `bhaskera-infer` is included but not re-tested. A vLLM-based engine and an OpenAI-compatible server are on the `inference` and `serve` branches, not merged.
8. **`bhaskera-activate.sh` is machine-specific.** The committed copy points at another environment; `bash setup.sh` rewrites it for yours.

---

## Repository layout

```
Bhaskera/
├── configs/
│   ├── release/                 # the 15 configs tested in this release
│   ├── tests/                   # base config for the pipeline harness and tests
│   └── *.yaml                   # tokenise, custom-data and inference examples
├── scripts/
│   ├── train.sbatch             # Slurm launcher (Ray over all nodes) used for the test runs
│   ├── submit.sh                # older general-purpose Slurm launcher
│   ├── pp_validate.py           # pipeline-vs-unsplit-model correctness harness
│   ├── node_healthcheck.py      # node checks before a run
│   └── convert_mxfp4_to_bf16.py
├── src/bhaskera/
│   ├── config.py                # dataclass schema + YAML loader
│   ├── introspect.py            # ModelProfile auto-detection
│   ├── data/                    # local datasets, formats, packing, tokenisation cache
│   ├── distributed/             # FSDP2 / HSDP / DDP wrap, pipeline, MoE routing, checkpoints
│   ├── launcher/                # CLI entry points
│   ├── models/                  # loader, LoRA, QLoRA
│   ├── trainer/                 # training loops, MoE losses, evaluation, optimiser
│   ├── inference/               # generation, TurboQuant KV cache, speculative decoding
│   └── utils/                   # throughput / MFU, memory, watchdog, trackers
├── docs/                        # scale-matrix results; earlier profiling and robustness notes
├── ARCHITECTURE.md              # design and module reference
├── bhaskera-activate.sh         # written by setup.sh: loads CUDA, activates the venv
├── pyproject.toml
└── setup.sh                     # one-shot installer
```

---

## Dependencies

Core runtime (from `pyproject.toml`): `torch>=2.4`, `transformers>=4.45,<5.0.0`, `ray[train,data,default]>=2.10`, `peft>=0.10`, `accelerate>=0.20`, `datasets>=2.0`, `safetensors>=0.4`, `liger-kernel>=0.5`, `nvidia-ml-py>=12.535`, `omegaconf>=2.3.0`, `pyyaml>=6.0`, `numpy`, `pandas`, `psutil`, `tqdm`.

Optional extras: `qlora` (`bitsandbytes>=0.43`), `mlflow`, `wandb`, `flash` (`flash-attn>=2.5`), `inference` (`scipy`), `dev` (`pytest`, `ruff`), `observability` (W&B + MLflow).

## Tests

```bash
pip install -e ".[dev]"
pytest src
```
