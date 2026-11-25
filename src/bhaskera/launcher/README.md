# launcher

The commands you type, and the code that starts and watches a run.

## What you control

Commands (each is also available as `python -m bhaskera.launcher.<name>`):

| Command | What it does |
|---|---|
| `bhaskera-tokenize --config <yaml> --split both` | Tokenises train and validation data into a cache and prints the two paths to put in your recipe |
| `bhaskera-train --config <yaml> --num-workers N` | The training driver. On a cluster `slurm/train.sbatch` runs it for you |
| `bhaskera-preflight --config <yaml> --num-gpus N` | Checks a recipe on the login node: model files, tokenizer, libraries, cache, GPU-memory fit |
| `bhaskera-dashboard ray` | The home page for your jobs (see the main README). `start`, `stop`, `status` and `tunnel` manage the MLflow page |
| `bhaskera-infer --config <yaml> --prompt "..."` | Generate text from a checkpoint |
| `bhaskera-diag --num-workers N` | A short NCCL check; run it before a long job on a new cluster |
| `bhaskera-evaluate`, `bhaskera-sweep`, `bhaskera-data` | Benchmarks against the base model, hyperparameter sweeps, data cleaning |
| `bhaskera-nodes`, `bhaskera-diagnose` | Node health, and a plain explanation of why a job failed |
| `./submit.sh resume --run <run id> [--step N]` | Continue a run that stopped, from its last or an earlier checkpoint (`python -m bhaskera.launcher.resume show` says what it left) |

Environment variables:

| Variable | Meaning |
|---|---|
| `BHASKERA_DIR` | Where Bhaskera is installed. `submit.sh` needs it. Jobs are charged to your login name; set `SLURM_ACCOUNT` only to use another account |
| `BHASKERA_CLUSTER_PROFILE` | A cluster profile file (default `.bhaskera/cluster.yaml` in your run folder) |
| `BHASKERA_PARTITION`, `BHASKERA_MAX_HOURS`, `BHASKERA_GPUS_PER_NODE`, `BHASKERA_CPUS_PER_NODE`, `BHASKERA_MEM_PER_NODE`, `BHASKERA_NET_INTERFACE` | The same facts as the profile, for a cluster without one |
| `BHASKERA_EXCLUDE` | Nodes to keep jobs off (comma separated) |
| `EVAL_HOURS`, `SERVE_HOURS` | Time limit of a benchmark or chat job (default 2 hours) |
| `BHASKERA_HUB_LAUNCH=0` | Turns starting jobs from the home page off |
| `BHASKERA_WORKDIR`, `BHASKERA_RUNS`, `BHASKERA_KIT`, `BHASKERA_RECIPES` | Where the home page finds `submit.sh` and recipes: your run folder, else `~/bhaskera-runs` (made from the shared kit the first time; `BHASKERA_RUNS` changes where) |
| `BHASKERA_STRICT_CONFIG=1` | A config key that does nothing becomes an error, not a warning |
| `SKIP_PREFLIGHT=1`, `SKIP_QOS_CHECK=1` | Submit from the terminal even when a check fails. The home page never does this |

## What Bhaskera does itself

You do not call these directly.

- **Starting a run** (`train.py`, `worker.py`). The driver joins the Ray cluster that `train.sbatch` started (or starts a local one), counts the GPUs, and launches one worker per GPU. Each worker builds the model, wraps it for the chosen parallelism, and hands over to the trainer.
- **Choosing partition and hours** (`cluster_profile.py`, `qos.py`). The partition for a job of N nodes comes from the cluster profile; the time and job limits are read from Slurm, and a job that would be refused is stopped before it is submitted.
- **Watching nodes** (`nodecheck.py`, `diagnose.py`, `bad_nodes.py`). Every node is checked before training, a failed job ends with a plain explanation, and faulty nodes are remembered for three days.
- **The home page** (`dashboard.py`, `ray_hub.py`, `hub_state.py`, `hub_ui.py`, `hub_guard.py`, `launch.py`). One local address for all your jobs, with one SSH link per running job. It accepts connections only from your own user, only for local host names, and the Ray Dashboard is read-only through it. `launch.py` turns the Start form into a `submit.sh` call after checking every field.
- **Chat and benchmarks** (`serve.py`, `evaluate.py`). A small chat server for a tuned model, and the lm-evaluation-harness run that compares it with its base.
- **Sweeps and data tools** (`sweep.py`, `data_tools.py`). Plan, queue and rank many runs within your Slurm limits; inspect, dedupe, filter and split a JSONL file.
