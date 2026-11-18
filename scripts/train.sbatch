#!/bin/bash
# Bhaskera training launcher: starts Ray on every node of the job (ray symmetric-run) and runs bhaskera-train.
# This is the launcher used for the 125-run scale matrix (8-128 nodes, A100 80 GB, 2 GPUs per node).
#
# Run from the repository folder (the one that holds bhaskera-activate.sh, written by setup.sh):
#   mkdir -p logs
#   sbatch --nodes=8 scripts/train.sbatch configs/release/rel-ddp-qwen14b.yaml
# Add --partition=..., --time=..., --job-name=... as your cluster needs.
#
# Sized for nodes with 2 GPUs, 48 CPU cores and about 187 GB RAM, used exclusively. Change the
# --gres / --cpus-per-task / --mem lines (and NUM_GPUS below) for other node types.
#SBATCH --job-name=bhaskera
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=170G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
CONFIG="${1:?usage: sbatch scripts/train.sbatch <config.yaml>}"
REPO="${SLURM_SUBMIT_DIR:?run with sbatch from the repository folder}"
NUM_GPUS="${NUM_GPUS:-2}"                       # GPUs per node
IFACE="${NCCL_SOCKET_IFNAME:-ib0}"              # network interface Ray and NCCL use (InfiniBand here)

# Compute nodes usually have no internet: models and data must already be on disk.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MLFLOW_ALLOW_FILE_STORE=true             # MLflow 3 refuses its local file store without this
source "$REPO/bhaskera-activate.sh"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}" TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-3600}"
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# FSDP/HSDP need 32 so the all-gather can overlap compute; DDP/QLoRA use 1.
if grep -qE '^[[:space:]]*strategy:[[:space:]]*"?ddp' "$CONFIG" 2>/dev/null; then _CDMC=1; else _CDMC=32; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-$_CDMC}"
export NCCL_SOCKET_IFNAME="$IFACE"

nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" ip addr show "$IFACE" | awk '/inet / {print $2}' | cut -d/ -f1)
WORKER_COUNT=$(( SLURM_NNODES * NUM_GPUS ))
port=$(( 6379 + (SLURM_JOB_ID % 1000) ))
ip_head="${head_ip}:${port}"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}" RAY_raylet_start_wait_time_s=300 RAY_ADDRESS="$ip_head"

echo "=== bhaskera train: job $SLURM_JOB_ID, $SLURM_NNODES nodes, $WORKER_COUNT GPU workers ==="
echo "config=$CONFIG commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
    ray symmetric-run --address "$ip_head" --min-nodes "$SLURM_NNODES" \
        --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$NUM_GPUS" --metrics-export-port=8080 \
        -- \
        python -m bhaskera.launcher.train --config "$CONFIG" --num-workers "$WORKER_COUNT" --no-dashboard
RC=$?
echo "=== training exit code: $RC ==="
exit $RC
