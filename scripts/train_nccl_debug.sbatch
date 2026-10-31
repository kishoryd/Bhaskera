#!/bin/bash
# [NCCL_DEBUG overridable copy of prod_train.sbatch]
# Real (non-diagnostic) FSDP training run: SRC_ROOT's code, full max_steps,
# target_eval_loss early-stop, checkpointing on. Long time limit for actual
# convergence, not the 2h diagnostic budget.
#
# Usage: sbatch --nodes=N --job-name=J --export=ALL,SRC_ROOT=<worktree> prod_train.sbatch <config_path>
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=170G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/nsmapplication/dlapp/wk-kishor/Bhaskera/.claude_tmp/prod_%x_%j.out

set -uo pipefail
CONFIG="${1:?usage: sbatch prod_train.sbatch <config_path>}"
REPO=/scratch/nsmapplication/dlapp/wk-kishor/Bhaskera
SRC_ROOT="${SRC_ROOT:-$REPO}"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MLFLOW_ALLOW_FILE_STORE=true
source "$REPO/bhaskera-activate.sh"
export PYTHONPATH="$SRC_ROOT/src:${PYTHONPATH:-}"

export NCCL_TIMEOUT=3600 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
if grep -qE '^[[:space:]]*strategy:[[:space:]]*"?fsdp' "$CONFIG" 2>/dev/null; then _CDMC=32; else _CDMC=1; fi   # see scripts/submit.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-$_CDMC}
export NCCL_SOCKET_IFNAME=ib0

nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" ip addr show ib0 | awk '/inet / {print $2}' | cut -d/ -f1)
NUM_GPUS=2
WORKER_COUNT=$(( SLURM_NNODES * NUM_GPUS ))
port=$(( 6379 + (SLURM_JOB_ID % 1000) ))
ip_head="${head_ip}:${port}"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}" RAY_raylet_start_wait_time_s=300 RAY_ADDRESS="$ip_head"

echo "=== prod train: $SLURM_JOB_NAME ==="
echo "config=$CONFIG src_root=$SRC_ROOT commit=$(git -C "$SRC_ROOT" rev-parse --short HEAD)"
grep -E "^  name:|max_steps:|eval_steps:|target_eval_loss:|tasks:|every_n_steps:" "$CONFIG"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
    ray symmetric-run --address "$ip_head" --min-nodes "$SLURM_NNODES" \
        --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$NUM_GPUS" --metrics-export-port=8080 \
        -- \
        python -m bhaskera.launcher.train --config "$CONFIG" --num-workers "$WORKER_COUNT" --no-dashboard
RC=$?
echo "=== training exit code: $RC ==="
