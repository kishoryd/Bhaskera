#!/bin/bash
# Short training run under NVIDIA Nsight Systems (one report per node).
# Same Ray launch as a normal run, with `nsys profile` wrapped around
# `ray symmetric-run` so the raylet, and the training workers it forks, are traced.
# The capture window is set in-process (BHASKERA_NSYS_START_STEP/STOP_STEP), so only
# a couple of steady-state steps are recorded. When the capture ends, Nsight Systems
# writes its report and shuts the run down (stop-shutdown), so the run ends after the
# captured steps; BHASKERA_NSYS_FLUSH_SECONDS gives every node time to finish writing.
#
# Usage:
#   sbatch --nodes=8 --partition=gpu-small --job-name=nsys-q14b \
#     --export=ALL,SRC_ROOT=<this worktree> scripts/nsys_train.sbatch configs/profile/qwen3_14b_ddp_8node_nsys.yaml
#
# Reports land in $SRC_ROOT/nsys_reports/<jobid>/<hostname>.nsys-rep
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=170G
#SBATCH --time=01:00:00
#SBATCH --output=nsys_%x_%j.out

set -uo pipefail
CONFIG="${1:?usage: sbatch nsys_train.sbatch <config_path>}"
REPO=/scratch/nsmapplication/dlapp/wk-kishor/Bhaskera      # venv + activate script
SRC_ROOT="${SRC_ROOT:-$REPO}"
OUT="$SRC_ROOT/nsys_reports/$SLURM_JOB_ID"
mkdir -p "$OUT"

module load nsight/12.9.1
command -v nsys >/dev/null || { echo "nsys not found after 'module load nsight/12.9.1'"; exit 2; }

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MLFLOW_ALLOW_FILE_STORE=true
source "$REPO/bhaskera-activate.sh"
export PYTHONPATH="$SRC_ROOT/src:${PYTHONPATH:-}"

export NCCL_TIMEOUT=3600 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
if grep -qE '^[[:space:]]*strategy:[[:space:]]*"?fsdp' "$CONFIG" 2>/dev/null; then _CDMC=32; else _CDMC=1; fi   # see scripts/submit.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-$_CDMC}
export NCCL_SOCKET_IFNAME=ib0

# Per-step NVTX markers and the capture window (see utils/nvtx_prof.py)
export BHASKERA_NSYS=1
export BHASKERA_NSYS_START_STEP="${BHASKERA_NSYS_START_STEP:-12}"
export BHASKERA_NSYS_STOP_STEP="${BHASKERA_NSYS_STOP_STEP:-14}"

nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" ip addr show ib0 | awk '/inet / {print $2}' | cut -d/ -f1)
NUM_GPUS=2
WORKER_COUNT=$(( SLURM_NNODES * NUM_GPUS ))
port=$(( 6379 + (SLURM_JOB_ID % 1000) ))
ip_head="${head_ip}:${port}"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}" RAY_raylet_start_wait_time_s=300 RAY_ADDRESS="$ip_head"

echo "=== nsys train: $SLURM_JOB_NAME | config=$CONFIG | reports=$OUT ==="
echo "capture steps ${BHASKERA_NSYS_START_STEP}..$((BHASKERA_NSYS_STOP_STEP - 1)) | commit=$(git -C "$SRC_ROOT" rev-parse --short HEAD)"

# --kill-on-bad-exit=0 / --wait: when the head node finishes first, do not kill the other
# nodes while their nsys is still writing its report (lost 3-6 of 8 reports before this).
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --kill-on-bad-exit=0 --wait=600 \
    nsys profile \
        --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
        --trace-fork-before-exec=true --force-overwrite=true \
        --output="$OUT/%h" \
        ray symmetric-run --address "$ip_head" --min-nodes "$SLURM_NNODES" \
            --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$NUM_GPUS" --metrics-export-port=8080 \
            -- \
            python -m bhaskera.launcher.train --config "$CONFIG" --num-workers "$WORKER_COUNT" --no-dashboard
RC=$?
echo "=== training exit code: $RC ==="
ls -la "$OUT" 2>/dev/null
