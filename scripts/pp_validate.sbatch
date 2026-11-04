#!/bin/bash
# Validation matrix for strategy: pipeline (scripts/pp_validate.py): tiny Llama, 8 GPUs, no Ray.
#   sbatch --nodes=4 --partition=gpu-debug scripts/pp_validate.sbatch
#SBATCH --job-name=pp-validate
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/nsmapplication/dlapp/wk-kishor/Bhaskera/.claude_tmp/pp_validate_%j.out

set -uo pipefail
REPO=/scratch/nsmapplication/dlapp/wk-kishor/Bhaskera
SRC_ROOT="${SRC_ROOT:-$PWD}"
source "$REPO/bhaskera-activate.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_SOCKET_IFNAME=ib0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head" ip addr show ib0 | awk '/inet / {print $2}' | cut -d/ -f1)
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))
WORK="$REPO/.claude_tmp/pp_validate_work_$SLURM_JOB_ID"; mkdir -p "$WORK"
echo "=== pp-validate: nodes=$SLURM_NNODES commit=$(git -C "$SRC_ROOT" rev-parse --short HEAD) ==="
cd "$SRC_ROOT"
srun --kill-on-bad-exit=1 bash -c \
  'RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID exec python -u scripts/pp_validate.py --work "$0"' "$WORK"
echo "=== exit $? ==="
