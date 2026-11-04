#!/bin/bash
# Pipeline-parallel LoRA benchmark (scripts/pp_lora_bench.py), one process per GPU, no Ray.
#
#   sbatch --nodes=2 --partition=gpu-debug scripts/pp_lora_bench.sbatch --check --pp 4 --microbatches 4
#   sbatch --nodes=8 --partition=gpu-small scripts/pp_lora_bench.sbatch \
#       --model <hf model dir> --pp 4 --microbatches 16
#
# RUNS="args one;args two" runs several configurations in one allocation, one after another.
#SBATCH --job-name=pp-lora
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
#SBATCH --output=pp_lora_%x_%j.out

set -uo pipefail
REPO=/scratch/nsmapplication/dlapp/wk-kishor/Bhaskera      # venv + activate script
SRC_ROOT="${SRC_ROOT:-$PWD}"
source "$REPO/bhaskera-activate.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME=ib0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head" ip addr show ib0 | awk '/inet / {print $2}' | cut -d/ -f1)
echo "=== pp-lora: nodes=$SLURM_NNODES commit=$(git -C "$SRC_ROOT" rev-parse --short HEAD) ==="

run() {
    export MASTER_PORT=$(( 29500 + (SLURM_JOB_ID + RANDOM) % 1000 ))
    echo "--- run: $*"
    srun --kill-on-bad-exit=1 bash -c \
        'RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID exec python -u "$0" "$@"' \
        "$SRC_ROOT/scripts/pp_lora_bench.py" "$@"
    echo "--- exit $?"
}

if [ -n "${RUNS:-}" ]; then
    IFS=';' read -ra R <<< "$RUNS"
    for a in "${R[@]}"; do run $a; done
else
    run "$@"
fi
