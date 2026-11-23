#!/bin/bash
# =============================================================================
# Bhaskera multimodal fine-tuning (image / video / audio), Ray-on-SLURM.
#
#   sbatch scripts/submit_multimodal.sh configs/multimodal/fsdp.yaml
#
# gpu-small's QOS needs at least 5 nodes / 10 GPUs / 240 CPUs per job, hence the
# defaults below; override with sbatch flags (e.g. --nodes=8).
# BHASKERA_VENV picks the Python environment (default: ../Bhaskera/.venv, which has
# av and soundfile for video and audio).
# =============================================================================
#SBATCH --job-name=bhaskera-mm
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=170G
#SBATCH --partition=gpu-small
#SBATCH --time=04:00:00
#SBATCH --output=logs/bhaskera-mm_%j_%N.out
#SBATCH --error=logs/bhaskera-mm_%j_%N.err

set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/submit_multimodal.sh <config.yaml> [extra train args]}"
shift
EXTRA_ARGS="$*"
[ -f "$CONFIG" ] || CONFIG="${SLURM_SUBMIT_DIR}/${CONFIG}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MLFLOW_ALLOW_FILE_STORE=true

if command -v module &>/dev/null; then module load cuda/12.4 2>/dev/null || true; fi
BHASKERA_VENV="${BHASKERA_VENV:-${SLURM_SUBMIT_DIR}/../Bhaskera/.venv}"
source "${BHASKERA_VENV}/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-3600}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS=32   # FSDP: 1 serializes the all-gather stream behind compute

if ls /sys/class/infiniband/*/ports/*/state 2>/dev/null | grep -q "4: ACTIVE"; then
    echo "[NCCL] InfiniBand ACTIVE port detected — enabling IB transport"
    export NCCL_IB_DISABLE=0
    IB_DEVICES=$(ls /sys/class/infiniband/ 2>/dev/null | head -1)
    if [ -n "$IB_DEVICES" ]; then
        export NCCL_IB_HCA="${NCCL_IB_HCA:-$IB_DEVICES}"
    fi
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    if ip link show 2>/dev/null | grep -q "ib0"; then
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    fi
elif ibstat 2>/dev/null | grep -q "Rate"; then
    echo "[NCCL] RoCE detected — enabling IB transport (RoCE mode)"
    export NCCL_IB_DISABLE=0
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
else
    echo "[NCCL] No active InfiniBand/RoCE found — using TCP sockets"
    export NCCL_IB_DISABLE=1
    ETH_IF=$(ip -o link show up 2>/dev/null \
             | awk -F': ' '{print $2}' \
             | grep -v -E '^(lo|docker|veth|br-)' \
             | head -1)
    if [ -n "$ETH_IF" ]; then
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$ETH_IF}"
    fi
    export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
    export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
fi

echo "[NCCL] NCCL_IB_DISABLE=${NCCL_IB_DISABLE}  NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-auto}"
echo "[NCCL] NCCL_TIMEOUT=${NCCL_TIMEOUT}  TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT}"

nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes_array[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" ip addr show ib0 | awk '/inet / {print $2}' | cut -d/ -f1)

NUM_GPUS=2
WORKER_COUNT=$(( SLURM_NNODES * NUM_GPUS ))

port=$(( 6379 + (SLURM_JOB_ID % 1000) ))

ip_head="${head_ip}:${port}"
export ip_head

export RAY_raylet_start_wait_time_s=120

mkdir -p logs

echo "========================================================"
echo "  Bhaskera multimodal (Ray symmetric-run)"
echo "  Config    : $CONFIG"
echo "  Job       : $SLURM_JOB_ID"
echo "  Nodes     : $SLURM_NNODES  (head: $head_node / $head_ip)"
echo "  Total GPUs: $WORKER_COUNT"
echo "  Python    : $(command -v python)"
echo "========================================================"

export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
export RAY_raylet_start_wait_time_s=300
export RAY_ADDRESS="$ip_head"

srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    ray symmetric-run \
        --address "$ip_head" \
        --min-nodes "$SLURM_JOB_NUM_NODES" \
        --num-cpus="$SLURM_CPUS_PER_TASK" \
        --num-gpus="$NUM_GPUS" \
        --metrics-export-port=8080 \
        -- \
        python -m bhaskera.launcher.train \
            --config "$CONFIG" \
            --num-workers "$WORKER_COUNT" \
            $EXTRA_ARGS
