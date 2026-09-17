#!/bin/bash
#SBATCH --job-name=bhaskera-llama70b-64
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --reservation=working_nodes
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
##SBATCH --nodelist=cbgpu[0065,0069-0071,0073-0080,0086,0098,0100,0103-0105,0108-0112,0130,0132,0134,0136-0137,0139-0144,0147,0151-0152,0158-0161,0164-0169,0171,0173,0176,0186-0200,0204-0210,0212-0214,0216,0225,0228-0230,0232-0236,0238-0239,0241,0243,0245-0248,0251-0258,0260,0262-0264,0274-0276,0278-0280,0284-0288,0298-0301,0304,0306-0310]
#SBATCH --exclude=cbgpu0003,cbgpu0009,cbgpu0017,cbgpu0024,cbgpu0026,cbgpu0037,cbgpu0059,cbgpu0071,cbgpu0074,cbgpu0081,cbgpu0085,cbgpu0091,cbgpu0100,cbgpu0101,cbgpu0109,cbgpu0110,cbgpu0111,cbgpu0129,cbgpu0133,cbgpu0135,cbgpu0138,cbgpu0146,cbgpu0147,cbgpu0149,cbgpu0157,cbgpu0168,cbgpu0192,cbgpu0214,cbgpu0294,cbgpu0295,cbgpu0299,cbgpu0317
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --output=logs/bhaskera-llama70b-64_%j_%N.out
#SBATCH --error=logs/bhaskera-llama70b-64_%j_%N.err

set -euo pipefail
EXTRA_ARGS="$*"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export MLFLOW_ALLOW_FILE_STORE=true

source "${SLURM_SUBMIT_DIR}/bhaskera-activate.sh"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-3600}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

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
echo "  Bhaskera Ray-on-SLURM (symmetric-run) — Llama-2-70B LoRA"
echo "  Job       : $SLURM_JOB_ID"
echo "  Nodes     : $SLURM_NNODES  (head: $head_node / $head_ip)"
echo "  GPUs/node : $NUM_GPUS"
echo "  Total GPUs: $WORKER_COUNT"
echo "  Address   : $ip_head"
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
	--\
        python -m bhaskera.launcher.train \
            --config "${SLURM_SUBMIT_DIR}/configs/llama2_70b_lora_64node.yaml" \
            --num-workers "$WORKER_COUNT" \
            $EXTRA_ARGS
