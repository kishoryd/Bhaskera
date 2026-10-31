#!/bin/bash
# Kernel-level profile (NVIDIA Nsight Compute) of Qwen3-14B DDP training on one node, 2 GPUs.
# Nsight Systems shows WHEN kernels run; Nsight Compute shows HOW WELL each one runs
# (tensor-core use, memory bandwidth, occupancy, stalls).
#
# Why it is set up this way:
#   * ncu re-runs every profiled kernel several times, so it is slow. It only profiles
#     kernels whose name matches KERNEL_REGEX, only inside the capture window
#     (BHASKERA_NSYS_START_STEP..STOP_STEP, via cudaProfilerStart/Stop), and at most
#     LAUNCH_COUNT of them per process.
#   * NCCL kernels are NOT matched. Replaying a collective on one rank would hang the others.
#   * One node (gpu-debug) is enough: kernel efficiency does not depend on node count.
#
# Usage:
#   sbatch scripts/ncu_train.sbatch configs/profile/qwen3_14b_ddp_1node_ncu.yaml
# Override the kernel filter or count:
#   sbatch --export=ALL,KERNEL_REGEX='gemm',LAUNCH_COUNT=30 scripts/ncu_train.sbatch <config>
#
# Reports: $SRC_ROOT/ncu_reports/<jobid>/*.ncu-rep   (open in the Nsight Compute GUI)
#SBATCH --job-name=ncu-q14b
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=170G
#SBATCH --time=01:00:00
#SBATCH --output=ncu_%x_%j.out

set -uo pipefail
CONFIG="${1:?usage: sbatch ncu_train.sbatch <config_path>}"
REPO=/scratch/nsmapplication/dlapp/wk-kishor/Bhaskera
SRC_ROOT="${SRC_ROOT:-$PWD}"
OUT="$SRC_ROOT/ncu_reports/$SLURM_JOB_ID"
mkdir -p "$OUT"
KERNEL_REGEX="${KERNEL_REGEX:-gemm|flash|fmha}"
LAUNCH_COUNT="${LAUNCH_COUNT:-40}"

module load nsight/12.9.1
command -v ncu >/dev/null || { echo "ncu not found after 'module load nsight/12.9.1'"; exit 2; }

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MLFLOW_ALLOW_FILE_STORE=true
source "$REPO/bhaskera-activate.sh"
export PYTHONPATH="$SRC_ROOT/src:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN NCCL_TIMEOUT=3600 TORCH_NCCL_BLOCKING_WAIT=1
if grep -qE '^[[:space:]]*strategy:[[:space:]]*"?fsdp' "$CONFIG" 2>/dev/null; then _CDMC=32; else _CDMC=1; fi   # see scripts/submit.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-$_CDMC}

# Capture window: the 13th step only (0-based index 12). NVTX labels come on too.
export BHASKERA_NSYS=1 BHASKERA_NSYS_START_STEP=12 BHASKERA_NSYS_STOP_STEP=13 BHASKERA_NSYS_FLUSH_SECONDS=5

NUM_GPUS=2
head_ip=$(hostname -I | awk '{print $1}')
port=$(( 6379 + (SLURM_JOB_ID % 1000) ))
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}" RAY_raylet_start_wait_time_s=300
ip_head="${head_ip}:${port}"; export RAY_ADDRESS="$ip_head"

echo "=== ncu train: $SLURM_JOB_NAME | config=$CONFIG | kernels=/$KERNEL_REGEX/ x$LAUNCH_COUNT per process | reports=$OUT ==="

ncu --target-processes all --profile-from-start off \
    --kernel-name regex:"$KERNEL_REGEX" --launch-count "$LAUNCH_COUNT" \
    --section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
    --section Occupancy --section LaunchStats --section WarpStateStats \
    --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,dram__bytes.sum.per_second \
    --force-overwrite --export "$OUT/%h_%p" \
    ray symmetric-run --address "$ip_head" --min-nodes 1 \
        --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$NUM_GPUS" --metrics-export-port=8080 \
        -- \
        python -m bhaskera.launcher.train --config "$CONFIG" --num-workers "$NUM_GPUS" --no-dashboard
RC=$?
echo "=== training exit code: $RC ==="
ls -la "$OUT" 2>/dev/null
