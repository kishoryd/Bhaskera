#!/bin/bash
# Network facts + FSDP-style all-gather bandwidth, one variant set per job.
#
#   sbatch --nodes=8 --partition=gpu-small scripts/allgather_bench.sbatch
#   sbatch --nodes=8 --partition=gpu-small --export=ALL,NCCL_PROTO=Simple scripts/allgather_bench.sbatch
#
# BENCH_GROUPS (space-separated) picks the gathers measured, default "world group:16 group:4".
# For each NCCL setting in BENCH_SWEEP (space-separated KEY=VALUE[;KEY=VALUE]), the benchmark
# runs once more with that setting; "base" means the environment as submitted.
#SBATCH --job-name=ag-bench
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=ag_bench_%j.out

set -uo pipefail
REPO=/scratch/nsmapplication/dlapp/wk-kishor/Bhaskera
SRC_ROOT="${SRC_ROOT:-$PWD}"
source "$REPO/bhaskera-activate.sh"
export NCCL_SOCKET_IFNAME=ib0
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))
export BENCH_GROUPS="${BENCH_GROUPS:-world group:16 group:4}"

echo "=== node facts ($(hostname)) ==="
nvidia-smi --query-gpu=name,power.limit,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader
nvidia-smi topo -m 2>/dev/null | sed -n '1,8p'
(ibstat 2>/dev/null || ibv_devinfo 2>/dev/null) | grep -E "CA '|State:|Rate:|Link layer|hca_id|active_speed|active_width" | head -24
ls /sys/class/infiniband 2>/dev/null | tr '\n' ' '; echo
echo "NIC NUMA node: $(cat /sys/class/infiniband/mlx5_0/device/numa_node 2>/dev/null)"
lscpu | grep -E "^NUMA node[0-9] CPU|^Socket|^Model name" ; command -v numactl || echo "numactl: not installed"
lsmod | grep -iE "peermem|nv_peer" || echo "nvidia-peermem: not loaded"

for setting in ${BENCH_SWEEP:-base}; do
    echo "=== setting: $setting ==="
    ENVSET=(); SRUNX=(); WRAP=()
    [ "$setting" != base ] && IFS=';' read -ra ENVSET <<< "$setting"
    # BIND=none|nic|gpu: CPU/memory placement of each rank (NCCL stages through host
    # memory when GPUDirect RDMA is off, so the staging buffers' NUMA node matters).
    for kv in "${ENVSET[@]}"; do case "$kv" in
        BIND=none) SRUNX=(--cpu-bind=none) ;;
        BIND=nic)  n=$(cat /sys/class/infiniband/mlx5_0/device/numa_node); SRUNX=(--cpu-bind=none); WRAP=(numactl --cpunodebind=$n --membind=$n) ;;
        BIND=gpu)  SRUNX=(--cpu-bind=none); WRAP=(numactl --cpunodebind=0 --membind=0) ;;
    esac; done
    env "${ENVSET[@]}" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
        srun --kill-on-bad-exit=1 "${SRUNX[@]}" "${WRAP[@]}" python -u "$SRC_ROOT/scripts/allgather_bench.py" 2>&1 \
        | grep -E "^\[bench\]|GPU Direct|Using network" \
        | sort | uniq -c | sort -rn | awk '{$1=""; print}' | sort | head -30
done
