#!/bin/bash
# 2-minute check that Nsight Compute can read GPU performance counters here.
# If counters are restricted to admins, ncu prints ERR_NVGPUCTRPERM and kernel profiling
# is not possible for a normal user, so run this before anything bigger.
#
#   sbatch scripts/ncu_smoke.sbatch
#SBATCH --job-name=ncu-smoke
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=ncu_smoke_%j.out

set -uo pipefail
REPO=/scratch/nsmapplication/dlapp/wk-kishor/Bhaskera
module load nsight/12.9.1
source "$REPO/bhaskera-activate.sh"
command -v ncu || { echo "ncu not found"; exit 2; }
ncu --version | tail -1

PYFILE=$(mktemp --suffix=.py)
cat > "$PYFILE" <<'PY'
import torch
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
print("matmul done", c.float().abs().mean().item())
PY

# one GEMM kernel, tensor-core and memory throughput as % of peak
ncu --kernel-name regex:gemm --launch-skip 2 --launch-count 1 \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum \
    python "$PYFILE"
echo "=== ncu exit code: $? ==="
