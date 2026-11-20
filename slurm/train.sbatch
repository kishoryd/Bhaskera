#!/bin/bash
# Bhaskera multi-node LoRA / QLoRA fine-tuning on PARAM Rudra (2 x A100 80 GB per node).
# This is the launcher used for every benchmark run in the PARAM Rudra user manual.
#
# Normally started by ./submit.sh, which picks the partition and time limit for you:
#     ./submit.sh recipes/llama-3.3-70b-qlora.yaml 8 12
# Direct use:
#     sbatch -A <account> --nodes=8 --partition=gpu-small --time=12:00:00 train.sbatch <config.yaml>
#
# Needs BHASKERA_DIR = your Bhaskera clone (the folder with bhaskera-activate.sh),
# exported in the shell you submit from (submit.sh checks this).
#SBATCH --job-name=bhaskera
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=170G                     # host RAM per node; the GPU partitions default to only 20 GB
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
CONFIG="${1:?usage: sbatch train.sbatch <config.yaml>}"
: "${BHASKERA_DIR:?export BHASKERA_DIR=/scratch/<project>/<username>/Bhaskera before submitting}"

# Compute nodes have no internet: everything is read from /scratch.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MLFLOW_ALLOW_FILE_STORE=true
source "$BHASKERA_DIR/bhaskera-activate.sh"          # loads CUDA 12.4 + Bhaskera's Python env

# NCCL over InfiniBand (GPUDirect RDMA is not usable on PARAM Rudra GPU nodes).
export NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=WARN NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_TIMEOUT=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DDP / QLoRA: 1 connection keeps gradient all-reduce in order; FSDP / pipeline: 32 lets transfers overlap compute.
if grep -qE '^[[:space:]]*strategy:[[:space:]]*"?ddp' "$CONFIG"; then CDMC=1; else CDMC=32; fi
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-$CDMC}

# One Ray cluster across all nodes of this job, addressed over ib0.
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_ip=$(srun --nodes=1 --ntasks=1 -w "${nodes[0]}" ip -4 addr show ib0 | awk '/inet / {print $2}' | cut -d/ -f1)
if [[ -z "$head_ip" ]]; then
    echo "=== could not read the ib0 address of ${nodes[0]} (Slurm controller unreachable or ib0 down); resubmit ==="
    exit 1
fi
port=$(( 6379 + SLURM_JOB_ID % 1000 ))
export RAY_ADDRESS="${head_ip}:${port}" RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}" RAY_raylet_start_wait_time_s=300

# Time-series charts in the Ray Dashboard (optional). Prometheus collects the metrics of every node and Grafana
# draws them inside the dashboard. install-observability.sh installs both; without them the dashboard still
# lists every node and GPU, only the charts are missing. They run on the first node, next to the Ray head.
OBS="${BHASKERA_OBSERVABILITY:-${BHASKERA_HOME:+$BHASKERA_HOME/observability}}"
OBS_PID=""
if [[ -n "$OBS" && -x "$OBS/prometheus/prometheus" && -x "$OBS/grafana/bin/grafana" ]]; then
    # Both listen on the first node only (127.0.0.1), like the Ray Dashboard, so other users of the cluster cannot reach them.
    export RAY_PROMETHEUS_HOST="http://127.0.0.1:9090" RAY_GRAFANA_HOST="http://127.0.0.1:3000" \
           RAY_GRAFANA_IFRAME_HOST="http://grafana-${SLURM_JOB_ID}.localhost:8265"
    (
        metrics="$RAY_TMPDIR/ray/session_latest/metrics"
        for _ in $(seq 1 120); do            # Ray writes its Prometheus and Grafana settings when the head starts
            [[ -f "$metrics/prometheus/prometheus.yml" && -f "$metrics/grafana/grafana.ini" ]] && break
            sleep 5
        done
        if [[ ! -f "$metrics/grafana/grafana.ini" ]]; then
            echo "=== charts: Ray wrote no Prometheus/Grafana settings, skipped ==="
            exit 0
        fi
        "$OBS/prometheus/prometheus" --config.file="$metrics/prometheus/prometheus.yml" \
            --web.listen-address=127.0.0.1:9090 --storage.tsdb.path="$RAY_TMPDIR/prometheus-data" \
            --storage.tsdb.retention.time=3d > "$RAY_TMPDIR/prometheus.log" 2>&1 &
        GF_PATHS_DATA="$RAY_TMPDIR/grafana-data" GF_PATHS_LOGS="$RAY_TMPDIR/grafana-logs" \
        GF_PATHS_PLUGINS="$RAY_TMPDIR/grafana-plugins" GF_SERVER_HTTP_ADDR=127.0.0.1 GF_SERVER_HTTP_PORT=3000 \
        GF_ANALYTICS_REPORTING_ENABLED=false GF_ANALYTICS_CHECK_FOR_UPDATES=false \
            "$OBS/grafana/bin/grafana" server --homepath "$OBS/grafana" --config "$metrics/grafana/grafana.ini" \
            > "$RAY_TMPDIR/grafana.log" 2>&1 &
        echo "=== charts: Prometheus and Grafana started on $(hostname) ==="
        wait
    ) &
    OBS_PID=$!
    trap '[[ -n "$OBS_PID" ]] && { pkill -P "$OBS_PID" 2>/dev/null; kill "$OBS_PID" 2>/dev/null; }' EXIT
else
    echo "=== charts: Prometheus/Grafana not installed (see install-observability.sh); the Ray Dashboard still lists every node ==="
fi
GPUS_PER_NODE=2
WORKERS=$(( SLURM_NNODES * GPUS_PER_NODE ))

echo "=== bhaskera: job $SLURM_JOB_ID, $SLURM_NNODES nodes, $WORKERS GPUs, config $CONFIG ==="
echo "=== bhaskera: $(git -C "$BHASKERA_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown) in $BHASKERA_DIR ==="

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
    ray symmetric-run --address "$RAY_ADDRESS" --min-nodes "$SLURM_NNODES" \
        --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPUS_PER_NODE" \
        -- \
        python -m bhaskera.launcher.train --config "$CONFIG" --num-workers "$WORKERS" --no-dashboard
RC=$?
echo "=== training exit code: $RC ==="
exit $RC
