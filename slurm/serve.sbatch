#!/bin/bash
# Chat with a fine-tuned model on a GPU node: loads the base model and its adapter and serves a small chat API on the node.
# Normally started by ./submit.sh, which picks the partition and time limit for you:
#     ./submit.sh serve --run <mlflow run id>
#     ./submit.sh serve --model <base folder> --adapter <adapter folder>
# The Bhaskera dashboard (bhaskera-dashboard ray) links the server by itself and shows a chat window for it.
# The server stops when the time limit ends (SERVE_HOURS, default 2) or when you cancel the job.
#
# The #SBATCH lines are defaults for submitting this file by hand; ./submit.sh overrides them from the cluster profile.
#SBATCH --job-name=bhaskera-serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=170G
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
: "${BHASKERA_DIR:?export BHASKERA_DIR=<your Bhaskera folder> (the module sets it)}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$BHASKERA_DIR/bhaskera-activate.sh"

echo "=== bhaskera serve: job $SLURM_JOB_ID on $(hostname): $* ==="
python -m bhaskera.launcher.serve "$@"
RC=$?
echo "=== server stopped, exit code: $RC ==="
exit $RC
