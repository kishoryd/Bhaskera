#!/bin/bash
# Benchmark a fine-tuned model against its base model with lm-evaluation-harness (one node).
# Normally started by ./submit.sh, which picks the partition and time limit for you:
#     ./submit.sh eval --run <mlflow run id> --tasks arc_easy,hellaswag --limit 200
#     ./submit.sh eval --model <base folder> --adapter <adapter folder> --tasks mmlu
# Run  python -m bhaskera.launcher.evaluate prepare --tasks <tasks>  once on a machine with internet first: compute
# nodes usually have none, so the datasets must already be in the Hugging Face cache.
#
# The #SBATCH lines are defaults for submitting this file by hand; ./submit.sh overrides them from the cluster profile.
#SBATCH --job-name=bhaskera-eval
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
export MLFLOW_ALLOW_FILE_STORE=true
source "$BHASKERA_DIR/bhaskera-activate.sh"

echo "=== bhaskera eval: job $SLURM_JOB_ID on $(hostname): $* ==="
python -m bhaskera.launcher.evaluate run "$@"
RC=$?
echo "=== evaluation exit code: $RC ==="
if [[ $RC -ne 0 ]]; then
    JOBLOG="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/.*StdOut=//p' | head -1)"
    [[ -n "$JOBLOG" ]] && python -m bhaskera.launcher.diagnose "$JOBLOG" --no-remember || true
fi
exit $RC
