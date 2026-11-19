#!/bin/bash
# Bhaskera: tokenise JSONL training data once into a Parquet cache (CPU work: one CPU node, no GPU).
# Normally started by:   ./submit.sh tokenize recipes/tokenize-chatml.yaml
# At the end the log prints the two lines (data.tokenized_path / data.val_tokenized_path)
# to paste into your training recipe.
#SBATCH --job-name=bhaskera-tokenize
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
CONFIG="${1:?usage: sbatch tokenize.sbatch <tokenize-config.yaml>}"
: "${BHASKERA_DIR:?export BHASKERA_DIR=/scratch/<project>/<username>/Bhaskera before submitting}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$BHASKERA_DIR/bhaskera-activate.sh"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"

python -m bhaskera.launcher.tokenize --config "$CONFIG" --split both
echo "=== tokenize exit code: $? ==="
