#!/bin/bash
# Submit a Bhaskera job on PARAM Rudra with the right partition and time limit.
#
#   ./submit.sh <recipe.yaml> <nodes> [hours]      training (runs bhaskera-preflight first)
#   ./submit.sh tokenize <tokenize.yaml>           tokenise data (gpu-debug, 1 node)
#
# Needs:  export BHASKERA_DIR=/scratch/<project>/<username>/Bhaskera
#         export SLURM_ACCOUNT=<your_account>      (your project account, used as sbatch -A)
# Optional: export BHASKERA_EXCLUDE=cbgpuXXXX,...  keep jobs off faulty nodes (passed as sbatch --exclude)
set -euo pipefail
cd "$(dirname "$0")"
: "${BHASKERA_DIR:?export BHASKERA_DIR=/scratch/<project>/<username>/Bhaskera}"
: "${SLURM_ACCOUNT:?export SLURM_ACCOUNT=<your_account>  (your PARAM Rudra project account)}"
mkdir -p logs
EXCL=()
[[ -n "${BHASKERA_EXCLUDE:-}" ]] && EXCL=(--exclude="$BHASKERA_EXCLUDE")

if [[ "${1:-}" == "tokenize" ]]; then
    cfg="${2:?usage: ./submit.sh tokenize <tokenize.yaml>}"
    sbatch -A "$SLURM_ACCOUNT" "${EXCL[@]}" tokenize.sbatch "$cfg"
    exit
fi

cfg="${1:?usage: ./submit.sh <recipe.yaml> <nodes> [hours]}"
nodes="${2:?usage: ./submit.sh <recipe.yaml> <nodes> [hours]}"
hours="${3:-}"

# PARAM Rudra GPU partitions: node range and maximum walltime (hours).
if   (( nodes <= 4 ));   then part=gpu-debug;   max=1
elif (( nodes <= 25 ));  then part=gpu-small;   max=24
elif (( nodes <= 100 )); then part=gpu-large;   max=12
elif (( nodes <= 200 )); then part=gpu-massive; max=6
else echo "at most 200 GPU nodes per job on PARAM Rudra"; exit 1; fi
hours="${hours:-$max}"
if (( hours > max )); then
    echo "note: $part allows at most ${max} h; using ${max} h (checkpoints let the next job resume)"
    hours=$max
fi

# Pipeline recipes need GPUs divisible by the number of stages.
stages=$(awk '/^[[:space:]]*stages:/ {print $2; exit}' "$cfg")
if grep -qE '^[[:space:]]*strategy:[[:space:]]*"?pipeline' "$cfg" && [[ -n "$stages" ]] && (( (2 * nodes) % stages )); then
    echo "pipeline with $stages stages needs a GPU count divisible by $stages; $nodes nodes = $((2*nodes)) GPUs"
    exit 1
fi

echo "== preflight (login node, a few seconds) =="
source "$BHASKERA_DIR/bhaskera-activate.sh"
if ! python -m bhaskera.launcher.preflight --config "$cfg" --num-gpus $(( 2 * nodes )); then
    echo "preflight reported FAIL: fix the items above, or rerun with SKIP_PREFLIGHT=1 to submit anyway"
    [[ "${SKIP_PREFLIGHT:-0}" == 1 ]] || exit 1
fi

name=$(basename "$cfg" .yaml)
jid=$(sbatch --parsable -A "$SLURM_ACCOUNT" "${EXCL[@]}" --comment=bhaskera --job-name="$name" --nodes="$nodes" \
      --partition="$part" --time="${hours}:00:00" train.sbatch "$cfg")
echo "submitted job $jid: $name on $nodes nodes ($((2*nodes)) GPUs), $part, ${hours} h"
echo "follow it:  tail -f logs/${name}_${jid}.out | grep --line-buffered -E '\[step|eval|exit code'"
