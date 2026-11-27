#!/bin/bash
# Submit an ExaAI job on PARAM Rudra with the right partition and time limit.
#
#   ./submit.sh <recipe.yaml> <nodes> [hours]      training (runs bhaskera-preflight first)
#   ./submit.sh tokenize <tokenize.yaml>           tokenise data (gpu-debug, 1 node)
#   ./submit.sh resume --run <run id> [--step N] [--nodes N] [--hours H]
#                                                  continue a run that stopped, from its last (or an earlier) checkpoint
#
# Needs:  export BHASKERA_DIR=/scratch/<project>/<username>/ExaAI
#         SLURM_ACCOUNT is optional: jobs are charged to your login name, which is your account here. To use another: export SLURM_ACCOUNT=<account>
# Optional: export BHASKERA_EXCLUDE=cbgpuXXXX,...  keep jobs off faulty nodes (passed as sbatch --exclude)
set -euo pipefail
cd "$(dirname "$0")"
# This folder is the user's run folder: everything the user creates stays in it (see src/bhaskera/paths.py). A copy a user already keeps in the
# home folder (~/mlflow-runs, ~/.cache/huggingface) is used until a copy or a link exists here, so that no run disappears.
export BHASKERA_WORKDIR="$PWD"
if [[ -z "${MLFLOW_TRACKING_URI:-}" ]]; then
    if [[ -e "$PWD/mlflow-runs" || ! -e "$HOME/mlflow-runs" ]]; then export MLFLOW_TRACKING_URI="file://$PWD/mlflow-runs"; else export MLFLOW_TRACKING_URI="file://$HOME/mlflow-runs"; fi
fi
if [[ -z "${HF_HOME:-}" && -z "${HF_DATASETS_CACHE:-}" ]]; then
    if [[ -e "$PWD/.cache/huggingface" || ! -e "$HOME/.cache/huggingface" ]]; then export HF_HOME="$PWD/.cache/huggingface"; fi
fi
: "${BHASKERA_DIR:?export BHASKERA_DIR=/scratch/<project>/<username>/ExaAI}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-$(id -un)}"; export SLURM_ACCOUNT
[[ -f "$BHASKERA_DIR/bhaskera-activate.sh" ]] || { echo "$BHASKERA_DIR/bhaskera-activate.sh is missing: run setup.sh there first (it writes this file for the installation)"; exit 1; }
mkdir -p logs
EXCL=()
# Keep jobs off nodes that earlier jobs of yours found faulty (see: python -m bhaskera.launcher.bad_nodes), and
# off the ones in BHASKERA_EXCLUDE.
SKIP="$(python -m bhaskera.launcher.bad_nodes exclude --extra "${BHASKERA_EXCLUDE:-}" 2>/dev/null || echo "${BHASKERA_EXCLUDE:-}")"
if [[ -n "$SKIP" ]]; then
    EXCL=(--exclude="$SKIP")
    [[ "$SKIP" != "${BHASKERA_EXCLUDE:-}" ]] && echo "skipping nodes that failed before: ${SKIP}"
fi

# DRY_RUN=1 runs every check (partition limits, preflight) and stops before sbatch; the dashboard's "Check first" uses it.
sb() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "dry run: every check passed; nothing was submitted" >&2; echo "dry"; else sbatch --parsable "$@"; fi
}

# A recipe may name a Hugging Face model or dataset (model.name: owner/name, data.train_path: hf://owner/name). They are downloaded here, on the
# login node (compute nodes have no internet), and the job uses a resolved copy of the recipe. A dry run only says what would be downloaded.
fetch_names() {                       # sets FETCHED to the recipe the job should use; ends the script itself for a dry run
    local rc
    FETCHED="$(BHASKERA_NO_FETCH=$([[ "${DRY_RUN:-0}" == 1 ]] && echo 1 || echo 0) python -m bhaskera.launcher.fetch "$1")"; rc=$?
    if [[ $rc -eq 10 ]]; then echo "dry run: the recipe names Hugging Face repositories that would be downloaded (above); run without DRY_RUN to download and check the rest" >&2; exit 0; fi
    [[ $rc -eq 0 ]] || exit 1
}

if [[ "${1:-}" == "tokenize" ]]; then
    cfg="${2:?usage: ./submit.sh tokenize <tokenize.yaml>}"
    fetch_names "$cfg"; cfg="$FETCHED"
    sbatch -A "$SLURM_ACCOUNT" "${EXCL[@]}" tokenize.sbatch "$cfg"
    exit
fi

if [[ "${1:-}" == "resume" ]]; then
    shift
    [[ $# -gt 0 ]] || { echo "usage: ./submit.sh resume --run <mlflow run id> [--step N] [--nodes N] [--hours H]"; exit 1; }
    source "$BHASKERA_DIR/bhaskera-activate.sh"
    # the recipe, nodes and time of the stopped job, and the checkpoint to continue from (or the reason it cannot be done)
    plan="$(python -m bhaskera.launcher.resume plan "$@" --submit-dir "$PWD")" || exit 1
    read -r r_recipe r_nodes r_hours r_step r_run <<< "$plan"
    [[ "$r_step" != "0" ]] && export BHASKERA_RESUME_STEP="$r_step"
    export BHASKERA_RESUME_OF="$r_run"
    set -- "$r_recipe" "$r_nodes" "$r_hours"
fi

if [[ "${1:-}" == "eval" || "${1:-}" == "serve" ]]; then
    what="$1"; shift
    [[ $# -gt 0 ]] || { echo "usage: ./submit.sh $what --run <mlflow run id> ..."; exit 1; }
    source "$BHASKERA_DIR/bhaskera-activate.sh"
    # One node of the partition a one-node job goes to (cluster profile). EVAL_HOURS / SERVE_HOURS set the time limit (default 2 h).
    read -r part max gpn cpn net mem <<< "$(python -m bhaskera.launcher.cluster_profile plan 1)"
    if [[ "$what" == "eval" ]]; then hours="${EVAL_HOURS:-2}"; else hours="${SERVE_HOURS:-2}"; fi
    [[ "$max" != "-" ]] && (( hours > ${max%.*} )) && hours=${max%.*}
    EXTRA=()
    [[ "$part" != "-" ]] && EXTRA+=(--partition="$part")
    [[ "$gpn" != "-" ]]  && EXTRA+=(--gres="gpu:$gpn")
    [[ "$cpn" != "-" ]]  && EXTRA+=(--cpus-per-task="$cpn")
    [[ "$mem" != "-" ]]  && EXTRA+=(--mem="$mem")
    python -m bhaskera.launcher.qos check 1 "$hours" || { echo "not submitted: see the note above"; exit 1; }
    jid=$(sb -A "$SLURM_ACCOUNT" "${EXCL[@]}" "${EXTRA[@]}" --comment="bhaskera-$what" --time="${hours}:00:00" "$what.sbatch" "$@")
    [[ "$jid" == dry ]] && exit 0
    if [[ "$what" == "eval" ]]; then
        echo "submitted job $jid: evaluation, ${part:+$part, }${hours} h"
        echo "follow it:  tail -f logs/bhaskera-eval_${jid}.out"
    else
        echo "submitted job $jid: chat server, ${part:+$part, }${hours} h"
        echo "Loading the model takes a few minutes. Then open the ExaAI dashboard (bhaskera-dashboard ray): the chat window appears."
        echo "follow it:  tail -f logs/bhaskera-serve_${jid}.out"
    fi
    exit
fi

cfg="${1:?usage: ./submit.sh <recipe.yaml> <nodes> [hours]}"
nodes="${2:?usage: ./submit.sh <recipe.yaml> <nodes> [hours]}"
hours="${3:-}"

source "$BHASKERA_DIR/bhaskera-activate.sh"
fetch_names "$cfg"; cfg="$FETCHED"

# Where a job of this size goes, and what each node offers, come from the cluster profile
# (slurm/clusters/*.yaml, found through $BHASKERA_CLUSTER_PROFILE or ~/.bhaskera/cluster.yaml).
# Without a profile: $BHASKERA_PARTITION (else the cluster's default partition), and the hours you give.
plan="$(python -m bhaskera.launcher.cluster_profile plan "$nodes")" || exit 1
read -r part max gpn cpn net mem <<< "$plan"
[[ "$part" == "-" ]] && part=""
if [[ "$max" == "-" ]]; then
    : "${hours:?give the hours: ./submit.sh <recipe.yaml> <nodes> <hours> (no partition limit is known for this cluster)}"
else
    hours="${hours:-$max}"
    if (( hours > ${max%.*} )); then
        echo "note: ${part:-the partition} allows at most ${max} h; using ${max} h (checkpoints let the next job resume)"
        hours=${max%.*}
    fi
fi
gpus_total=""
[[ "$gpn" != "-" ]] && gpus_total=$(( gpn * nodes ))

# Pipeline recipes need GPUs divisible by the number of stages.
stages=$(awk '/^[[:space:]]*stages:/ {print $2; exit}' "$cfg")
if [[ -n "$gpus_total" ]] && grep -qE '^[[:space:]]*strategy:[[:space:]]*"?pipeline' "$cfg" && [[ -n "$stages" ]] && (( gpus_total % stages )); then
    echo "pipeline with $stages stages needs a GPU count divisible by $stages; $nodes nodes = $gpus_total GPUs"
    exit 1
fi

# The partition's QoS limits the hours of a job and how many jobs one user may have submitted and running.
echo "== partition limits =="
if ! python -m bhaskera.launcher.qos check "$nodes" "$hours"; then
    echo "not submitted: see the note above (SKIP_QOS_CHECK=1 submits anyway)"
    [[ "${SKIP_QOS_CHECK:-0}" == 1 ]] || exit 1
fi

echo "== preflight (login node, a few seconds) =="
if ! python -m bhaskera.launcher.preflight --config "$cfg" --num-gpus "${gpus_total:-0}"; then
    echo "preflight reported FAIL: fix the items above, or rerun with SKIP_PREFLIGHT=1 to submit anyway"
    [[ "${SKIP_PREFLIGHT:-0}" == 1 ]] || exit 1
fi

name=$(basename "$cfg" .yaml)
# What the profile knows overrides the defaults written in train.sbatch.
EXTRA=()
[[ -n "$part" ]]      && EXTRA+=(--partition="$part")
[[ "$gpn" != "-" ]]   && EXTRA+=(--gres="gpu:$gpn")
[[ "$cpn" != "-" ]]   && EXTRA+=(--cpus-per-task="$cpn")
[[ "$mem" != "-" ]]   && EXTRA+=(--mem="$mem")
jid=$(sb -A "$SLURM_ACCOUNT" "${EXCL[@]}" "${EXTRA[@]}" --comment=bhaskera --job-name="$name" --nodes="$nodes" \
      --time="${hours}:00:00" train.sbatch "$cfg")
[[ "$jid" == dry ]] && exit 0
echo "submitted job $jid: $name on $nodes nodes${gpus_total:+ ($gpus_total GPUs)}${part:+, $part}, ${hours} h"
echo "follow it:  tail -f logs/${name}_${jid}.out | grep --line-buffered -E '\[step|eval|exit code'"
