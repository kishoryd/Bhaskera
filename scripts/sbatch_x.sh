#!/bin/bash
# sbatch wrapper that excludes nodes recorded bad by scripts/preflight_nodes.sh.
#
#   scripts/sbatch_x.sh [sbatch options] <script> [script args]
#
# Merges three sources into one --exclude: nodes in BHASKERA_BAD_NODES
# (default ./.bad_nodes) recorded within BAD_NODE_TTL_DAYS (default 7; nodes
# get repaired), the target script's own "#SBATCH --exclude=" line, and any
# --exclude passed here. Merging matters: an --exclude on the sbatch command
# line replaces the script's #SBATCH --exclude rather than adding to it.
# DRY_RUN=1 prints the sbatch command instead of running it.
set -euo pipefail

bad_file="${BHASKERA_BAD_NODES:-$(pwd)/.bad_nodes}"
ttl_s=$(( ${BAD_NODE_TTL_DAYS:-7} * 86400 ))
now=$(date +%s)

args=()
cli_excl=""
script=""
for a in "$@"; do
    if [ -z "$script" ] && [[ "$a" == --exclude=* ]]; then
        cli_excl="${a#--exclude=}"
    else
        args+=("$a")
        if [ -z "$script" ] && [[ "$a" != -* ]] && [ -f "$a" ]; then
            script="$a"
        fi
    fi
done

script_excl=""
if [ -n "$script" ]; then
    script_excl=$(grep -m1 -oP '^#SBATCH\s+--exclude=\K\S+' "$script" || true)
fi
recent_bad=""
if [ -f "$bad_file" ]; then
    recent_bad=$(awk -v now="$now" -v ttl="$ttl_s" 'now - $1 <= ttl {print $2}' "$bad_file" | sort -u | paste -sd, -)
fi

merged=$(printf '%s\n' "$script_excl" "$cli_excl" "$recent_bad" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -)
cmd=(sbatch)
[ -n "$merged" ] && cmd+=("--exclude=$merged")
cmd+=("${args[@]}")

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '%q ' "${cmd[@]}"; echo
else
    echo "[sbatch_x] excluding: ${merged:-<none>}"
    "${cmd[@]}"
fi
