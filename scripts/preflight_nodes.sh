#!/bin/bash
# Node health preflight, run inside an sbatch allocation before training:
#
#   source "$SLURM_SUBMIT_DIR/scripts/preflight_nodes.sh" || exit $?
#
# Runs scripts/node_healthcheck.py once per node (timeout PREFLIGHT_TIMEOUT_S,
# default 300s). A node is bad if it reports errors, does not answer before
# the timeout, or its slowest GPU runs below PREFLIGHT_MIN_FRAC (default 0.7)
# of the fastest node's matmul throughput. Bad nodes are appended to
# BHASKERA_BAD_NODES (default $SLURM_SUBMIT_DIR/.bad_nodes) as
# "<epoch> <host> <reason>", and the job returns 3 so it ends in minutes
# instead of hanging for hours. Resubmit with scripts/sbatch_x.sh to exclude them.
# Set BHASKERA_PREFLIGHT=0 to skip.

if [ "${BHASKERA_PREFLIGHT:-1}" = "0" ]; then
    echo "[preflight] skipped (BHASKERA_PREFLIGHT=0)"
    return 0 2>/dev/null || exit 0
fi

_pf_repo="${SLURM_SUBMIT_DIR:-$(pwd)}"
_pf_bad="${BHASKERA_BAD_NODES:-$_pf_repo/.bad_nodes}"
_pf_out="$(mktemp)"
_pf_gpus="${PREFLIGHT_EXPECTED_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"

echo "[preflight] checking $SLURM_NNODES node(s), expecting $_pf_gpus GPU(s) each"
timeout "${PREFLIGHT_TIMEOUT_S:-300}" \
    srun --overlap --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    python "$_pf_repo/scripts/node_healthcheck.py" --expected-gpus "$_pf_gpus" \
    > "$_pf_out" 2>/dev/null

python3 - "$_pf_out" "$_pf_bad" "${PREFLIGHT_MIN_FRAC:-0.7}" $(scontrol show hostnames "$SLURM_JOB_NODELIST") <<'PY'
import json, statistics, sys, time
out_path, bad_path, min_frac, *nodes = sys.argv[1:]
reports = {}
for line in open(out_path):
    try:
        r = json.loads(line)
        reports[r["host"].split(".")[0]] = r
    except (json.JSONDecodeError, KeyError):
        pass
bad = {}
for n in nodes:
    r = reports.get(n.split(".")[0])
    if r is None:
        bad[n] = "no report before timeout (hung or unreachable)"
    elif not r["ok"]:
        bad[n] = "; ".join(r["errors"])[:200]
worst = {n: min(t for t in r["tflops"] if t) for n, r in reports.items()
         if r["ok"] and any(r["tflops"])}
if worst:
    # Reference is the fastest healthy node, not the median: with only a few
    # nodes a slow one drags the median down enough to pass itself (2 nodes at
    # 100% and 60% -> median 80%, and 0.7 x 80% < 60%). A GPU can't beat its
    # hardware peak, so the fastest node is a safe yardstick.
    ref = max(worst.values())
    for n, t in worst.items():
        if t < float(min_frac) * ref:
            bad[n] = f"slow GPU: {t:.0f} TFLOPS vs fastest node {ref:.0f}"
    print(f"[preflight] bf16 matmul TFLOPS (slowest GPU per node): "
          f"median {statistics.median(worst.values()):.0f}, "
          f"min {min(worst.values()):.0f}, max {ref:.0f}")
if bad:
    with open(bad_path, "a") as f:
        for n, why in bad.items():
            f.write(f"{int(time.time())} {n} {why}\n")
    for n, why in bad.items():
        print(f"[preflight] BAD NODE {n}: {why}")
    print(f"[preflight] {len(bad)} bad node(s) recorded in {bad_path}; "
          "resubmit with scripts/sbatch_x.sh to exclude them")
    sys.exit(3)
print(f"[preflight] all {len(nodes)} node(s) healthy")
PY
_pf_rc=$?
rm -f "$_pf_out"
return $_pf_rc 2>/dev/null || exit $_pf_rc
