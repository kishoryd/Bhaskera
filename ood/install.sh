#!/usr/bin/env bash
# Install the Bhaskera app into Open OnDemand's system apps. Run it as root (or with sudo) on EVERY OnDemand web node, from this folder.
#
#   sudo ./install.sh --partition <cpu partition> [--hours 8] [--cluster <id>]
#
# It copies bhaskera/ to the apps folder and writes the three site settings (cluster id, partition, hours) into the form once, so that
# users have nothing to fill in. Run it again to change them. Nothing else on the system is touched.
set -euo pipefail
cd "$(dirname "$0")"

DEST=/var/www/ood/apps/sys/bhaskera
CLUSTERS=/etc/ood/config/clusters.d
PARTITION="" HOURS=8 CLUSTER="" CHECK_SLURM=1
usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="${2:?}"; shift 2 ;;
        --hours)     HOURS="${2:?}"; shift 2 ;;
        --cluster)   CLUSTER="${2:?}"; shift 2 ;;
        --dest)      DEST="${2:?}"; shift 2 ;;
        --clusters-dir) CLUSTERS="${2:?}"; shift 2 ;;
        --skip-slurm-check) CHECK_SLURM=0; shift ;;
        -h|--help)   usage 0 ;;
        *)           echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$PARTITION" ]] || { echo "give the partition the session job runs in: --partition <name> (a CPU partition whose time limit is at least --hours)" >&2; exit 1; }
[[ "$HOURS" =~ ^[0-9]+$ && "$HOURS" -ge 1 ]] || { echo "--hours is a whole number of hours" >&2; exit 1; }

if [[ -z "$CLUSTER" ]]; then
    mapfile -t found < <(find "$CLUSTERS" -maxdepth 1 -name '*.yml' -printf '%f\n' 2>/dev/null | sed 's/\.yml$//' | sort)
    if [[ ${#found[@]} -eq 1 ]]; then
        CLUSTER="${found[0]}"
    else
        echo "cannot tell which cluster to use (found in $CLUSTERS: ${found[*]:-nothing}); give it with --cluster <id>" >&2
        exit 1
    fi
fi
[[ -f "$CLUSTERS/$CLUSTER.yml" ]] || echo "warning: $CLUSTERS/$CLUSTER.yml does not exist here; the portal needs that file for the cluster id '$CLUSTER'" >&2

if [[ "$CHECK_SLURM" == 1 ]] && command -v sinfo >/dev/null 2>&1; then
    sinfo -h -o '%P' 2>/dev/null | tr -d '*' | grep -qx "$PARTITION" || { echo "Slurm has no partition called '$PARTITION'" >&2; exit 1; }
    LIMIT="$(sinfo -h -p "$PARTITION" -o '%l' 2>/dev/null | head -1)"
    echo "partition $PARTITION: time limit $LIMIT (the session asks for $HOURS h)"
fi

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -r bhaskera "$DEST"
sed -i "s|__CLUSTER__|$CLUSTER|; s|__PARTITION__|$PARTITION|; s|__HOURS__|$HOURS|" "$DEST/form.yml.erb"
chmod -R a+rX,go-w "$DEST"
grep -q '__' "$DEST/form.yml.erb" && { echo "a placeholder was left in $DEST/form.yml.erb" >&2; exit 1; }

echo "installed: $DEST   (cluster $CLUSTER, partition $PARTITION, $HOURS h)"
echo "next:      ./check.sh        # on this web node, as a check of the portal's settings"
echo "then:      sign in to the portal, Interactive Apps, Bhaskera, Launch"
