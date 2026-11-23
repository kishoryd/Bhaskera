#!/usr/bin/env bash
# Check the portal's settings for the Bhaskera app. Run on an OnDemand web node (any user may run it; reading ood_portal.yml may need sudo).
#
#   ./check.sh [--dest /var/www/ood/apps/sys/bhaskera] [--portal-conf /etc/ood/config/ood_portal.yml] [--nodes "cbcn0001 cbgpu0001"]
#
# It checks that the app is installed, that the cluster id exists, and that the portal's host_regex accepts the names of the compute nodes the session can
# land on (otherwise the Open button gets "Forbidden"). It cannot check the firewall: see the last line it prints.
set -uo pipefail
DEST=/var/www/ood/apps/sys/bhaskera
PORTAL=/etc/ood/config/ood_portal.yml
NODES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="${2:?}"; shift 2 ;;
        --portal-conf) PORTAL="${2:?}"; shift 2 ;;
        --nodes) NODES="${2:?}"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
bad=0
ok()   { echo "  ok    $*"; }
fail() { echo "  FAIL  $*"; bad=1; }
note() { echo "  note  $*"; }

echo "The app"
if [[ -f "$DEST/manifest.yml" && -f "$DEST/form.yml.erb" ]]; then ok "$DEST is installed"; else fail "$DEST is missing: run install.sh"; fi
if grep -q '__' "$DEST/form.yml.erb" 2>/dev/null; then fail "the form still has a placeholder (__CLUSTER__, __PARTITION__ or __HOURS__): run install.sh again"; fi
CL="$(sed -n 's/^cluster: *"\(.*\)"/\1/p' "$DEST/form.yml.erb" 2>/dev/null | head -1)"
if [[ -n "$CL" && -f "/etc/ood/config/clusters.d/$CL.yml" ]]; then ok "cluster '$CL' has /etc/ood/config/clusters.d/$CL.yml"; else fail "no cluster file for '$CL' in /etc/ood/config/clusters.d"; fi

echo "The reverse proxy"
if [[ -r "$PORTAL" ]]; then
    RX="$(sed -n "s/^host_regex: *['\"]\{0,1\}\(.*[^'\"]\)['\"]\{0,1\} *$/\1/p" "$PORTAL" | head -1)"
    RN="$(sed -n "s/^rnode_uri: *['\"]\{0,1\}\([^'\" ]*\).*/\1/p" "$PORTAL" | head -1)"
    [[ -n "$RN" ]] && ok "rnode_uri is $RN" || note "rnode_uri is not set in $PORTAL; the default is /rnode"
    if [[ -z "$RX" ]]; then
        note "host_regex is not set in $PORTAL: set it to a pattern that matches your compute node names, then run update_ood_portal and restart the web server"
    else
        ok "host_regex is $RX"
        if [[ -z "$NODES" ]] && command -v sinfo >/dev/null 2>&1; then
            NODES="$(sinfo -N -h -o %N 2>/dev/null | awk '{p=$0; sub(/[0-9]+$/, "", p); if (!(p in seen)) {seen[p]=1; print $0}}' | tr '\n' ' ')"      # one real name of each node type
        fi
        if [[ -n "$NODES" ]]; then
            python3 - "$RX" $NODES <<'PY' || bad=1
import re, sys
rx, names = sys.argv[1], sys.argv[2:]
pat = re.compile(r"^(?:%s)$" % rx)
miss = [n for n in names if not pat.match(n)]
for n in names:
    print(f"  {'ok  ' if pat.match(n) else 'FAIL'}  host_regex {'accepts' if pat.match(n) else 'REJECTS'} {n}")
sys.exit(1 if miss else 0)
PY
        else
            note "no node names to test: pass some with --nodes \"name1 name2\" (a name from each node type the session may land on)"
        fi
    fi
else
    note "cannot read $PORTAL (try sudo); skipped the host_regex and rnode checks"
fi

echo "Not checked here"
note "the firewall: the web node must reach the compute nodes on the session's port (any port from 1024 up). If another interactive app (Jupyter, a desktop) already works, it does."
exit $bad
