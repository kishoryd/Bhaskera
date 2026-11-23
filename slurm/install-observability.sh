#!/bin/bash
# Install Prometheus and Grafana so that the Ray Dashboard of a Bhaskera job can draw its time-series charts
# (GPU, CPU and memory over time for every node). Optional: without them the dashboard still lists every node
# and GPU with its current load, only the charts are missing.
#
#   bash install-observability.sh [TARGET]        # default: $BHASKERA_HOME/observability
#
# Run it once on a login node (it downloads from the internet). Both programs are fetched from their official
# release pages, checked against the published checksums and unpacked as they are, with no root needed:
#   TARGET/prometheus/prometheus      TARGET/grafana/bin/grafana-server
# train.sbatch starts them on the first node of a job when it finds them there.
set -euo pipefail

TARGET="${1:-${BHASKERA_HOME:+$BHASKERA_HOME/observability}}"
: "${TARGET:?give a target folder, or load the bhaskera module first}"
PROM_VERSION="${PROM_VERSION:-2.53.2}"          # Prometheus long-term-support release
GRAFANA_VERSION="${GRAFANA_VERSION:-11.2.2}"    # Grafana OSS

mkdir -p "$TARGET"
WORK="$(mktemp -d "$TARGET/.download.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fetch() {  # fetch <url> <file>
    curl -fL --retry 3 --connect-timeout 20 -sS -o "$2" "$1"
}

echo "== Prometheus $PROM_VERSION"
P="prometheus-$PROM_VERSION.linux-amd64"
fetch "https://github.com/prometheus/prometheus/releases/download/v$PROM_VERSION/$P.tar.gz" "$WORK/$P.tar.gz"
fetch "https://github.com/prometheus/prometheus/releases/download/v$PROM_VERSION/sha256sums.txt" "$WORK/prom.sha256"
( cd "$WORK" && grep " $P.tar.gz\$" prom.sha256 | sha256sum -c - )
tar -xzf "$WORK/$P.tar.gz" -C "$WORK"
rm -rf "$TARGET/prometheus" && mv "$WORK/$P" "$TARGET/prometheus"

echo "== Grafana $GRAFANA_VERSION"
G="grafana-$GRAFANA_VERSION.linux-amd64"
fetch "https://dl.grafana.com/oss/release/$G.tar.gz" "$WORK/$G.tar.gz"
fetch "https://dl.grafana.com/oss/release/$G.tar.gz.sha256" "$WORK/graf.sha256"
EXPECT="$(cut -d' ' -f1 "$WORK/graf.sha256")"
ACTUAL="$(sha256sum "$WORK/$G.tar.gz" | cut -d' ' -f1)"
[[ "$EXPECT" == "$ACTUAL" ]] || { echo "Grafana checksum mismatch"; exit 1; }
echo "$G.tar.gz: OK"
tar -xzf "$WORK/$G.tar.gz" -C "$WORK"
rm -rf "$TARGET/grafana" && mv "$WORK/grafana-v$GRAFANA_VERSION" "$TARGET/grafana"

chmod -R a+rX "$TARGET"
echo "== installed in $TARGET"
"$TARGET/prometheus/prometheus" --version 2>&1 | head -1
"$TARGET/grafana/bin/grafana" --version 2>&1 | head -1 || "$TARGET/grafana/bin/grafana-server" --version 2>&1 | head -1
