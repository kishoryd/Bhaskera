#!/bin/bash
# Shared (system-wide) install of Bhaskera on PARAM Rudra, used through `module load bhaskera/<version>`.
# Run by a system administrator, from a login node (needs internet), from the repository folder:
#
#   bash scripts/install_system.sh
#
# Defaults follow the PARAM Rudra AIStack layout. Override any location to stage a test install:
#   ENV_PREFIX=/home/apps/miniconda/envs/bhaskera     Python environment
#   SHARE=/home/apps/bhaskera/<version>               configs, scripts and docs for users
#   MODDIR=/home/apps/AIStack/bhaskera                modulefile folder (file name = version)
#
# The environment is built from scripts/requirements-lock.txt: the exact package versions used for the
# PARAM Rudra test campaign (torch 2.6.0+cu124, transformers 4.57.6, ray 2.56.1, flash-attn 2.7.3, ...).
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(sed -nE 's/^version *= *"([^"]+)".*/\1/p' "$SRC/pyproject.toml")"
ENV_PREFIX="${ENV_PREFIX:-/home/apps/miniconda/envs/bhaskera}"
SHARE="${SHARE:-/home/apps/bhaskera/$VERSION}"
MODDIR="${MODDIR:-/home/apps/AIStack/bhaskera}"
CONDA="${CONDA:-/home/apps/miniconda/bin/conda}"
CUDA_HOME_SPACK="${CUDA_HOME_SPACK:-/home/apps/spack/opt/spack/linux-cascadelake/cuda-12.4.1-gj4i3qdfysl6ubmy36qifghkhu77nxv4}"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
# The shared Miniconda needs these (its plugins and libmamba solver do not load on PARAM Rudra);
# the miniconda module sets the same two variables.
export CONDA_NO_PLUGINS=true CONDA_SOLVER=classic

echo "Bhaskera $VERSION  (source: $SRC, commit $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo n/a))"
echo "  environment : $ENV_PREFIX"
echo "  user files  : $SHARE"
echo "  modulefile  : $MODDIR/$VERSION"

# 1. Python 3.11 environment
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    "$CONDA" create -y -q -p "$ENV_PREFIX" python=3.11 pip
fi
PY="$ENV_PREFIX/bin/python"
"$PY" -m pip install -q --upgrade pip

# 2. Exact package set from the tested environment, then Bhaskera itself (no dependency changes)
"$PY" -m pip install -q --extra-index-url "$TORCH_INDEX" -r "$SRC/scripts/requirements-lock.txt"
"$PY" -m pip install -q --no-deps "$SRC"

# 3. Files users copy or call: tested configs, the Slurm launcher, docs
mkdir -p "$SHARE"
rm -rf "$SHARE/configs" "$SHARE/scripts" "$SHARE/docs"
cp -r "$SRC/configs" "$SRC/scripts" "$SRC/docs" "$SHARE/"
cp "$SRC/README.md" "$SRC/ARCHITECTURE.md" "$SHARE/"
git -C "$SRC" rev-parse HEAD > "$SHARE/COMMIT" 2>/dev/null || true
chmod -R a+rX "$ENV_PREFIX" "$SHARE"

# 4. Modulefile (TCL, same format as the other AIStack modules)
mkdir -p "$MODDIR"
sed -e "s#@VERSION@#$VERSION#g" -e "s#@ENV@#$ENV_PREFIX#g" -e "s#@SHARE@#$SHARE#g" -e "s#@CUDA@#$CUDA_HOME_SPACK#g" \
    "$SRC/scripts/modulefile.tcl.in" > "$MODDIR/$VERSION"
chmod a+r "$MODDIR/$VERSION"

# 5. Smoke test
"$PY" - <<'EOF'
import torch, transformers, ray, peft, bitsandbytes, flash_attn, liger_kernel, bhaskera
from bhaskera.launcher import train, tokenize, preflight, dashboard   # noqa: F401
print("OK  torch", torch.__version__, "| transformers", transformers.__version__, "| ray", ray.__version__,
      "| flash-attn", flash_attn.__version__)
EOF
"$ENV_PREFIX/bin/bhaskera-preflight" --help > /dev/null && echo "OK  bhaskera-preflight"
echo "Done. Users run:  module load bhaskera/$VERSION"
