#!/bin/bash
# Install Bhaskera centrally for all PARAM Rudra users (run once, by the maintainer, on a login node).
#
#   bash install-bhaskera.sh                      # installs to /home/apps/bhaskera/1.1.1
#   PREFIX=/some/other/path bash install-bhaskera.sh
#
# Layout created under $PREFIX:
#   src/        Bhaskera source (git clone, pinned to $REF)
#   python/     the Python 3.11 interpreter the environment uses (must be readable by everyone)
#   venv/       Python 3.11 environment: torch 2.6.0+cu124, flash-attn 2.7.3 (prebuilt), bitsandbytes
#   observability/  Prometheus and Grafana (charts in the Ray Dashboard; skip with WITH_CHARTS=0)
#   share/kit/  train.sbatch, tokenize.sbatch, submit.sh, make-quickstart.py, recipes/  (users copy this to /scratch)
# Then install the modulefile:  bhaskera/1.1.1.lua  ->  <a MODULEPATH dir>/bhaskera/1.1.1.lua
#
# The environment is built at its final path on purpose: Python environments cannot be moved.
set -euo pipefail
PREFIX="${PREFIX:-/home/apps/bhaskera/1.1.1}"
REPO="${REPO:-https://github.com/kishoryd/Bhaskera.git}"
REF="${REF:-v1.1.1}"                         # release tag to install
KIT="$(cd "$(dirname "$0")" && pwd)"         # folder holding train.sbatch, submit.sh, recipes/
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

echo "== Bhaskera -> $PREFIX (repo $REPO @ $REF)"
mkdir -p "$PREFIX"
[[ -d "$PREFIX/src/.git" ]] || git clone -q "$REPO" "$PREFIX/src"
git -C "$PREFIX/src" fetch -q origin && git -C "$PREFIX/src" checkout -q "$REF"

# Keep the Python interpreter inside $PREFIX. By default uv puts it in the installing user's home folder, and the
# environment would only be a link to it, which other users cannot read.
export UV_PYTHON_INSTALL_DIR="$PREFIX/python"

# Login nodes have no GPU, so CUDA must be named explicitly or setup.sh installs a CPU-only torch.
cd "$PREFIX/src"
BHASKERA_CUDA=12.4 BHASKERA_VENV="$PREFIX/venv" bash setup.sh

source "$PREFIX/venv/bin/activate"
uv pip install "$FA_WHEEL"                   # prebuilt: no nvcc needed (there is no CUDA 12 toolkit module)
# Exact versions of the environment used for every benchmark run. Newer Ray / grpcio releases break
# multi-node start-up on PARAM Rudra (InvalidAuthToken, or workers timing out waiting for the head node).
uv pip install -r "$KIT/requirements-paramrudra.lock"

mkdir -p "$PREFIX/share/kit/recipes"
cp "$KIT"/train.sbatch "$KIT"/tokenize.sbatch "$KIT"/submit.sh "$KIT"/make-quickstart.py "$PREFIX/share/kit/"
cp "$KIT"/recipes/*.yaml "$PREFIX/share/kit/recipes/"
chmod +x "$PREFIX/share/kit/submit.sh"

# Prometheus and Grafana for the time-series charts of the Ray Dashboard (needs internet; skip with WITH_CHARTS=0).
# Without them the dashboard still lists every node and GPU, only the charts are missing.
if [[ "${WITH_CHARTS:-1}" == "1" ]]; then
    bash "$KIT/install-observability.sh" "$PREFIX/observability"
fi
chmod -R a+rX "$PREFIX"

python - <<'EOF'
import torch, flash_attn, bitsandbytes, bhaskera, transformers, ray, peft
print(f"OK  torch {torch.__version__} (CUDA {torch.version.cuda})  flash-attn {flash_attn.__version__}  "
      f"bitsandbytes {bitsandbytes.__version__}  transformers {transformers.__version__}  ray {ray.__version__}  peft {peft.__version__}")
EOF
echo "== done. Modulefile: set 'root' to $PREFIX in bhaskera/1.1.1.lua and place it in a MODULEPATH directory."
