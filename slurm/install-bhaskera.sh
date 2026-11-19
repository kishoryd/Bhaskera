#!/bin/bash
# Install Bhaskera centrally for all PARAM Rudra users (run once, by the maintainer, on a login node).
#
#   bash install-bhaskera.sh                      # installs to /home/apps/bhaskera/2.2.0
#   PREFIX=/some/other/path bash install-bhaskera.sh
#
# Layout created under $PREFIX:
#   src/        Bhaskera source (git clone, pinned to $REF)
#   venv/       Python 3.11 environment: torch 2.6.0+cu124, flash-attn 2.7.3 (prebuilt), bitsandbytes
#   share/kit/  train.sbatch, tokenize.sbatch, submit.sh, recipes/  (users copy this to /scratch)
# Then install the modulefile:  bhaskera/2.2.0.lua  ->  <a MODULEPATH dir>/bhaskera/2.2.0.lua
#
# The environment is built at its final path on purpose: Python environments cannot be moved.
set -euo pipefail
PREFIX="${PREFIX:-/home/apps/bhaskera/2.2.0}"
REPO="${REPO:-https://github.com/kishoryd/Bhaskera.git}"
REF="${REF:-70dfcfc}"                        # the commit validated for the PARAM Rudra user manual
KIT="$(cd "$(dirname "$0")" && pwd)"         # folder holding train.sbatch, submit.sh, recipes/
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

echo "== Bhaskera -> $PREFIX (repo $REPO @ $REF)"
mkdir -p "$PREFIX"
[[ -d "$PREFIX/src/.git" ]] || git clone -q "$REPO" "$PREFIX/src"
git -C "$PREFIX/src" fetch -q origin && git -C "$PREFIX/src" checkout -q "$REF"

# Login nodes have no GPU, so CUDA must be named explicitly or setup.sh installs a CPU-only torch.
cd "$PREFIX/src"
BHASKERA_CUDA=12.4 BHASKERA_VENV="$PREFIX/venv" bash setup.sh

source "$PREFIX/venv/bin/activate"
uv pip install "$FA_WHEEL"                   # prebuilt: no nvcc needed (there is no CUDA 12 toolkit module)
# Versions validated on PARAM Rudra (newer Ray needs an auth token between nodes and breaks multi-node jobs)
uv pip install "ray[train,data,default]==2.56.1" peft==0.20.0 accelerate==1.14.0 liger-kernel==0.8.1 mlflow==3.15.0 pyarrow==25.0.0 bitsandbytes==0.50.2

mkdir -p "$PREFIX/share/kit/recipes"
cp "$KIT"/train.sbatch "$KIT"/tokenize.sbatch "$KIT"/submit.sh "$PREFIX/share/kit/"
cp "$KIT"/recipes/*.yaml "$PREFIX/share/kit/recipes/"
chmod +x "$PREFIX/share/kit/submit.sh"
chmod -R a+rX "$PREFIX"

python - <<'EOF'
import torch, flash_attn, bitsandbytes, bhaskera, transformers, ray, peft
print(f"OK  torch {torch.__version__} (CUDA {torch.version.cuda})  flash-attn {flash_attn.__version__}  "
      f"bitsandbytes {bitsandbytes.__version__}  transformers {transformers.__version__}  ray {ray.__version__}  peft {peft.__version__}")
EOF
echo "== done. Modulefile: set 'root' to $PREFIX in bhaskera/2.2.0.lua and place it in a MODULEPATH directory."
