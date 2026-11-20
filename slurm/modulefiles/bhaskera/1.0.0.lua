-- Lmod modulefile: Bhaskera 1.0.0 (distributed LoRA / QLoRA fine-tuning of LLMs) on PARAM Rudra.
-- Install with install-bhaskera.sh, then copy this file to <MODULEPATH dir>/bhaskera/1.0.0.lua
local root = "/home/apps/bhaskera/1.0.0"
local quickstart = "/home/apps/bhaskera/quickstart"   -- shared models, tokenised data and ready recipes (see make-quickstart.py)

whatis("Name: Bhaskera")
whatis("Version: 1.0.0")
whatis("Description: Ray-native distributed LoRA/QLoRA fine-tuning of LLMs (DDP, QLoRA, Pipeline, FSDP, HSDP) on A100 GPU nodes")
whatis("URL: https://github.com/kishoryd/Bhaskera")

family("condaenv")                                -- same family as AIStack Python envs: one at a time

setenv("BHASKERA_HOME", root)
setenv("BHASKERA_DIR",  pathJoin(root, "src"))    -- used by train.sbatch / tokenize.sbatch / submit.sh
setenv("BHASKERA_KIT",  pathJoin(root, "share/kit"))
setenv("BHASKERA_QUICKSTART", quickstart)
setenv("VIRTUAL_ENV",   pathJoin(root, "venv"))
prepend_path("PATH",    pathJoin(root, "venv/bin"))
setenv("MLFLOW_ALLOW_FILE_STORE", "true")

help([[
Bhaskera 1.0.0 -- distributed LoRA / QLoRA fine-tuning of large language models
on PARAM Rudra GPU nodes (2 x A100 80 GB per node), 1 to 128+ nodes.

  Includes: torch 2.6.0+cu124, flash-attn 2.7.3, bitsandbytes, transformers 4.57, ray, peft.

Quick start (models, tokenised data and ready recipes are on shared storage; nothing to download):

  module load bhaskera/1.0.0
  cp -r $BHASKERA_KIT /scratch/<project>/<username>/bhaskera-runs
  cd /scratch/<project>/<username>/bhaskera-runs
  cp $BHASKERA_QUICKSTART/recipes/*.yaml recipes/
  export SLURM_ACCOUNT=<your_account>
  ./submit.sh recipes/qwen3-14b-qlora.yaml 8 1        # recipe, nodes, hours

User manual: https://cdacindia.github.io/ParamRudra-20PF-UserManual/bhaskera/
]])
