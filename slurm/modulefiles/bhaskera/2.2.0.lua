-- Lmod modulefile: Bhaskera 2.2.0 (distributed LoRA / QLoRA fine-tuning of LLMs) on PARAM Rudra.
-- Install with install-bhaskera.sh, then copy this file to <MODULEPATH dir>/bhaskera/2.2.0.lua
local root = "/home/apps/bhaskera/2.2.0"

whatis("Name: Bhaskera")
whatis("Version: 2.2.0")
whatis("Description: Ray-native distributed LoRA/QLoRA fine-tuning of LLMs (DDP, QLoRA, Pipeline, FSDP, HSDP) on A100 GPU nodes")
whatis("URL: https://github.com/kishoryd/Bhaskera")

family("condaenv")                                -- same family as AIStack Python envs: one at a time

setenv("BHASKERA_HOME", root)
setenv("BHASKERA_DIR",  pathJoin(root, "src"))    -- used by train.sbatch / tokenize.sbatch / submit.sh
setenv("BHASKERA_KIT",  pathJoin(root, "share/kit"))
setenv("VIRTUAL_ENV",   pathJoin(root, "venv"))
prepend_path("PATH",    pathJoin(root, "venv/bin"))
setenv("MLFLOW_ALLOW_FILE_STORE", "true")

help([[
Bhaskera 2.2.0 -- distributed LoRA / QLoRA fine-tuning of large language models
on PARAM Rudra GPU nodes (2 x A100 80 GB per node), 1 to 128+ nodes.

  Includes: torch 2.6.0+cu124, flash-attn 2.7.3, bitsandbytes, transformers 4.57, ray, peft.

Get started (copy the job scripts and tested recipes to your /scratch area):

  module load bhaskera/2.2.0
  cp -r $BHASKERA_KIT /scratch/<project>/<username>/bhaskera-runs
  cd /scratch/<project>/<username>/bhaskera-runs
  export SLURM_ACCOUNT=<your_account>
  ./submit.sh recipes/qwen3-14b-qlora.yaml 8 12     # recipe, nodes, hours

User manual: https://cdacindia.github.io/ParamRudra-20PF-UserManual/bhaskera/
]])
