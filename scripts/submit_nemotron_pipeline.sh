#!/bin/bash
#SBATCH --job-name=bhaskera-nemotron-pipeline
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --mem=170G
#SBATCH --partition=gpu-small
#SBATCH --time=24:00:00
#SBATCH --output=logs/nemotron_pipeline_%j.out
#SBATCH --error=logs/nemotron_pipeline_%j.err
#
# Tokenizes all 6 Nemotron configs sequentially, submitting each model's
# 8-node training job the moment its tokenization finishes. Runs as a SLURM
# batch job (not tmux/nohup on the login node) because this login node kills
# all user processes on session end (tmux's own server included) — SLURM's
# job manager is the only thing here actually independent of my session.
#
# --nodes=5 satisfies gpu-small's QOS floor (MinTRES cpu=240); the work
# itself is single-node/single-process (Ray Data local execution), so only
# this script's head node (node 0 of the allocation) does anything — the
# other 4 sit idle for the job's duration. Wasteful, but correct, and it's
# the only floor this cluster's QOS actually allows for a CPU-only job.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

declare -A SUBMIT=(
  [qwen3_14b]="submit_qwen3_14b_ddp_8_nemotron.sh"
  [llama33_70b]="submit_llama33_70b_fsdp_8_nemotron.sh"
  [mistral_large]="submit_mistral_large_fsdp_8_nemotron.sh"
  [qwen3_30b_a3b]="submit_qwen3_30b_a3b_fsdp_8_nemotron.sh"
  [gpt_oss_120b]="submit_gpt_oss_120b_fsdp_8_nemotron.sh"
  [qwen3_235b_a22b]="submit_qwen3_235b_a22b_fsdp_8_nemotron.sh"
)

for key in qwen3_14b llama33_70b mistral_large qwen3_30b_a3b gpt_oss_120b qwen3_235b_a22b; do
  cfg="configs/tokenize_${key}_nemotron.yaml"
  echo "[$(date '+%F %T')] START tokenize $key" | tee -a logs/tokenize/_pipeline.log
  python3 -m bhaskera.launcher.tokenize --config "$cfg" --split both \
      > "logs/tokenize/${key}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date '+%F %T')] DONE  tokenize $key" | tee -a logs/tokenize/_pipeline.log
    jobid=$(sbatch --parsable "scripts/${SUBMIT[$key]}" 2>>logs/tokenize/_pipeline.log)
    if [ -n "$jobid" ]; then
      echo "[$(date '+%F %T')] SUBMITTED $key -> job $jobid" | tee -a logs/tokenize/_pipeline.log
    else
      echo "[$(date '+%F %T')] SBATCH_FAIL $key" | tee -a logs/tokenize/_pipeline.log
    fi
  else
    echo "[$(date '+%F %T')] FAIL  tokenize $key (rc=$rc, see logs/tokenize/${key}.log) — NOT submitting" | tee -a logs/tokenize/_pipeline.log
  fi
done
echo "[$(date '+%F %T')] ALL_TOKENIZE_AND_SUBMIT_FINISHED" | tee -a logs/tokenize/_pipeline.log
