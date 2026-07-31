#!/bin/bash
#SBATCH --job-name=SpecPT_USE_eval
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --partition=sporc
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --account=redshift

EXP_NAME=${1:-use_stageA}
CKPT=${2:-}

source ~/.bashrc
conda activate specpt
cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

if [ -z "$CKPT" ]; then
  python scripts/eval_universal_latent.py --exp_name $EXP_NAME
else
  python scripts/eval_universal_latent.py --exp_name $EXP_NAME --ckpt $CKPT
fi
