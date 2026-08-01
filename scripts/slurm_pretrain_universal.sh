#!/bin/bash
#SBATCH --job-name=SpecPT_USE
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --partition=sporc
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --account=redshift

EXP_NAME=${1:-use_stageA}
STAGE=${2:-A}
RESUME=${3:-}

source ~/.bashrc
conda activate specpt
cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

RESUME_ARG=""
if [ -n "$RESUME" ]; then
  RESUME_ARG="--resume $RESUME"
fi

python scripts/pretrain_universal_encoder.py \
    --exp_name $EXP_NAME \
    --stage $STAGE \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --patience 20 \
    --stop_on_drift \
    $RESUME_ARG
