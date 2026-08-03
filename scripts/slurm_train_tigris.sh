#!/bin/bash
#SBATCH --job-name=SpecPT_Z
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=64g
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --account=redshift

EXP_NAME=${1:-exp_000}

source ~/.bashrc
conda activate pytorch
unset PYTHONPATH PYTHONHOME
cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

python -m src.specpt.training.train \
    --config "configs/${EXP_NAME}.yaml" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT"
