#!/bin/bash -l
#SBATCH --job-name=SpecPT_Exp
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --time=1-23:0:00
#SBATCH --account=redshift
#SBATCH --nodes=1
#SBATCH --partition=sporc
#SBATCH --mem=128g
#SBATCH --gres=gpu:a100:1

EXP_NAME=${1:-exp_000}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
spack load cuda@12.4.0 /obxqih4

cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

python -m src.specpt.training.train \
    --config "configs/${EXP_NAME}.yaml" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT"
