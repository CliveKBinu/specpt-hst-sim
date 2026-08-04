#!/bin/bash
#SBATCH --job-name=SpecPT_PartialUnfreeze
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=64g
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --account=redshift

EXP_NAME=${1:-tracka_control_z_partial_unfreeze}
CONFIG=${2:-configs/tracka_control_z.yaml}
CKPT=${3:-checkpoints/tracka_control_z_best_model.pth}
UNFREEZE=${4:-2}
ENCODER_LR=${5:-1e-6}

source ~/.bashrc
conda activate pytorch
unset PYTHONPATH PYTHONHOME
cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

python scripts/finetune_tracka_real.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --exp-name "$EXP_NAME" \
    --epochs 150 \
    --lr 1e-5 \
    --encoder-lr "$ENCODER_LR" \
    --unfreeze-encoder-layers "$UNFREEZE" \
    --batch-size 64 \
    --patience 25
