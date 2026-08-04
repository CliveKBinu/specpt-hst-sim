#!/bin/bash
#SBATCH --job-name=SpecPT_TransferTrackA
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

EXP_NAME=${1:-tracka_control_z_transfer}
CONFIG=${2:-configs/tracka_control_z.yaml}
CKPT=${3:-checkpoints/tracka_control_z_best_model.pth}

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
    --batch-size 64 \
    --patience 25
