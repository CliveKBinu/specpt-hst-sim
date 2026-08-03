#!/bin/bash
#SBATCH --job-name=SpecPT_AutoEnc
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

EXP_NAME=${1:-autoencoder_tracka_control}

source ~/.bashrc
conda activate pytorch
unset PYTHONPATH PYTHONHOME
cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

export PYTHONPATH="/home/ckb2084/research/specpt-hst-sim:${PYTHONPATH}"

python scripts/train_autoencoder.py \
    --config "configs/${EXP_NAME}.yaml"
