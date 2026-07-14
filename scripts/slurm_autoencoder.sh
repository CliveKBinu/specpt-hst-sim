#!/bin/bash -l
#SBATCH --job-name=SpecPT_AutoEnc
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

conda activate pytorch
spack load cuda@12.4.0 /obxqih4

cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

export PYTHONPATH="/home/ckb2084/research/specpt-hst-sim:${PYTHONPATH}"

python scripts/train_autoencoder.py \
    --config configs/autoencoder_regrid.yaml
