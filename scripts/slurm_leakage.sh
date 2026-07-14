#!/bin/bash -l
#SBATCH --job-name=SpecPT_Leakage
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --time=0-12:0:00
#SBATCH --account=redshift
#SBATCH --nodes=1
#SBATCH --partition=sporc
#SBATCH --mem=256g
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

DATA_PATH="/home/ckb2084/research/SpecPT/data/grism_specPT_augumented_v2_more_data.pkl"

source /home/ckb2084/conda/etc/profile.d/conda.sh
conda activate pytorch
spack load cuda@12.4.0 /obxqih4

cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

python scripts/analyze_leakage.py \
    --data "$DATA_PATH" \
    --out outputs/leakage \
    --dup-batch-mb 512 \
    --self-dup-sample 5000 \
    --seed 42

# Quick smoke-test alternative (uncomment to run on 50k rows):
# python scripts/analyze_leakage.py \
#     --data "$DATA_PATH" \
#     --out outputs/leakage_smoke \
#     --rows-limit 50000 \
#     --dup-batch-mb 256 \
#     --self-dup-sample 1000 \
#     --seed 42
