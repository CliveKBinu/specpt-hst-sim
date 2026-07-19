#!/bin/bash -l
#SBATCH --job-name=SpecPT_AutoEncEval
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --time=0-04:0:00
#SBATCH --account=redshift
#SBATCH --nodes=1
#SBATCH --partition=sporc
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:1

source /home/ckb2084/conda/etc/profile.d/conda.sh
conda activate pytorch

cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

export PYTHONPATH="/home/ckb2084/research/specpt-hst-sim:${PYTHONPATH}"

python scripts/eval_autoencoder_reconstruction.py \
    --ckpt checkpoints/autoencoder_regrid_autoencoder_best.pth \
    --data /home/ckb2084/research/SpecPT/data/grism_specPT_v5.pkl \
    --run-name rose-dragon-2 \
    --output-dir outputs/autoencoder_reconstruction \
    --batch-size 128 \
    --min-snr 2.5
