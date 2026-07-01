#!/bin/bash -l
#SBATCH --job-name=SpecPT_FTReal
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --time=0-08:0:00
#SBATCH --account=redshift
#SBATCH --nodes=1
#SBATCH --partition=sporc
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:1

source /home/ckb2084/conda/etc/profile.d/conda.sh
conda activate pytorch

cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

STAGE=${STAGE:-1}
INIT=${INIT:-/home/ckb2084/research/specpt-hst-sim/checkpoints/exp_032_best_model.pth}

python scripts/finetune_real_3dhst.py --stage ${STAGE} --init ${INIT}
