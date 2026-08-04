#!/bin/bash -l
#SBATCH --job-name=SpecPT_EvalTrackA
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --time=0-02:00:00
#SBATCH --account=redshift
#SBATCH --nodes=1
#SBATCH --partition=sporc
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:1

EXP_NAME=${1:-tracka_small_z}
CONFIG=${2:-configs/${EXP_NAME}.yaml}
CHECKPOINT=${3:-checkpoints/${EXP_NAME}_best_model.pth}

source /home/ckb2084/conda/etc/profile.d/conda.sh
conda activate pytorch

cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

python scripts/eval_tracka_real.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --exp-name "$EXP_NAME"
