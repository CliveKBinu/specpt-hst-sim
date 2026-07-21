#!/bin/bash -l
#SBATCH --job-name=SpecPT_LossAblate
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --time=0-6:0:00
#SBATCH --account=redshift
#SBATCH --nodes=1
#SBATCH --partition=sporc
#SBATCH --mem=64g
#SBATCH --gres=gpu:a100:1

source /home/ckb2084/conda/etc/profile.d/conda.sh
conda activate pytorch
spack load cuda@12.4.0 /obxqih4

cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

EXP_NAME=${EXP_NAME:-exp_047_huber_linear}
LOSS=${LOSS:-huber_nmad}

python scripts/finetune_regrid_real.py \
  --mode no_augment --head_type simple --freeze_backbone \
  --batch_size 128 --lr 3e-4 --weight_decay 1e-3 \
  --epochs 300 --patience 30 --seed 42 \
  --val_split 0.1 --test_split 0.1 \
  --loss ${LOSS} --exp_name ${EXP_NAME}
