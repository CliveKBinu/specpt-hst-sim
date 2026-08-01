#!/bin/bash
#SBATCH --job-name=SpecPT_Fac
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --partition=sporc
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --account=redshift

EXP_NAME=${1:-fac_stage1}
STAGE=${2:-1}
SIM_SUBSET=${3:-}
INIT_CKPT=${4:-}

source ~/.bashrc
conda activate specpt
cd /home/ckb2084/research/specpt-hst-sim
git pull origin main

SIM_ARG=""
if [ -n "$SIM_SUBSET" ]; then
  SIM_ARG="--sim_subset_size $SIM_SUBSET"
fi

INIT_ARG=""
if [ -n "$INIT_CKPT" ]; then
  INIT_ARG="--init_ckpt $INIT_CKPT"
fi

python scripts/pretrain_factorized_encoder.py \
    --exp_name $EXP_NAME \
    --stage $STAGE \
    --epochs 100 \
    --batch_size 64 \
    --z_lr 3e-4 \
    --encoder_lr 1e-5 \
    --weight_decay 1e-5 \
    --patience 15 \
    --stop_on_drift \
    $SIM_ARG \
    $INIT_ARG
