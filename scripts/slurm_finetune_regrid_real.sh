#!/bin/bash -l
#SBATCH --job-name=SpecPT_FTRegrid
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/err/%x_%j.err
#SBATCH --mail-user=slack:@ckb2084
#SBATCH --mail-type=ALL
#SBATCH --time=0-12:0:00
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

MODE=${MODE:-no_augment}
EXP_NAME=${EXP_NAME:-exp_037}
HEAD_TYPE=${HEAD_TYPE:-simple}
FREEZE_FLAG=""
if [ "${FREEZE_BACKBONE:-false}" = "true" ]; then
    FREEZE_FLAG="--freeze_backbone"
fi

python scripts/finetune_regrid_real.py --mode ${MODE} --exp_name ${EXP_NAME} --head_type ${HEAD_TYPE} ${FREEZE_FLAG}
