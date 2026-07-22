#!/bin/bash
#SBATCH --job-name=SpecPT_RNC_unfrozen
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

EXP_NAME=exp_051_RNC_unfrozen

source ~/.bashrc
conda activate specpt
cd /home/ckb2084/research/specpt-hst-sim
git pull

python scripts/rnc_stage1.py --exp_name $EXP_NAME --batch_size 128
python scripts/rnc_stage2.py --exp_name $EXP_NAME --stage1_ckpt checkpoints/${EXP_NAME}_stage1_best.pth
