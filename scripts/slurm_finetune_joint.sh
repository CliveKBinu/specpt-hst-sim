#!/bin/bash
#SBATCH --job-name=SpecPT_Joint
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

source ~/.bashrc
conda activate specpt
cd /home/ckb2084/research/specpt-hst-sim
git pull

python scripts/finetune_joint_sim_real.py \
    --exp_name exp_048b_joint_corrected \
    --lr_policy flat \
    --lr 1e-5 \
    --epochs 50 \
    --patience 15 \
    --batch_size 128 \
    --real_frac 0.25 \
    --loss_weight_real 1.0 \
    --loss_weight_sim 0.5 \
    --loss_weight_recon 0.1 \
    --sim_subset_size 14000
