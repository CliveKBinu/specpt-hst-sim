#!/bin/bash -l
#SBATCH --job-name=ckpt_check
#SBATCH --output=outputs/logs/ckpt_check_%j.out
#SBATCH --time=0-00:02:00
#SBATCH --nodes=1
#SBATCH --partition=sporc
#SBATCH --mem=8g
#SBATCH --gres=gpu:a100:1

source /home/ckb2084/conda/etc/profile.d/conda.sh
conda activate pytorch

cd /home/ckb2084/research/specpt-hst-sim

python3 -c "import torch; c=torch.load('checkpoints/exp_032_best_model.pth',map_location='cpu',weights_only=False); s=c['model_state_dict']; [print(f'{k}: NaN={torch.isnan(s[k]).sum().item()} sum={s[k].sum().item():.4f}') for k in ['prediction.0.weight','prediction.0.bias','prediction.3.weight','prediction.3.bias']]"
