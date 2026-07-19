#!/bin/bash -l
#SBATCH --partition=sporc
#SBATCH --time=0-00:05:00
#SBATCH --output=/home/ckb2084/research/specpt-hst-sim/outputs/logs/test_gpu_%j.out
#SBATCH --error=/home/ckb2084/research/specpt-hst-sim/outputs/err/test_gpu_%j.err
#SBATCH --gres=gpu:a100:1

source /home/ckb2084/conda/etc/profile.d/conda.sh
conda activate pytorch
which python
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
echo "DONE"
