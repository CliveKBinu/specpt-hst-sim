#!/bin/bash
EXP_NAME=$1
ssh ckb2084@sporcsubmit.rc.rit.edu "cd ~/specpt-hst-sim && sbatch scripts/slurm_train.sh $EXP_NAME"
