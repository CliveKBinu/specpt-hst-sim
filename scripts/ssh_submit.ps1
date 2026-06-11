param($ExpName)
ssh -o ConnectTimeout=60 ckb2084@sporcsubmit.rc.rit.edu "cd /home/ckb2084/research/specpt-hst-sim && sbatch scripts/slurm_train.sh $ExpName"
