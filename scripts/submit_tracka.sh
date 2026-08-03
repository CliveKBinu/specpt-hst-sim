#!/bin/bash
# Chain a Track A autoencoder job into its redshift-training job on the cluster.
#
# Submits the AE training job (tigris) then, only after it completes OK,
# submits the redshift-training job (sporc) with an SLURM afterok dependency.
#
# Usage (run on the cluster login node, or via ssh from the orchestrator):
#     bash scripts/submit_tracka.sh <ae_exp_name> <z_exp_name>
#     bash scripts/submit_tracka.sh autoencoder_tracka_control tracka_control_z
set -euo pipefail

AE_EXP=${1:?usage: submit_tracka.sh <ae_exp_name> <z_exp_name>}
Z_EXP=${2:?usage: submit_tracka.sh <ae_exp_name> <z_exp_name>}

cd /home/ckb2084/research/specpt-hst-sim

echo "Submitting AE job: $AE_EXP"
AE_JOB=$(sbatch --parsable "scripts/slurm_autoencoder_tigris.sh" "$AE_EXP")
echo "AE job id: $AE_JOB"

echo "Submitting redshift job: $Z_EXP (afterok:$AE_JOB)"
sbatch --dependency="afterok:${AE_JOB}" "scripts/slurm_train.sh" "$Z_EXP"

echo "DONE — AE=$AE_JOB chained into $Z_EXP"
