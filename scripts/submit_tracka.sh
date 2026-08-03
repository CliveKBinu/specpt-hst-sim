#!/bin/bash
# Chain a Track A autoencoder job into its redshift-training job on the cluster.
#
# Submits the AE training job then, only after it completes OK, submits the
# redshift-training job with an SLURM afterok dependency. Both jobs run on the
# SAME cluster (tigris) so the cross-job dependency is valid — SLURM
# dependencies cannot span separate clusters (tigris vs sporc).
#
# Must be run from the tigris login node (tigris.rc.rit.edu) where the tigris
# partition exists.
#
# Usage:
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
sbatch --dependency="afterok:${AE_JOB}" "scripts/slurm_train_tigris.sh" "$Z_EXP"

echo "DONE — AE=$AE_JOB chained into $Z_EXP"
