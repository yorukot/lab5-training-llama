#!/usr/bin/env bash
#SBATCH --job-name=gpt2-cov-screen
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157,25a-hgpn146,25a-hgpn150,25a-hgpn071,25a-hgpn069,25a-hgpn173,25a-hgpn124
set -euo pipefail
study_root="${1:?pass the study root}"
case_id="${SLURM_ARRAY_TASK_ID:?use a Slurm array}"
exec bash src/exp/covariance_revision/run.sbatch "$study_root" "screen-$case_id" reference "$study_root/configs/$case_id.json" 42
