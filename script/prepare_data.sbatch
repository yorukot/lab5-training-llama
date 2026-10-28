#!/usr/bin/env bash
#SBATCH --job-name=lab5-prepare
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157,25a-hgpn146
#SBATCH --output=/work/%u/lab5-training-llama/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Submit from the project root: sbatch script/prepare_data.sbatch
export HF_HUB_CACHE=/work/$USER/lab5-training-llama/hf_cache/hub
export HF_DATASETS_CACHE=/work/$USER/lab5-training-llama/hf_cache/datasets
export PYTHONUNBUFFERED=1

uv sync --frozen --extra cu130
exec .venv/bin/python src/prepare.py
