#!/usr/bin/env bash
#SBATCH --job-name=lab5-logits
#SBATCH --account=acd115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157,25a-hgpn146,25a-hgpn150,25a-hgpn071,25a-hgpn069,25a-hgpn173
#SBATCH --output=/work/%u/lab5-training-llama/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Submit from the project root: sbatch scripts/distillation.sbatch
export HF_HUB_CACHE=/work/$USER/lab5-training-llama/hf_cache/hub
export HF_DATASETS_CACHE=/work/$USER/lab5-training-llama/hf_cache/datasets
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=6

uv sync --frozen --extra cu130
exec .venv/bin/torchrun --standalone --nproc_per_node=8 src/distillation/generate_logits.py
