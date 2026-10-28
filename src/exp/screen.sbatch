#!/usr/bin/env bash
#SBATCH --job-name=screen150m
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157,25a-hgpn146,25a-hgpn150,25a-hgpn071,25a-hgpn069,25a-hgpn173
#SBATCH --output=/work/%u/lab5-training-llama/logs/%x-%j.out
set -euo pipefail

# One 150M screening run (note/optimizer-research/03-candidates.md). The code runs from a
# snapshot in $ROOT/code, so edits to the repo never reach a queued job:
#   sbatch src/exp/screen.sbatch src/exp/screen/<name>.json   (paths inside the snapshot)
# The run goes to $ROOT/runs/<name>.
ROOT=/work/$USER/lab5-training-llama/experiments/screen150m-v1
export HF_HUB_CACHE=/work/$USER/lab5-training-llama/hf_cache/hub
export HF_DATASETS_CACHE=/work/$USER/lab5-training-llama/hf_cache/datasets
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=6
export WANDB_PROJECT=lab-training-llama
export PYTHONPATH=$ROOT/code/src
cd "$ROOT"
exec /home/$USER/lab5-training-llama/.venv/bin/python -m exp.train \
    --config "code/$1" \
    --output "runs/$(basename "$1" .json)" \
    --data-dir /work/$USER/lab5-training-llama/prepare-66ea6ae/data/bestfit \
    --cache-dir /work/$USER/lab5-training-llama/experiments/hg150m-20261016-v1/study/assets \
    --resume auto
