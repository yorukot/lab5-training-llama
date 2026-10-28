#!/usr/bin/env bash
#SBATCH --job-name=lab5-train
#SBATCH --account=acd115198
#SBATCH --partition=8gpus
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1600G
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157,25a-hgpn146,25a-hgpn150,25a-hgpn071,25a-hgpn069,25a-hgpn173
#SBATCH --output=/work/%u/lab5-training-llama/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Submit from the project root: sbatch script/train.sbatch <run ID: AA, AB, AC, ...>
export RUN_NAME="lab5-yoru-${1:?pass the run ID, e.g. sbatch script/train.sbatch AA}-$SLURM_JOB_ID" && echo "Run name: $RUN_NAME"
export HF_HUB_CACHE=/work/$USER/lab5-training-llama/hf_cache/hub
export HF_DATASETS_CACHE=/work/$USER/lab5-training-llama/hf_cache/datasets
export PYTHONUNBUFFERED=1
# One node only, so NCCL's setup can go over loopback. Left to pick, it takes an InfiniBand IP
# interface, and on some nodes that connection fails ("No route to host") or hangs.
export NCCL_SOCKET_IFNAME=lo

uv sync --frozen --extra cu130
exec .venv/bin/torchrun --standalone --nproc_per_node=8 src/training.py
