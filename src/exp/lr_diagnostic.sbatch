#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_CACHE="/work/$USER/lab5-training-llama/hf_cache/hub"
export HF_DATASETS_CACHE="/work/$USER/lab5-training-llama/hf_cache/datasets"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=6
export NCCL_SOCKET_IFNAME=lo
export WANDB_PROJECT=lab-training-llama
export WANDB_ENTITY="${WANDB_ENTITY:-cerulean-labs}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "/home/$USER/lab5-training-llama/.venv/bin/python" -m exp.lr_diagnostic worker --root "$1"
