#!/usr/bin/env bash
set -euo pipefail
mode="$1"
study_root="$2"
export HF_HUB_CACHE="/work/$USER/lab5-training-llama/hf_cache/hub"
export HF_DATASETS_CACHE="/work/$USER/lab5-training-llama/hf_cache/datasets"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=6
export NCCL_SOCKET_IFNAME=lo
export WANDB_PROJECT=lab-training-llama
export WANDB_ENTITY="${WANDB_ENTITY:-cerulean-labs}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Reuse the existing CUDA environment. No package installation or data preparation.
experiment_python="/home/$USER/lab5-training-llama/.venv/bin/python"
if [[ "$mode" == worker ]]; then
    exec "$experiment_python" -m exp.campaign worker --root "$study_root" --batch "$3" --index "$SLURM_ARRAY_TASK_ID"
else
    exec "$experiment_python" -m exp.campaign advance --root "$study_root" --submit
fi
