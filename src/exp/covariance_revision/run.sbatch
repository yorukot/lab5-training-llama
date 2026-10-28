#!/usr/bin/env bash
#SBATCH --job-name=gpt2-opt-reference
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157,25a-hgpn146,25a-hgpn150,25a-hgpn071,25a-hgpn069,25a-hgpn173,25a-hgpn124
set -euo pipefail
study_root="${1:?pass the study root}"
run_name="${2:?pass the run name}"
mode="${3:-reference}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_PROJECT=lab-training-llama
export WANDB_ENTITY="${WANDB_ENTITY:-cerulean-labs}"
export BENCHMARK_CACHE_ROOT="$(mktemp -d /tmp/gpt2-opt-${SLURM_JOB_ID}.XXXXXXXX)"
export TORCHINDUCTOR_COMPILE_THREADS=2
export BENCHMARK_CACHE_CONDITION=fresh_per_run_per_rank_node_local
benchmark_args=(--output "$study_root/runs/$run_name" --data-dir "$study_root/data/fineweb10B" --seed "${5:-42}")
if [[ -n "${4:-}" ]]; then
    benchmark_args+=(--experiment "$4")
fi
if [[ "$mode" == smoke ]]; then
    benchmark_args+=(--smoke-steps 2)
    if [[ -n "${4:-}" ]]; then
        "/home/$USER/lab5-training-llama/.venv/bin/python" -m torch.distributed.run \
            --standalone --nproc-per-node=8 -m exp.covariance_revision.test_revision --distributed
    fi
elif [[ "$mode" != reference ]]; then
    echo "Unknown mode: $mode" >&2
    exit 2
fi
exec "/home/$USER/lab5-training-llama/.venv/bin/python" -m torch.distributed.run \
    --standalone --nproc-per-node=8 -m exp.covariance_revision.driver run "${benchmark_args[@]}"
