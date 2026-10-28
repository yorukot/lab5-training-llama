#!/usr/bin/env bash
#SBATCH --job-name=gpt2-research
#SBATCH --account=acd115198
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=00:10:00
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157,25a-hgpn146,25a-hgpn150,25a-hgpn071,25a-hgpn069,25a-hgpn173
set -euo pipefail
study_root="${1:?pass study root}"
run_name="${2:?pass run name}"
config_path="${3:?pass frozen config path}"
seed="${4:-42}"
mode="${5:-train}"
protocol_path="${6:?pass immutable protocol snapshot}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_PROJECT=lab-training-llama
export WANDB_ENTITY="${WANDB_ENTITY:-cerulean-labs}"
export BENCHMARK_CACHE_ROOT="$(mktemp -d /tmp/gpt2-research-${SLURM_JOB_ID}.XXXXXXXX)"
export TORCHINDUCTOR_COMPILE_THREADS=2
export BENCHMARK_CACHE_CONDITION=fresh_per_run_per_rank_node_local
# Single-node jobs: prevent NCCL selecting an unreachable external interface.
export NCCL_SOCKET_IFNAME=lo
python_bin="/home/$USER/lab5-training-llama/.venv/bin/python"
benchmark_args=(--output "$study_root/runs/$run_name" --data-dir "$study_root/data/fineweb10B"
    --seed "$seed" --experiment "$config_path" --protocol "$protocol_path")
case "$mode" in
  smoke|verify) benchmark_args+=(--smoke-steps 2 --save-checkpoint) ;;
  train) benchmark_args+=(--save-checkpoint) ;;
  *) echo "Unknown research mode: $mode" >&2; exit 2 ;;
esac
if [[ "$mode" == verify ]]; then
    "$python_bin" -m torch.distributed.run --standalone --nproc-per-node=8 \
        -m exp.test_research_optimizers --distributed --nccl
    TORCHINDUCTOR_CACHE_DIR="$BENCHMARK_CACHE_ROOT/profile-inductor" \
      TRITON_CACHE_DIR="$BENCHMARK_CACHE_ROOT/profile-triton" \
      "$python_bin" -m exp.test_research_optimizers --gpu-profile
fi
"$python_bin" -m torch.distributed.run --standalone --nproc-per-node=8 \
    -m exp.gpt2_benchmark run "${benchmark_args[@]}"
if [[ "$mode" == smoke || "$mode" == verify ]]; then
    "$python_bin" -m exp.gpt2_evaluate --checkpoint "$study_root/runs/$run_name/checkpoint.pt" \
        --data-dir "$study_root/data/fineweb10B" --output "$study_root/runs/$run_name/development-eval.json" \
        --start 0 --tokens 524288 --batch-sequences 8 --device cuda
fi
