"""Run isolated optimizer experiments on the pinned GPT-2-derived benchmark.

With no experiment config the upstream training code is unchanged. Experiments
patch only the declared matrix optimizer, reduction, and training budget.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import re
import runpy
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from exp.common import atomic_json

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "vendor" / "nanogpt_optimization"
PROTOCOL = HERE.parent / "gpt2_protocol.json"
VAL_PATTERN = re.compile(r"^step:(\d+)/(\d+) val_loss:([^ ]+) train_time:([^ ]+)s")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(root=VENDOR):
    manifest = json.loads((root / "source_manifest.json").read_text())
    for name, info in manifest["files"].items():
        if sha256(root / name) != info["sha256"]:
            raise ValueError(f"Pinned source changed: {name}")
    return manifest


def shard_metadata(path, checksum=True):
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(1024)
    if len(header) != 1024:
        raise ValueError(f"Truncated shard header: {path}")
    magic, version, count = struct.unpack_from("<iii", header)
    if magic != 20240520 or version != 1 or count <= 0:
        raise ValueError(f"Invalid GPT-2 shard header: {path}")
    if path.stat().st_size != 1024 + 2 * count:
        raise ValueError(f"Truncated or extended shard: {path}")
    info = {"filename": path.name, "tokens": count, "bytes": path.stat().st_size}
    if checksum:
        info["sha256"] = sha256(path)
    return info


def prepare(data_dir, shards=20):
    from huggingface_hub import HfApi, hf_hub_download

    if not 1 <= shards <= 103:
        raise ValueError("Expected 1..103 training shards")
    data_dir.mkdir(parents=True, exist_ok=True)
    repo = "kjj0/fineweb10B-gpt2"
    request_path = data_dir / "download_request.json"
    request: dict[str, Any]
    if request_path.exists():
        request = json.loads(request_path.read_text())
        if request["shards"] != shards or request["repository"] != repo:
            raise ValueError("Existing data request differs; use another directory")
    else:
        request = {
            "repository": repo,
            "revision": HfApi().dataset_info(repo).sha,
            "shards": shards,
        }
        atomic_json(request_path, request)
    names = ["fineweb_val_000000.bin"] + [
        f"fineweb_train_{i:06d}.bin" for i in range(1, shards + 1)
    ]

    def fetch(name):
        path = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision=request["revision"],
            filename=name,
            local_dir=data_dir,
        )
        row = shard_metadata(path)
        print(json.dumps({"verified": name, "tokens": row["tokens"]}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, names))
    manifest = request | {"tokenizer": "gpt2", "files": records}
    atomic_json(data_dir / "dataset_manifest.json", manifest)
    return manifest


def parse_validation(line):
    match = VAL_PATTERN.match(line)
    if match is None:
        return None
    step, budget, loss, seconds = match.groups()
    nll, elapsed = float(loss), float(seconds)
    if not math.isfinite(nll) or not math.isfinite(elapsed) or elapsed < 0:
        raise FloatingPointError("Invalid validation metric")
    return {
        "step": int(step),
        "total_steps": int(budget),
        "nll": nll,
        "ppl": math.exp(nll),
        "training_seconds": elapsed,
        "training_tokens": int(step) * 524288,
    }


def first_target_observation(history, target_ppl, cost="end_to_end_gpu_hours"):
    """Observed crossing interval; never invent an interpolated crossing time."""
    if target_ppl <= 0 or not math.isfinite(target_ppl):
        raise ValueError("Invalid target PPL")
    previous = 0.0
    for row in history:
        now = row[cost]
        if now < previous:
            raise ValueError("Costs must be nondecreasing")
        if row["ppl"] <= target_ppl:
            return {
                "reached": True,
                "cost_interval": [previous, now],
                "first_observed_cost": now,
                "step": row["step"],
            }
        previous = now
    return {"reached": False, "right_censored_after": previous}


def smoke_source(source, steps):
    """Explicit smoke-only patch; all full reproduction code remains unchanged."""
    if not 1 <= steps <= 10:
        raise ValueError("Smoke tests allow 1..10 steps")
    tree = ast.parse(source)
    changed = {}
    values = {"train_steps": steps, "val_tokens": 524288}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in values:
                node.value = ast.Constant(values[target.id])
                changed[target.id] = changed.get(target.id, 0) + 1
    if changed != {"train_steps": 1, "val_tokens": 1}:
        raise ValueError("Upstream structure changed; refuse an ambiguous patch")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


class MetricStream(io.TextIOBase):
    def __init__(self, stream, output, start, world, run=None):
        self.stream, self.output, self.start = stream, output, start
        self.world, self.run = world, run
        self.pending = ""
        self.history = []

    def write(self, text):
        self.stream.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            row = parse_validation(line)
            if row is not None:
                row["end_to_end_seconds"] = time.perf_counter() - self.start
                row["end_to_end_gpu_hours"] = (
                    row["end_to_end_seconds"] * self.world / 3600
                )
                row["training_gpu_hours"] = row["training_seconds"] * self.world / 3600
                self.history.append(row)
                atomic_json(self.output / "history.json", self.history)
                if self.run is not None:
                    self.run.log(
                        {f"benchmark/{k}": v for k, v in row.items()}, step=row["step"]
                    )
        return len(text)

    def flush(self):
        self.stream.flush()


def run_reference(
    output, data_dir, seed=42, smoke_steps=None, use_wandb=True, experiment=None
):
    start = time.perf_counter()
    if smoke_steps is not None and not 1 <= smoke_steps <= 10:
        raise ValueError("Smoke tests allow 1..10 steps")
    # Set before importing Torch/Inductor. Ranks must not concurrently populate
    # the same compiler cache; the batch launcher supplies fresh node-local space.
    if cache_root := os.environ.get("BENCHMARK_CACHE_ROOT"):
        cache = Path(cache_root) / f"rank-{int(os.environ['LOCAL_RANK'])}"
        cache.mkdir(parents=True, exist_ok=False)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    import torch

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world not in (1, 2, 4, 8):
        raise ValueError("Use torchrun with 1, 2, 4 or 8 workers")
    output, data_dir = output.resolve(), data_dir.resolve()
    source_manifest = verify_source()
    protocol = json.loads(PROTOCOL.read_text())
    data_manifest = json.loads((data_dir / "dataset_manifest.json").read_text())
    if source_manifest["commit"] != protocol["upstream_commit"]:
        raise ValueError("Protocol/source mismatch")
    if data_manifest["repository"] != protocol["data"]["repository"]:
        raise ValueError("Wrong data repository")
    if (
        data_manifest["shards"] < protocol["data"]["training_shards"]
        and not smoke_steps
    ):
        raise ValueError("Insufficient training shards for the reference budget")
    # Rank zero rechecks contents; all ranks check headers and sizes.
    for info in data_manifest["files"]:
        current = shard_metadata(data_dir / info["filename"], checksum=rank == 0)
        if current != {key: info[key] for key in current}:
            raise ValueError("Data shard metadata changed")
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    link = output / "data" / "fineweb10B"
    try:
        link.symlink_to(data_dir, target_is_directory=True)
    except FileExistsError:
        if link.resolve() != data_dir:
            raise ValueError("Existing data link points to another dataset") from None
    torch.manual_seed(seed)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.manual_seed_all(seed)
    source_path = VENDOR / "train_gpt_simple.py"
    source = source_path.read_text()
    effective = source
    if experiment is not None:
        from exp.covariance_revision.candidate import experiment_source

        effective = experiment_source(effective, experiment)
    if smoke_steps:
        effective = smoke_source(effective, smoke_steps)
    metadata = {
        "protocol": protocol,
        "source": source_manifest,
        "wrapper_sha256": sha256(__file__),
        "dataset_manifest_sha256": sha256(data_dir / "dataset_manifest.json"),
        "dataset_revision": data_manifest["revision"],
        "effective_source_sha256": hashlib.sha256(effective.encode()).hexdigest(),
        "seed": seed,
        "world_size": world,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "smoke_steps": smoke_steps,
        "comparison_eligible": smoke_steps is None,
        "experiment": experiment,
        "experiment_source_sha256": {
            name: sha256(HERE / name) for name in ("candidate.py", "covariance.py")
        }
        if experiment is not None
        else None,
        "shared_dependency_sha256": {
            name: sha256(HERE.parent / name) for name in ("tcd.py", "common.py")
        },
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
        "compile_threads_per_rank": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        "cache_condition": os.environ.get("BENCHMARK_CACHE_CONDITION", "unspecified"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "time_scope": "runner entry including torch import/setup/compile/data/train/eval/logging; excludes Python startup, wrapper import, shared download and Slurm launch; allocation cost must also be reported",
    }
    wandb_run = None
    if rank == 0:
        if (output / "result.json").exists():
            raise ValueError("Use a fresh output directory; upstream has no resume")
        atomic_json(output / "metadata.json", metadata)
        (output / "effective_source.py").write_text(effective)
        atomic_json(output / "result.json", {"status": "running", "metadata": metadata})
        if use_wandb:
            import wandb

            wandb_run = wandb.init(
                project="lab-training-llama",
                entity=os.environ.get("WANDB_ENTITY", "cerulean-labs"),
                group="gpt2-covariance-revision-v1"
                if experiment
                else "gpt2-optimizer-reference-v1",
                name=output.name,
                job_type="smoke"
                if smoke_steps
                else (
                    "optimizer-experiment" if experiment else "baseline-reproduction"
                ),
                config=metadata,
            )
    logger = MetricStream(sys.stdout, output, start, world, wandb_run)
    original_cwd, original_argv = Path.cwd(), sys.argv[:]
    namespace: dict[str, Any]
    try:
        os.chdir(output)
        execution_path = source_path
        if smoke_steps or experiment is not None:
            execution_path = output / f"effective_source_rank{rank}.py"
            execution_path.write_text(effective)
        sys.argv = [str(execution_path)]
        with contextlib.redirect_stdout(logger if rank == 0 else sys.stdout):
            if smoke_steps or experiment is not None:
                namespace = {"__name__": "__main__", "__file__": str(execution_path)}
                exec(compile(effective, str(execution_path), "exec"), namespace)  # noqa: S102 -- source hash verified above
            else:
                namespace = runpy.run_path(str(source_path), run_name="__main__")
        if rank == 0:
            final = logger.history[-1]
            expected = smoke_steps or (
                experiment["steps"] if experiment else protocol["training"]["steps"]
            )
            parameters = sum(p.numel() for p in namespace["model"].parameters())
            if (
                final["step"] != expected
                or parameters != protocol["model"]["parameters"]
            ):
                raise ValueError("Reference run did not meet its declared budget/model")
            elapsed = time.perf_counter() - start
            result = {
                "status": "completed",
                "metadata": metadata,
                "final": final,
                "actual_parameters": parameters,
                "end_to_end_seconds": elapsed,
                "end_to_end_gpu_hours": elapsed * world / 3600,
                "peak_allocated_bytes_rank0": torch.cuda.max_memory_allocated(),
                "target_crossing": first_target_observation(
                    logger.history, math.exp(3.28)
                ),
                "reference_target_observed": final["nll"] < 3.28
                and smoke_steps is None,
                "claim": (
                    "Single experimental run; requires tuned controls and confirmation"
                    if experiment
                    else "Single reference run; not a new-optimizer comparison"
                ),
                "wandb_url": wandb_run.url if wandb_run else None,
            }
            atomic_json(output / "result.json", result)
            if wandb_run:
                wandb_run.summary.update(
                    {
                        "status": "completed",
                        "final_ppl": final["ppl"],
                        "end_to_end_gpu_hours": result["end_to_end_gpu_hours"],
                    }
                )
    except BaseException as error:
        if rank == 0:
            atomic_json(
                output / "result.json",
                {"status": "error", "error": repr(error), "metadata": metadata},
            )
        raise
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data-dir", type=Path, required=True)
    prep.add_argument("--shards", type=int, default=20)
    run = sub.add_parser("run")
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--smoke-steps", type=int)
    run.add_argument("--no-wandb", action="store_true")
    run.add_argument("--experiment", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data_dir, args.shards)
    else:
        run_reference(
            args.output,
            args.data_dir,
            args.seed,
            args.smoke_steps,
            not args.no_wandb,
            json.loads(args.experiment.read_text()) if args.experiment else None,
        )
