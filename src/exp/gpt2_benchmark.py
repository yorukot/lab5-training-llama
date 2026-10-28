"""Reproduce a pinned, independently tuned GPT-2-derived optimizer benchmark.

The historical reference executes unmodified upstream training code. Explicit
research configurations apply narrow checked optimizer and evaluation patches.
This wrapper records provenance, seeds, timing, W&B and optional checkpoints;
the pinned architecture, training data order and auxiliary AdamW are retained.
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

from exp.common import atomic_json, fingerprint

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "nanogpt_optimization"
PROTOCOL = HERE / "gpt2_protocol.json"
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


def validate_dataset_files(data_dir, manifest):
    """Bind the upstream filename globs to exactly the verified manifest."""
    shards = manifest.get("shards")
    if type(shards) is not int or not 1 <= shards <= 103:
        raise ValueError("Invalid manifest training-shard count")
    names = [record["filename"] for record in manifest["files"]]
    expected = {"fineweb_val_000000.bin"} | {
        f"fineweb_train_{index:06d}.bin" for index in range(1, shards + 1)
    }
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError(
            "Dataset manifest does not declare the exact benchmark shard set"
        )
    actual = {
        path.name
        for pattern in ("fineweb_train_*.bin", "fineweb_val_*.bin")
        for path in Path(data_dir).glob(pattern)
    }
    if actual != expected:
        raise ValueError("Data loader globs differ from the verified dataset manifest")


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


def validation_source(source, token_offset, token_count, frequency=None):
    """Replace only validation acquisition, count and optional cadence."""
    if (
        type(token_offset) is not int
        or token_offset < 0
        or type(token_count) is not int
        or token_count <= 0
        or token_count % 524288
        or (frequency is not None and (type(frequency) is not int or frequency <= 0))
    ):
        raise ValueError("Invalid training validation configuration")
    tree = ast.parse(source)
    changed = {"tokens": 0, "loader": 0, "frequency": 0}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = ast.unparse(node.targets[0])
        if target == "val_tokens":
            node.value = ast.Constant(token_count)
            changed["tokens"] += 1
        elif target == "(val_inputs, val_targets)":
            if (
                ast.unparse(node.value)
                != "next(distributed_data_generator('data/fineweb10B/fineweb_val_*.bin', val_tokens))"
            ):
                raise ValueError("Upstream validation acquisition changed")
            node.value = ast.Call(
                ast.Name("load_validation_batch", ast.Load()),
                [
                    ast.Constant("data/fineweb10B/fineweb_val_000000.bin"),
                    ast.Constant(token_offset),
                    ast.Name("val_tokens", ast.Load()),
                ],
                [],
            )
            changed["loader"] += 1
        elif target == "val_step_freq":
            if frequency is not None:
                node.value = ast.Constant(frequency)
            changed["frequency"] += 1
    if changed != {"tokens": 1, "loader": 1, "frequency": 1}:
        raise ValueError(f"Ambiguous upstream validation patch: {changed}")
    tree.body.insert(
        1, ast.ImportFrom("exp.gpt2_evaluate", [ast.alias("load_validation_batch")], 0)
    )
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def validation_configuration(
    protocol, token_offset=None, token_count=None, frequency=None, smoke_steps=None
):
    """Training validation may use only the registered development interval."""
    development = protocol.get("evaluation", {}).get(
        "development",
        {"token_offset": 0, "token_count": protocol["training"]["validation_tokens"]},
    )
    offset = development["token_offset"] if token_offset is None else token_offset
    count = development["token_count"] if token_count is None else token_count
    if smoke_steps is not None:
        count = 524288
    if (
        type(offset) is not int
        or type(count) is not int
        or offset < development["token_offset"]
        or count <= 0
        or offset + count > development["token_offset"] + development["token_count"]
        or count % 524288
        or (frequency is not None and (type(frequency) is not int or frequency <= 0))
    ):
        raise ValueError("Training validation must remain within the development slice")
    confirmation = protocol.get("evaluation", {}).get("confirmation", {})
    if (
        "token_offset" in confirmation
        and "token_count" in confirmation
        and max(offset, confirmation["token_offset"])
        < min(
            offset + count, confirmation["token_offset"] + confirmation["token_count"]
        )
    ):
        raise ValueError("Training validation overlaps reserved confirmation targets")
    return {"token_offset": offset, "token_count": count, "frequency": frequency}


def research_identity(
    protocol, experiment, source_manifest, dataset_manifest_hash, dataset_revision
):
    """Record exact code and data independently of method hyperparameters."""
    source_names = ["gpt2_benchmark.py", "gpt2_evaluate.py"]
    if experiment is not None:
        source_names += ["gpt2_candidate.py", "research_optimizers.py", "tcd.py"]
    hashes = {name: sha256(HERE / name) for name in source_names}
    method = experiment["method"] if experiment else "muon"
    training = {
        k: v for k, v in protocol["training"].items() if k != "matrix_optimizer"
    }
    training["steps"] = experiment["steps"] if experiment else training["steps"]
    training["total_training_tokens"] = training["steps"] * training["tokens_per_step"]
    training["upstream_source_sha256"] = source_manifest["files"][
        "train_gpt_simple.py"
    ]["sha256"]
    shared = {
        "architecture": protocol["model"],
        "data_sha256": dataset_manifest_hash,
        "tokenizer": protocol["data"]["tokenizer"],
        "dataset_revision": dataset_revision,
        "data_order": "deterministic_sorted_training_shards; seed controls initialization",
        "training_config_sha256": fingerprint(training),
    }
    return {
        "method": method,
        "shared_identity": shared,
        "method_config_sha256": fingerprint(
            experiment or protocol["training"]["matrix_optimizer"]
        ),
        "method_source_sha256": hashes,
    }


def validate_protocol(protocol):
    """A research protocol cannot silently redefine the pinned training task."""
    reference = json.loads(PROTOCOL.read_text())
    if protocol.get("model") != reference["model"]:
        raise ValueError("Research protocol must preserve the pinned model")
    for field in (
        "steps",
        "tokens_per_step",
        "total_training_tokens",
        "microbatch_sequences_per_gpu",
        "cooldown_fraction",
        "warmup_steps",
        "aux_optimizer",
    ):
        if protocol.get("training", {}).get(field) != reference["training"][field]:
            raise ValueError(f"Research protocol changes fixed training field: {field}")
    if protocol.get("data", {}).get("tokenizer") != "gpt2":
        raise ValueError("Research protocol changes tokenizer")
    return protocol


def adaptive_optimizer_diagnostics(model, optimizer, rank):
    """Read only small, rank-owned adaptive state after the training loop ends."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    parameters = []
    for group in optimizer.param_groups:
        if group.get("method") != "shampoo_adaptive":
            continue
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            blocks = []
            for index, block in enumerate(state["blocks"]):
                scalars = {}
                for key in ("left_alpha", "right_alpha", "left_scatter", "right_scatter"):
                    value = block[key]
                    if value.ndim != 1:
                        raise ValueError(f"Expected one scalar per tile for adaptive diagnostic {key}")
                    scalars[key] = value.detach().cpu().tolist()
                blocks.append({
                    "block_index": index,
                    "left_tile_dimension": int(block["left"].shape[-1]),
                    "right_tile_dimension": int(block["right"].shape[-1]),
                    "last_root_refresh_step": int(block["last_refresh_step"]),
                    **scalars,
                })
            parameters.append({
                "name": names[id(parameter)], "shape": list(parameter.shape),
                "optimizer_step": int(state["step"]), "alpha_max": group["alpha_max"],
                "update_frequency": group["update_frequency"], "blocks": blocks,
            })
    return {
        "schema_version": 1, "method": "shampoo_adaptive", "rank": rank,
        "scope": "rank-owned optimizer state after training; no cross-rank aggregation",
        "alpha_observation": "left_alpha/right_alpha were applied at last_root_refresh_step; not a training-time average",
        "scatter_observation": "left_scatter/right_scatter are current at optimizer_step and may be newer than the applied alpha",
        "tile_order": "blocks follow homogeneous region order; scalar arrays follow row-major tile order within each region",
        "parameters": parameters,
    }


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
    output,
    data_dir,
    seed=42,
    smoke_steps=None,
    use_wandb=True,
    experiment=None,
    protocol_path=PROTOCOL,
    save_checkpoint=False,
    validation_start=None,
    validation_tokens=None,
    validation_frequency=None,
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
    protocol_path = Path(protocol_path).resolve()
    protocol = validate_protocol(json.loads(protocol_path.read_text()))
    if experiment is not None:
        from exp.gpt2_candidate import validate_experiment

        experiment = validate_experiment(experiment)
        if experiment["method"] in ("tcd1", "fold_rms") and world != 8:
            raise ValueError("TCD fold collection requires exactly eight workers")
    validation = validation_configuration(
        protocol, validation_start, validation_tokens, validation_frequency, smoke_steps
    )
    data_manifest = json.loads((data_dir / "dataset_manifest.json").read_text())
    validate_dataset_files(data_dir, data_manifest)
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
    from exp.gpt2_evaluate import validate_slice

    validate_slice(
        data_dir / "fineweb_val_000000.bin",
        validation["token_offset"],
        validation["token_count"],
    )
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
        from exp.gpt2_candidate import experiment_source

        effective = experiment_source(effective, experiment)
    patch_validation = (
        protocol_path != PROTOCOL.resolve()
        or validation_start is not None
        or validation_tokens is not None
        or validation_frequency is not None
    )
    if patch_validation:
        effective = validation_source(
            effective,
            validation["token_offset"],
            validation["token_count"],
            validation["frequency"],
        )
    if smoke_steps:
        effective = smoke_source(effective, smoke_steps)
    metadata = {
        "protocol": protocol,
        "protocol_sha256": sha256(protocol_path),
        "source": source_manifest,
        "wrapper_sha256": sha256(__file__),
        "dataset_manifest_sha256": sha256(data_dir / "dataset_manifest.json"),
        "dataset_revision": data_manifest["revision"],
        "effective_source_sha256": hashlib.sha256(effective.encode()).hexdigest(),
        "seed": seed,
        "world_size": world,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "smoke_steps": smoke_steps,
        "comparison_eligible": smoke_steps is None,
        "experiment": experiment,
        "validation": validation,
        "experiment_source_sha256": {
            name: sha256(HERE / name) for name in ("gpt2_candidate.py", "tcd.py")
        }
        if experiment is not None
        else None,
        "compile_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
        "compile_threads_per_rank": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        "cache_condition": os.environ.get("BENCHMARK_CACHE_CONDITION", "unspecified"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "time_scope": "runner entry including torch import/setup/compile/data/train/eval/logging; excludes Python startup, wrapper import, shared download and Slurm launch; allocation cost must also be reported",
    }
    metadata.update(
        research_identity(
            protocol,
            experiment,
            source_manifest,
            metadata["dataset_manifest_sha256"],
            data_manifest["revision"],
        )
    )
    metadata["shared_identity"]["runtime"] = {
        key: metadata[key] for key in ("torch", "cuda", "gpu", "world_size")
    }
    metadata["shared_identity"]["precision"] = (
        "pinned upstream: BF16 embedding parameters/gradients and activations; "
        "FP32 other parameters/gradients and loss; compiled forward without autocast"
    )
    metadata["experiment_source_sha256"] = metadata["method_source_sha256"]
    if smoke_steps is not None or (
        experiment is not None and experiment["steps"] != protocol["training"]["steps"]
    ):
        metadata["comparison_eligible"] = False
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
                group="gpt2-tcd1-v1" if experiment else "gpt2-optimizer-reference-v1",
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
        if smoke_steps or experiment is not None or patch_validation:
            execution_path = output / f"effective_source_rank{rank}.py"
            execution_path.write_text(effective)
        sys.argv = [str(execution_path)]
        with contextlib.redirect_stdout(logger if rank == 0 else sys.stdout):
            if smoke_steps or experiment is not None or patch_validation:
                namespace = {"__name__": "__main__", "__file__": str(execution_path)}
                exec(compile(effective, str(execution_path), "exec"), namespace)  # noqa: S102 -- source hash verified above
            else:
                namespace = runpy.run_path(str(source_path), run_name="__main__")
        diagnostics = None
        if experiment is not None and experiment["method"] == "shampoo_adaptive":
            # Upstream has already destroyed the process group. Read owner state
            # locally; this introduces no collectives or training-loop logging.
            atomic_json(
                output / f"optimizer-diagnostics-rank{rank}.json",
                adaptive_optimizer_diagnostics(namespace["model"], namespace["optimizer2"], rank),
            )
            diagnostics = {
                "rank_files": [f"optimizer-diagnostics-rank{owner}.json" for owner in range(world)],
                "scope": "post-training rank-local state; inspect all rank files",
            }
        if rank == 0:
            if not logger.history:
                raise ValueError("Run produced no validation observations")
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
            checkpoint = None
            if save_checkpoint:
                from exp.gpt2_evaluate import export_checkpoint

                checkpoint = export_checkpoint(
                    output / "checkpoint.pt",
                    namespace["model"],
                    metadata,
                    final["training_tokens"],
                )
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
                "checkpoint": checkpoint,
                "optimizer_diagnostics": diagnostics,
                **{
                    key: metadata[key]
                    for key in (
                        "method",
                        "shared_identity",
                        "method_config_sha256",
                        "method_source_sha256",
                    )
                },
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
    run.add_argument("--protocol", type=Path, default=PROTOCOL)
    run.add_argument("--save-checkpoint", action="store_true")
    run.add_argument("--validation-start", type=int)
    run.add_argument("--validation-tokens", type=int)
    run.add_argument("--validation-frequency", type=int)
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
            protocol_path=args.protocol,
            save_checkpoint=args.save_checkpoint,
            validation_start=args.validation_start,
            validation_tokens=args.validation_tokens,
            validation_frequency=args.validation_frequency,
        )
