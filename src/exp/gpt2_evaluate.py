"""Evaluate an exported pinned GPT-2 checkpoint on an explicit target slice.

Offsets index the next-token target stream: offset zero uses shard token 0 as
input and shard token 1 as target. No trainer setup or training code is executed.
"""

import argparse
import ast
import json
import math
import os
import time
from pathlib import Path

from exp.common import atomic_json
from exp.gpt2_benchmark import PROTOCOL, VENDOR, sha256, shard_metadata, verify_source

CHECKPOINT_VERSION = 1
MODEL_CLASSES = (
    "RMSNorm",
    "Linear",
    "Rotary",
    "CausalSelfAttention",
    "MLP",
    "Block",
    "GPT",
)


def model_classes():
    """Load only audited class definitions, never upstream imports or setup."""
    import torch

    verify_source()
    source = (VENDOR / "train_gpt_simple.py").read_text()
    definitions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name in MODEL_CLASSES
    ]
    if tuple(node.name for node in definitions) != MODEL_CLASSES:
        raise ValueError("Pinned model class structure changed")
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "Tensor": torch.Tensor,
    }
    exec(  # noqa: S102 -- hash-verified class definitions only, no trainer setup
        compile(
            ast.Module(body=definitions, type_ignores=[]), "pinned_gpt2_model", "exec"
        ),
        namespace,
    )
    return namespace


def build_model(spec):
    import torch

    expected = json.loads(PROTOCOL.read_text())["model"]
    if spec != expected:
        raise ValueError("Checkpoint model does not match the pinned architecture")
    # Meta allocation avoids constructing a second 650 MB copy before loading.
    with torch.device("meta"):
        model = model_classes()["GPT"](
            spec["vocab_size"], spec["layers"], spec["width"]
        )
    if sum(p.numel() for p in model.parameters()) != spec["parameters"]:
        raise ValueError("Checkpoint model parameter count mismatch")
    return model


def validate_slice(path, token_offset, token_count, sequence_length=1024):
    if (
        type(token_offset) is not int
        or token_offset < 0
        or type(token_count) is not int
        or token_count <= 0
        or type(sequence_length) is not int
        or sequence_length <= 0
        or token_count % sequence_length
    ):
        raise ValueError("Invalid validation slice or sequence length")
    info = shard_metadata(path, checksum=False)
    if token_offset + token_count + 1 > info["tokens"]:
        raise ValueError("Validation slice exceeds shard bounds")
    return info


def slice_tensors(path, token_offset, token_count, sequence_length=1024):
    """Return CPU input/target views; exact-end slices are valid."""
    import torch

    info = validate_slice(path, token_offset, token_count, sequence_length)
    # Map uint16 tokens after the 1024-byte header without reading the full shard.
    mapped = torch.from_file(
        str(path), shared=False, size=512 + info["tokens"], dtype=torch.uint16
    )
    begin = 512 + token_offset
    values = mapped[begin : begin + token_count + 1]
    return (
        values[:-1].view(-1, sequence_length),
        values[1:].view(-1, sequence_length),
    )


def load_validation_batch(path, token_offset, token_count):
    """Training-only integration: preserve upstream rank and sequence layout."""
    import torch.distributed as dist

    world, rank = dist.get_world_size(), dist.get_rank()
    if token_count % (world * 64 * 1024):
        raise ValueError(
            "Validation tokens must fill 64-sequence microbatches on every rank"
        )
    validate_slice(path, token_offset, token_count)
    local_count = token_count // world
    inputs, targets = slice_tensors(
        path, token_offset + rank * local_count, local_count
    )
    import torch

    return inputs.to("cuda", dtype=torch.int32), targets.to("cuda", dtype=torch.int64)


def export_checkpoint(path, model, metadata, training_tokens):
    """Export evaluation weights only; this is deliberately not a resume file."""
    import torch

    path = Path(path)
    if path.exists():
        raise ValueError("Checkpoint already exists; use a fresh output directory")
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model": metadata["protocol"]["model"],
        "source_commit": metadata["source"]["commit"],
        "source_sha256": metadata["source"]["files"]["train_gpt_simple.py"]["sha256"],
        "seed": metadata["seed"],
        "training_tokens": training_tokens,
        "experiment": metadata["experiment"],
        "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
        "dataset_revision": metadata["dataset_revision"],
        "protocol_sha256": metadata["protocol_sha256"],
        "method": metadata["method"],
        "shared_identity": metadata["shared_identity"],
        "method_config_sha256": metadata["method_config_sha256"],
        "method_source_sha256": metadata["method_source_sha256"],
    }
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "filename": path.name,
        "sha256": sha256(path),
        "format_version": CHECKPOINT_VERSION,
    }


def load_checkpoint(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != CHECKPOINT_VERSION
    ):
        raise ValueError("Unsupported checkpoint format")
    manifest = verify_source()
    if (
        payload["source_commit"] != manifest["commit"]
        or payload["source_sha256"]
        != manifest["files"]["train_gpt_simple.py"]["sha256"]
    ):
        raise ValueError("Checkpoint source identity mismatch")
    model = build_model(payload["model"])
    expected = model.state_dict()
    state = payload["state_dict"]
    if state.keys() != expected.keys() or any(
        state[name].shape != expected[name].shape
        or state[name].dtype != expected[name].dtype
        for name in expected
    ):
        raise ValueError(
            "Checkpoint tensor names, shapes or dtypes differ from pinned model"
        )
    model.load_state_dict(state, strict=True, assign=True)
    return model, payload


def evaluate_batches(model, inputs, targets, batch_sequences=8, device="cuda"):
    import torch

    if type(batch_sequences) is not int or batch_sequences <= 0:
        raise ValueError("batch_sequences must be positive")
    if inputs.shape != targets.shape or targets.ndim != 2 or targets.numel() == 0:
        raise ValueError("Expected equally shaped nonempty sequence tensors")
    model.eval()
    total_nll, count = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(targets), batch_sequences):
            x = inputs[start : start + batch_sequences].to(device, dtype=torch.int32)
            y = targets[start : start + batch_sequences].to(device, dtype=torch.int64)
            value = float(model(x, y).item())  # upstream forward returns SUM
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite evaluation loss")
            total_nll += value
            count += y.numel()
    nll = total_nll / count
    return {
        "nll": nll,
        "ppl": math.exp(nll),
        "token_count": count,
        "aggregation": "token_weighted_mean",
    }


def evaluation_runtime(device):
    import torch

    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(device))
        if torch.device(device).type == "cuda"
        else None,
        "precision": "eager pinned model dtypes; FP32 batch loss sum; float64 host aggregation",
    }


def confirmation_gate(
    payload,
    token_offset,
    token_count,
    data_sha256,
    batch_sequences,
    device,
    runtime,
    manifest_path=None,
):
    """Bind any reserved-target read to the previously frozen comparison."""
    from exp.research_report import (
        HOLDOUT_OFFSET,
        HOLDOUT_TOKENS,
        protocol_digest,
        read_json,
        validate_manifest,
    )

    overlaps = max(token_offset, HOLDOUT_OFFSET) < min(
        token_offset + token_count, HOLDOUT_OFFSET + HOLDOUT_TOKENS
    )
    if manifest_path is None:
        if overlaps:
            raise ValueError(
                "Reserved targets require --confirmation-manifest before evaluation"
            )
        return None
    manifest = read_json(Path(manifest_path))
    validate_manifest(manifest)
    if not manifest["locked"]:
        raise ValueError("Confirmation manifest must be locked")
    expected, method = manifest["expected"], payload["method"]
    if method not in (manifest["baseline_method"], manifest["candidate_method"]):
        raise ValueError("Checkpoint method is absent from the frozen comparison")
    if payload["seed"] not in manifest["seeds"]:
        raise ValueError("Checkpoint seed is absent from the frozen comparison")
    for field in ("training_tokens", "shared_identity"):
        if payload[field] != expected[field]:
            raise ValueError(f"Checkpoint {field} differs from the frozen comparison")
    for field in ("method_config_sha256", "method_source_sha256"):
        if payload[field] != expected[field][method]:
            raise ValueError(f"Checkpoint {field} differs from the frozen comparison")
    actual_slice = {
        "slice_id": f"start-{token_offset}-count-{token_count}",
        "token_offset": token_offset,
        "token_count": token_count,
        "data_sha256": data_sha256,
    }
    if actual_slice != expected["evaluation"]:
        raise ValueError("Requested slice differs from the frozen confirmation slice")
    if {"device": device, "batch_sequences": batch_sequences} != expected[
        "evaluation_settings"
    ]:
        raise ValueError("Evaluation settings differ from the frozen comparison")
    if runtime != expected["evaluation_runtime"]:
        raise ValueError("Evaluation runtime differs from the frozen comparison")
    if sha256(__file__) != expected["method_source_sha256"][method]["gpt2_evaluate.py"]:
        raise ValueError("Evaluator source differs from the frozen comparison")
    return protocol_digest(manifest)


def evaluate(
    checkpoint,
    data_dir,
    output,
    token_offset,
    token_count,
    batch_sequences=8,
    device="cuda",
    confirmation_manifest=None,
):
    checkpoint, data_dir, output = Path(checkpoint), Path(data_dir), Path(output)
    if output.exists():
        raise ValueError("Evaluation output exists; use a fresh output path")
    started = time.perf_counter()
    atomic_json(output, {"status": "running"})
    try:
        from exp.research_report import HOLDOUT_OFFSET, HOLDOUT_TOKENS

        if confirmation_manifest is None and max(token_offset, HOLDOUT_OFFSET) < min(
            token_offset + token_count, HOLDOUT_OFFSET + HOLDOUT_TOKENS
        ):
            raise ValueError(
                "Reserved targets require --confirmation-manifest before evaluation"
            )
        model, payload = load_checkpoint(checkpoint)
        manifest_path = data_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            sha256(manifest_path) != payload["dataset_manifest_sha256"]
            or manifest["revision"] != payload["dataset_revision"]
        ):
            raise ValueError("Evaluation dataset identity differs from training")
        name = "fineweb_val_000000.bin"
        records = [record for record in manifest["files"] if record["filename"] == name]
        if len(records) != 1 or shard_metadata(data_dir / name) != records[0]:
            raise ValueError("Evaluation shard failed integrity verification")
        runtime = evaluation_runtime(device)
        confirmation_digest = confirmation_gate(
            payload,
            token_offset,
            token_count,
            records[0]["sha256"],
            batch_sequences,
            device,
            runtime,
            confirmation_manifest,
        )
        inputs, targets = slice_tensors(data_dir / name, token_offset, token_count)
        metrics = evaluate_batches(
            model.to(device), inputs, targets, batch_sequences, device
        )
        result = {
            key: payload[key]
            for key in (
                "seed",
                "training_tokens",
                "experiment",
                "method",
                "shared_identity",
                "method_config_sha256",
                "method_source_sha256",
                "protocol_sha256",
                "source_commit",
                "source_sha256",
                "dataset_manifest_sha256",
                "dataset_revision",
                "model",
            )
        }
        result.update(
            {
                "status": "completed",
                "checkpoint_sha256": sha256(checkpoint),
                "evaluator_sha256": sha256(__file__),
                "evaluation": metrics
                | {
                    "slice_id": f"start-{token_offset}-count-{token_count}",
                    "token_offset": token_offset,
                    "input_start": token_offset,
                    "first_target_shard_index": token_offset + 1,
                    "data_sha256": records[0]["sha256"],
                },
                "device": device,
                "batch_sequences": batch_sequences,
                "evaluation_runtime": runtime,
                "confirmation_protocol_sha256": confirmation_digest,
                "end_to_end_seconds": time.perf_counter() - started,
            }
        )
        atomic_json(output, result)
        return result
    except BaseException as error:
        atomic_json(output, {"status": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--batch-sequences", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confirmation-manifest", type=Path)
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.data_dir,
        args.output,
        args.start,
        args.tokens,
        args.batch_sequences,
        args.device,
        args.confirmation_manifest,
    )
