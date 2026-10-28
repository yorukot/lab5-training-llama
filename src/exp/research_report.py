"""Audit fixed-token optimizer confirmation without selecting on held-out results.

Run with ``python -m exp.research_report --manifest confirmation.json``. A manifest
contains ``schema_version: 1``, ``locked``, ``baseline_method``, ``candidate_method``,
``seeds``, ``expected`` and ``records``. Each records entry is an object with paths
``run`` (runner result.json) and ``evaluation`` (standalone checkpoint evaluation).
Paths are resolved relative to the manifest, never the shell working directory.

``expected`` contains ``training_tokens``, ``shared_identity``, ``evaluation`` and
``method_config_sha256`` (a mapping from each method to its frozen config hash).
Shared identity includes architecture, tokenizer, data_sha256 (dataset manifest),
and training_config_sha256. Evaluation identity includes slice_id, token_offset
(zero-based target stream), token_count, and data_sha256 (validation shard).
``method_source_sha256`` pins each method's implementation file/hash mapping and
is required for locked confirmation (optional for exploratory comparisons).
Locked ``evaluation_settings`` fixes ``device`` and ``batch_sequences``; the
actual evaluator hash must match the shared, frozen gpt2_evaluate.py source hash.
``evaluation_runtime`` pins the evaluator's actual torch/CUDA/GPU/precision
identity independently of the training runtime stored in the checkpoint.

The reporter verifies recorded provenance, not the honesty or creation time of a
lock declaration. Archive the manifest before running confirmation. Historical
references are displayed separately and never substitute for paired fresh seeds.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

CONFIRMATION_SEEDS = (101, 202, 303, 404, 505)
TRAINING_TOKENS = 3250 * 524288
HOLDOUT_OFFSET = 80_000_000
HOLDOUT_TOKENS = 10_485_760
TARGET_PPL_REDUCTION = 0.5
T_CRITICAL_DF4 = 2.7764451051977987
EVALUATION_KEYS = ("slice_id", "token_offset", "token_count", "data_sha256")
DIAGNOSTIC_KEYS = (
    "end_to_end_seconds",
    "end_to_end_gpu_hours",
    "allocation_gpu_hours",
    "peak_allocated_bytes_rank0",
    "peak_memory_bytes",
    "optimizer_seconds",
    "median_step_seconds",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"Invalid {label}")
    return value


def _hash(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        f"Invalid SHA-256 for {label}",
    )
    return value


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float), f"Invalid {label}: expected a number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"Invalid {label}: number is too large") from None
    _require(
        math.isfinite(number) and number >= 0,
        f"Invalid {label}: expected a finite nonnegative number",
    )
    return number


def _ppl(value: Any, label: str) -> float:
    nll = _finite(value, label)
    try:
        result = math.exp(nll)
    except OverflowError:
        raise ValueError(f"{label} causes PPL overflow") from None
    _require(math.isfinite(result), f"{label} causes PPL overflow")
    return result


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(),
        object_pairs_hook=_json_pairs,
        parse_constant=_invalid_constant,
    )
    _require(isinstance(value, dict), f"Expected a JSON object in {path}")
    return value


def protocol_digest(manifest: dict[str, Any]) -> str:
    """Identify the locked comparison independently of subsequently added paths."""
    protocol = {
        key: manifest[key]
        for key in (
            "schema_version",
            "locked",
            "baseline_method",
            "candidate_method",
            "seeds",
            "expected",
        )
    }
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode()
    ).hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> None:
    _require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1, "Expected schema_version 1")
    _require(type(manifest.get("locked")) is bool, "locked must be an explicit boolean")
    baseline, candidate = manifest.get("baseline_method"), manifest.get("candidate_method")
    _require(
        isinstance(baseline, str)
        and bool(baseline.strip())
        and isinstance(candidate, str)
        and bool(candidate.strip())
        and baseline != candidate,
        "Specify distinct baseline_method and candidate_method",
    )
    seeds = manifest.get("seeds")
    _require(isinstance(seeds, list) and bool(seeds), "Specify a nonempty seed list")
    for seed in seeds:
        _integer(seed, "seed")
    _require(len(seeds) == len(set(seeds)), "Duplicate manifest seeds")
    if manifest["locked"]:
        _require(seeds == list(CONFIRMATION_SEEDS), "Locked confirmation requires the five registered seeds")
    expected = manifest.get("expected")
    _require(isinstance(expected, dict), "Missing expected protocol identity")
    tokens = _integer(expected.get("training_tokens"), "training token budget", 1)
    if manifest["locked"]:
        _require(tokens == TRAINING_TOKENS, "Locked confirmation requires the full fixed-token budget")
    shared = expected.get("shared_identity")
    _require(isinstance(shared, dict), "Missing shared_identity")
    _require(bool(shared.get("architecture")), "Missing architecture identity")
    _require(isinstance(shared.get("tokenizer"), str) and bool(shared["tokenizer"]), "Missing tokenizer identity")
    for name in ("data_sha256", "training_config_sha256"):
        _hash(shared.get(name), f"shared_identity.{name}")
    if manifest["locked"]:
        for name in ("dataset_revision", "data_order"):
            _require(isinstance(shared.get(name), str) and bool(shared[name]), f"Missing shared_identity.{name}")
    evaluation = expected.get("evaluation")
    _require(isinstance(evaluation, dict), "Missing evaluation identity")
    _require(set(evaluation) == set(EVALUATION_KEYS), "Evaluation identity must contain exactly slice_id, token_offset, token_count, data_sha256")
    offset = _integer(evaluation.get("token_offset"), "evaluation token_offset")
    count = _integer(evaluation.get("token_count"), "evaluation token_count", 1)
    _require(evaluation.get("slice_id") == f"start-{offset}-count-{count}", "Evaluation slice_id does not match its target interval")
    _hash(evaluation.get("data_sha256"), "evaluation.data_sha256")
    if manifest["locked"]:
        _require((offset, count) == (HOLDOUT_OFFSET, HOLDOUT_TOKENS), "Locked confirmation requires the reserved holdout target interval")
    for name in ("method_config_sha256", "method_source_sha256"):
        if name == "method_source_sha256" and name not in expected and not manifest["locked"]:
            continue
        mapping = expected.get(name)
        _require(isinstance(mapping, dict) and set(mapping) == {baseline, candidate}, f"{name} must pin exactly the baseline and candidate")
        for method, value in mapping.items():
            if name == "method_config_sha256":
                _hash(value, f"{name}.{method}")
            else:
                _require(isinstance(value, dict) and bool(value), f"{name}.{method} must pin implementation files")
                for filename, digest in value.items():
                    _require(isinstance(filename, str) and bool(filename), "Invalid implementation filename")
                    _hash(digest, f"{name}.{method}.{filename}")
    if manifest["locked"]:
        source_maps = expected["method_source_sha256"]
        evaluator_hashes = [source_maps[method].get("gpt2_evaluate.py") for method in (baseline, candidate)]
        for digest in evaluator_hashes:
            _hash(digest, "frozen evaluator")
        _require(evaluator_hashes[0] == evaluator_hashes[1], "Methods must use the same frozen evaluator")
    if manifest["locked"] or "evaluation_settings" in expected:
        settings = expected.get("evaluation_settings")
        _require(isinstance(settings, dict) and set(settings) == {"device", "batch_sequences"}, "evaluation_settings must pin device and batch_sequences")
        _require(isinstance(settings["device"], str) and bool(settings["device"]), "Invalid evaluation device")
        _integer(settings["batch_sequences"], "evaluation batch_sequences", 1)
    if manifest["locked"] or "evaluation_runtime" in expected:
        runtime = expected.get("evaluation_runtime")
        _require(isinstance(runtime, dict) and set(runtime) == {"torch", "cuda", "gpu", "precision"}, "evaluation_runtime must pin torch, cuda, gpu, and precision")
        for key in ("torch", "precision"):
            _require(isinstance(runtime[key], str) and bool(runtime[key]), f"Invalid evaluation runtime {key}")
        for key in ("cuda", "gpu"):
            _require(runtime[key] is None or (isinstance(runtime[key], str) and bool(runtime[key])), f"Invalid evaluation runtime {key}")
    _require(isinstance(manifest.get("records"), list), "records must be a list")
    # Reject non-finite nested metadata, even for callers bypassing read_json.
    json.dumps(manifest, allow_nan=False)


def paired_summary(baseline_nll: list[float], candidate_nll: list[float]) -> dict[str, Any]:
    """Paired t interval on per-seed absolute PPL differences, with five seeds."""
    _require(len(baseline_nll) == len(candidate_nll) == 5, "All five paired seeds are required")
    baseline = [_ppl(value, "baseline NLL") for value in baseline_nll]
    candidate = [_ppl(value, "candidate NLL") for value in candidate_nll]
    delta = [left - right for left, right in zip(baseline, candidate, strict=True)]
    mean = statistics.mean(delta)
    sample_sd = statistics.stdev(delta)
    margin = T_CRITICAL_DF4 * sample_sd / math.sqrt(5)
    lower, upper = mean - margin, mean + margin
    _require(all(math.isfinite(value) for value in (mean, sample_sd, lower, upper)), "PPL differences overflow the confidence interval")
    return {
        "n_pairs": 5,
        "baseline_mean_ppl": statistics.mean(baseline),
        "candidate_mean_ppl": statistics.mean(candidate),
        "paired_ppl_reductions": delta,
        "mean_ppl_reduction": mean,
        "sample_sd_ppl_reduction": sample_sd,
        "ci95_ppl_reduction": [lower, upper],
        "interval": "two-sided Student t, df=4, training seed as replication unit",
        "seeds_improved": sum(value > 0 for value in delta),
        "meets_point_target": mean >= TARGET_PPL_REDUCTION,
        "interval_excludes_zero_in_improvement_direction": lower > 0,
        "passes_primary_criterion": mean >= TARGET_PPL_REDUCTION and lower > 0,
        "supports_true_reduction_at_least_target": lower > TARGET_PPL_REDUCTION,
    }


def _record_paths(entry: Any, base_dir: Path) -> tuple[Path, Path]:
    _require(isinstance(entry, dict) and set(entry) == {"run", "evaluation"}, "Each record must contain run and evaluation paths")
    for key in ("run", "evaluation"):
        _require(isinstance(entry[key], str) and bool(entry[key]), f"Invalid {key} path")
    return base_dir / entry["run"], base_dir / entry["evaluation"]


def _validate_record(
    run: dict[str, Any], evaluation: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    expected = manifest["expected"]
    method, seed = evaluation.get("method"), evaluation.get("seed")
    _require(method in (manifest["baseline_method"], manifest["candidate_method"]), "Unexpected method")
    _integer(seed, "record seed")
    _require(seed in manifest["seeds"], "Unexpected seed")
    _require(run.get("status") == evaluation.get("status") == "completed", "Training and standalone evaluation must both be completed")
    metadata = run.get("metadata", {})
    _require(isinstance(metadata, dict), "Missing run metadata")
    _require(run.get("method") == method, "Training/evaluation method mismatch")
    _require(metadata.get("seed") == seed, "Training/evaluation seed mismatch")
    _require(metadata.get("smoke_steps") is None, "Smoke runs cannot enter a research comparison")
    _require(type(metadata.get("comparison_eligible")) is bool, "Missing comparison eligibility declaration")
    if manifest["locked"]:
        _require(metadata["comparison_eligible"], "Ineligible training run for locked confirmation")
        _require(evaluation.get("confirmation_protocol_sha256") == protocol_digest(manifest), "Evaluation was not authorized by this frozen confirmation protocol")
    protocol_hash = _hash(metadata.get("protocol_sha256"), "run protocol")
    _require(evaluation.get("protocol_sha256") == protocol_hash, "Training/evaluation protocol hash mismatch")
    for artifact, label in ((run, "run"), (evaluation, "evaluation")):
        _require(artifact.get("shared_identity") == expected["shared_identity"], f"{label} shared identity mismatch (architecture/data/tokenizer/training config)")
        for name in ("method_config_sha256", "method_source_sha256"):
            if name in expected:
                _require(artifact.get(name) == expected[name][method], f"{label} {name} mismatch")
    if "evaluation_settings" in expected:
        _require({key: evaluation.get(key) for key in expected["evaluation_settings"]} == expected["evaluation_settings"], "Standalone evaluation settings mismatch")
    if "evaluation_runtime" in expected:
        _require(evaluation.get("evaluation_runtime") == expected["evaluation_runtime"], "Actual evaluation runtime mismatch")
    if "method_source_sha256" in expected:
        frozen_evaluator = expected["method_source_sha256"][method].get("gpt2_evaluate.py")
        if frozen_evaluator is not None:
            _require(evaluation.get("evaluator_sha256") == frozen_evaluator, "Actual evaluator hash differs from the frozen source")
    actual_tokens = run.get("final", {}).get("training_tokens")
    _integer(actual_tokens, "actual training tokens", 1)
    _integer(evaluation.get("training_tokens"), "evaluation training tokens", 1)
    _require(actual_tokens == evaluation["training_tokens"] == expected["training_tokens"], "Training token budget mismatch")
    checkpoint_hash = _hash(run.get("checkpoint", {}).get("sha256"), "run checkpoint")
    _require(evaluation.get("checkpoint_sha256") == checkpoint_hash, "Evaluation checkpoint mismatch")
    source = metadata.get("source", {})
    source_hash = source.get("files", {}).get("train_gpt_simple.py", {}).get("sha256")
    _hash(source_hash, "upstream source")
    _require(evaluation.get("source_sha256") == source_hash, "Training/evaluation source hash mismatch")
    _require(bool(source.get("commit")) and evaluation.get("source_commit") == source["commit"], "Training/evaluation source commit mismatch")
    _require(metadata.get("dataset_manifest_sha256") == evaluation.get("dataset_manifest_sha256") == expected["shared_identity"]["data_sha256"], "Dataset manifest hash mismatch")
    _require(bool(metadata.get("dataset_revision")) and evaluation.get("dataset_revision") == metadata["dataset_revision"], "Dataset revision mismatch")
    metrics = evaluation.get("evaluation")
    _require(isinstance(metrics, dict), "Missing evaluation metrics")
    _require({key: metrics.get(key) for key in EVALUATION_KEYS} == expected["evaluation"], "Evaluation slice or data hash mismatch")
    _require(metrics.get("aggregation") == "token_weighted_mean", "NLL must be a token-weighted mean")
    nll = _finite(metrics.get("nll"), "evaluation NLL")
    ppl = _ppl(nll, "evaluation NLL")
    if "ppl" in metrics:
        reported = _finite(metrics["ppl"], "reported PPL")
        _require(math.isclose(reported, ppl, rel_tol=1e-9, abs_tol=1e-12), "Reported PPL does not equal exp(NLL)")
    return {
        "method": method,
        "seed": seed,
        "nll": nll,
        "ppl": ppl,
        "training_tokens": actual_tokens,
        "checkpoint_sha256": checkpoint_hash,
    }


def summarize(manifest: dict[str, Any], base_dir: Path | str = ".") -> dict[str, Any]:
    """Return an auditable report; invalid/missing records cannot confirm a result."""
    validate_manifest(manifest)
    base_dir = Path(base_dir)
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    seen: set[tuple[str, int]] = set()
    errors = []
    diagnostics = []
    for index, entry in enumerate(manifest["records"]):
        run = evaluation = None
        try:
            run_path, evaluation_path = _record_paths(entry, base_dir)
            run = read_json(run_path)
            evaluation = read_json(evaluation_path)
            method, seed = evaluation.get("method"), evaluation.get("seed")
            _require(isinstance(method, str) and type(seed) is int, "Missing method/seed identity")
            key = (method, seed)
            _require(key not in seen, f"Duplicate method/seed record: {method}/{seed}")
            seen.add(key)
            rows[key] = _validate_record(run, evaluation, manifest)
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
            errors.append({"record_index": index, "record": entry, "error": str(error)})
        finally:
            if run is not None:
                metadata = run.get("metadata")
                metadata = metadata if isinstance(metadata, dict) else {}
                cost = {key: run.get(key) for key in DIAGNOSTIC_KEYS if key in run}
                final = run.get("final")
                if isinstance(final, dict):
                    cost.update({key: final[key] for key in ("training_seconds", "training_gpu_hours") if key in final})
                cost.update({
                    "record_index": index,
                    "method": run.get("method"),
                    "seed": metadata.get("seed"),
                    "training_status": run.get("status"),
                    "evaluation_status": evaluation.get("status") if evaluation else None,
                    "time_scope": metadata.get("time_scope"),
                })
                if evaluation and "end_to_end_seconds" in evaluation:
                    cost["standalone_evaluation_seconds"] = evaluation["end_to_end_seconds"]
                diagnostics.append(cost)
    baseline, candidate = manifest["baseline_method"], manifest["candidate_method"]
    missing = [
        {"method": method, "seed": seed}
        for seed in manifest["seeds"]
        for method in (baseline, candidate)
        if (method, seed) not in rows
    ]
    pairs = []
    for seed in manifest["seeds"]:
        left, right = rows.get((baseline, seed)), rows.get((candidate, seed))
        pairs.append({
            "seed": seed,
            "baseline": left,
            "candidate": right,
            "ppl_reduction": left["ppl"] - right["ppl"] if left and right else None,
        })
    complete = not errors and not missing
    confirmation = None
    status = "incomplete" if not complete else "exploratory"
    if complete and manifest["locked"]:
        confirmation = paired_summary(
            [rows[(baseline, seed)]["nll"] for seed in manifest["seeds"]],
            [rows[(candidate, seed)]["nll"] for seed in manifest["seeds"]],
        )
        status = "confirmed" if confirmation["passes_primary_criterion"] else "inconclusive"
    claims = {
        "incomplete": "Incomplete or invalid evidence; no confirmation claim is supported.",
        "exploratory": "Exploratory comparison; the protocol was not locked for confirmation.",
        "inconclusive": "Complete locked comparison did not meet the registered success criterion.",
        "confirmed": "Positive paired improvement with a point estimate meeting the 0.5 absolute PPL target.",
    }
    return {
        "schema_version": 1,
        "status": status,
        "claim": claims[status],
        "protocol_sha256": protocol_digest(manifest),
        "locked": manifest["locked"],
        "baseline_method": baseline,
        "candidate_method": candidate,
        "expected": manifest["expected"],
        "primary_target_absolute_ppl_reduction": TARGET_PPL_REDUCTION,
        "pairs": pairs,
        "confirmation": confirmation,
        "missing_records": missing,
        "errors": errors,
        "run_diagnostics": diagnostics,
        "historical_reference": manifest.get("historical_reference"),
        "limitations": [
            "The t interval assumes independent training seed pairs and approximately normal paired PPL differences; five pairs give limited precision.",
            "The interval conditions on the selected hyperparameters and does not include uncertainty from optimizer selection.",
            "The reserved target interval is from the same validation corpus, not an independent corpus.",
            "Recorded provenance equality does not prove lock creation time or independently reproduce training; archive the lock before observing holdout results.",
            "Runtime, memory, and historical reference values do not enter the fixed-token success criterion.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fixed-token optimizer research report",
        "",
        f"**Status: {report['status']}**. {report['claim']}",
        "",
        f"Baseline: `{report['baseline_method']}`. Candidate: `{report['candidate_method']}`.",
        "",
        "| Seed | Baseline NLL | Baseline PPL | Candidate NLL | Candidate PPL | PPL reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in report["pairs"]:
        values = [str(pair["seed"])]
        for side in ("baseline", "candidate"):
            values.extend(f"{pair[side][key]:.8f}" if pair[side] else "missing" for key in ("nll", "ppl"))
        values.append(f"{pair['ppl_reduction']:.8f}" if pair["ppl_reduction"] is not None else "missing")
        lines.append("| " + " | ".join(values) + " |")
    if summary := report["confirmation"]:
        lower, upper = summary["ci95_ppl_reduction"]
        lines += [
            "",
            f"Mean paired PPL reduction: **{summary['mean_ppl_reduction']:.8f}**; 95% paired t interval: [{lower:.8f}, {upper:.8f}].",
            "",
            "A claim that the underlying reduction is at least 0.5 is "
            + ("supported by this interval." if summary["supports_true_reduction_at_least_target"] else "not supported by this interval."),
        ]
    if report["errors"]:
        lines += ["", "Evidence errors:", ""]
        lines.extend(f"- Record {item['record_index']}: {item['error']}" for item in report["errors"])
    if report["historical_reference"] is not None:
        lines += ["", "Historical reference (excluded from confirmation):", "", "```json", json.dumps(report["historical_reference"], indent=2, allow_nan=False), "```"]
    if report["run_diagnostics"]:
        lines += [
            "", "Per-run diagnostics (excluded from the success criterion):", "",
            "| Method | Seed | Training status | Runner GPU-hours | Evaluation seconds | Peak memory bytes |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
        for row in report["run_diagnostics"]:
            values = [
                row.get("method"), row.get("seed"), row.get("training_status"),
                row.get("end_to_end_gpu_hours"), row.get("standalone_evaluation_seconds"),
                row.get("peak_memory_bytes", row.get("peak_allocated_bytes_rank0")),
            ]
            lines.append("| " + " | ".join("unknown" if value is None else str(value) for value in values) + " |")
        lines += ["", "Runner GPU-hours and rank-0 memory retain their recorded scope; allocation billing and additional diagnostics are preserved separately in JSON."]
    lines += ["", "Limitations:", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="Write the machine-readable report")
    parser.add_argument("--markdown", type=Path, help="Write the human-readable report")
    parser.add_argument("--stdout-format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args(argv)
    try:
        report = summarize(read_json(args.manifest), args.manifest.resolve().parent)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown = render_markdown(report)
    if args.json:
        args.json.write_text(encoded)
    if args.markdown:
        args.markdown.write_text(markdown)
    print(encoded if args.stdout_format == "json" else markdown, end="")
    return 2 if report["status"] == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
