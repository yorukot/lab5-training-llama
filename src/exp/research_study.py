"""Lock-protected allocation accounting for the fixed-token optimizer study.

Reserve the *Slurm wall limit*, including environment setup, before submitting.
Attach the resulting job ID immediately and reconcile from terminal ``sacct``
allocation records, not training timers. All failures consume their actual GPU
allocation time. Reservations and reconciliations are never inferred from PPL.

``--study`` is a JSON ledger file. Immutable, content-addressed JSON snapshots
live beside it in ``<ledger filename>.snapshots``. The lock is a separate inode
so atomic ledger replacement cannot invalidate concurrent-process locking.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


STAGE_GPU_HOURS = {
    "infrastructure": 4,
    "screening": 28,
    "development": 8,
    "confirmation": 20,
    "recovery": 4,
}
TOTAL_GPU_HOURS = 64
TERMINAL_STATES = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
     "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
     "SPECIAL_EXIT"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum,
             f"{name} must be an integer >= {minimum}")
    return value


def _seconds(value: Any) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             "elapsed_seconds must be finite and nonnegative")
    return float(value)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _parse_json(data: bytes) -> dict[str, Any]:
    result = json.loads(data, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    _require(isinstance(result, dict), "Expected a JSON object")
    return result


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, ledger: dict[str, Any]) -> None:
    data = (json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def _locked(study: Path, *, initialize: bool = False) -> Iterator[Path]:
    study = Path(study).expanduser().resolve()
    if initialize:
        study.parent.mkdir(parents=True, exist_ok=True)
    _require(study.parent.is_dir(), f"Study directory does not exist: {study.parent}")
    with study.with_name(study.name + ".lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield study
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _snapshot_dir(study: Path) -> Path:
    return study.with_name(study.name + ".snapshots")


def _snapshot(study: Path, source: Path) -> dict[str, str]:
    source = Path(source).expanduser().resolve()
    data = source.read_bytes()
    _parse_json(data)
    sha = _digest(data)
    directory = _snapshot_dir(study)
    directory.mkdir(exist_ok=True)
    snapshot = directory / f"{sha}.json"
    if snapshot.exists():
        _require(snapshot.read_bytes() == data, f"Snapshot integrity failure: {snapshot}")
    else:
        # Publish only a fully persisted snapshot. The study lock serializes writers.
        fd, temporary = tempfile.mkstemp(prefix=".snapshot.", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
            os.replace(temporary, snapshot)
            _sync_directory(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {"source_path": str(source), "snapshot_path": str(snapshot), "sha256": sha}


def _verify_snapshot(study: Path, path: Any, sha: Any) -> dict[str, Any]:
    _require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None,
             "Invalid snapshot SHA-256")
    expected = _snapshot_dir(study) / f"{sha}.json"
    _require(path == str(expected), "Snapshot path does not match its content address")
    data = expected.read_bytes()
    _require(_digest(data) == sha, f"Snapshot integrity failure: {expected}")
    return _parse_json(data)


def _protocol_limits(protocol: dict[str, Any]) -> dict[str, Any]:
    budget = protocol.get("budget")
    _require(isinstance(budget, dict), "Protocol must declare budget")
    total = budget.get("total_gpu_hours")
    stages = budget.get("stages")
    _require(type(total) in (int, float) and total == TOTAL_GPU_HOURS,
             "Protocol must declare the authorized 64 GPU-hour total")
    _require(isinstance(stages, dict) and set(stages) == set(STAGE_GPU_HOURS),
             "Protocol budget stages do not match the registered research plan")
    for stage, expected in STAGE_GPU_HOURS.items():
        _require(type(stages[stage]) in (int, float) and stages[stage] == expected,
                 f"Protocol cap for {stage} must be {expected} GPU-hours")
    return {"total_gpu_hours": total, "stages": copy.deepcopy(stages)}


def _validate(study: Path, ledger: dict[str, Any]) -> None:
    _require(ledger.get("schema_version") == 1, "Unsupported ledger schema")
    _integer(ledger.get("revision"), "revision")
    protocol = ledger.get("protocol")
    _require(isinstance(protocol, dict), "Missing protocol snapshot")
    protocol_data = _verify_snapshot(study, protocol.get("snapshot_path"), protocol.get("sha256"))
    _require(ledger.get("limits") == _protocol_limits(protocol_data),
             "Ledger budget limits differ from the immutable protocol")
    runs = ledger.get("runs")
    _require(isinstance(runs, dict), "Invalid run registry")
    jobs: set[str] = set()
    for run_id, row in runs.items():
        _require(isinstance(row, dict) and row.get("run_id") == run_id, "Run identity mismatch")
        _require(row.get("stage") in STAGE_GPU_HOURS, "Invalid run stage")
        _integer(row.get("gpus"), "gpus", 1)
        _integer(row.get("seed"), "seed")
        _integer(row.get("reserved_seconds"), "reserved_seconds", 1)
        _require(row.get("reserved_gpu_seconds") == row["gpus"] * row["reserved_seconds"],
                 "Inconsistent reserved usage")
        _require(row.get("protocol_sha256") == protocol["sha256"]
                 and row.get("protocol_snapshot_path") == protocol["snapshot_path"],
                 "Run protocol differs from study protocol")
        _verify_snapshot(study, row.get("config_snapshot_path"), row.get("config_sha256"))
        status = row.get("status")
        _require(status in {"reserved", "submitted", "reconciled", "released"}, "Invalid run status")
        job_id = row.get("job_id")
        if status in {"submitted", "reconciled"}:
            _require(isinstance(job_id, str) and re.fullmatch(r"\d+(?:_\d+)?", job_id) is not None,
                     "Submitted run requires a valid Slurm job ID")
            _require(job_id not in jobs, "Slurm job ID attached to multiple runs")
            jobs.add(job_id)
        else:
            _require(job_id is None, "Unsubmitted run cannot have a job ID")
        if status == "reconciled":
            _require(row.get("scheduler_state") in TERMINAL_STATES, "Nonterminal reconciliation")
            elapsed = _seconds(row.get("elapsed_seconds"))
            allocated_gpus = _integer(row.get("allocated_gpus"), "allocated_gpus")
            _require(row.get("actual_gpu_seconds") == allocated_gpus * elapsed
                     and math.isfinite(row["actual_gpu_seconds"]),
                     "Inconsistent actual usage")
            _require(type(row.get("wall_limit_exceeded")) is bool
                     and row["wall_limit_exceeded"] == (elapsed > row["reserved_seconds"]),
                     "Inconsistent wall limit overrun flag")
            _require(type(row.get("allocation_mismatch")) is bool
                     and row["allocation_mismatch"] == (allocated_gpus != row["gpus"]),
                     "Inconsistent GPU allocation mismatch flag")
        else:
            _require(all(row.get(name) is None for name in
                         ("scheduler_state", "elapsed_seconds", "actual_gpu_seconds")),
                     "Unreconciled run cannot carry actual usage")


def _read(study: Path) -> dict[str, Any]:
    _require(study.is_file(), f"Study is not initialized: {study}")
    ledger = _parse_json(study.read_bytes())
    _validate(study, ledger)
    return ledger


def _usage(ledger: dict[str, Any]) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage, cap in ledger["limits"]["stages"].items():
        rows = [row for row in ledger["runs"].values() if row["stage"] == stage]
        charged = sum(row["actual_gpu_seconds"] for row in rows if row["status"] == "reconciled")
        reserved = sum(row["reserved_gpu_seconds"] for row in rows
                       if row["status"] in {"reserved", "submitted"})
        stages[stage] = {
            "limit_gpu_hours": cap, "charged_gpu_hours": charged / 3600,
            "reserved_gpu_hours": reserved / 3600,
            "committed_gpu_hours": (charged + reserved) / 3600,
            "remaining_gpu_hours": cap - (charged + reserved) / 3600,
        }
    charged = sum(row["charged_gpu_hours"] for row in stages.values())
    reserved = sum(row["reserved_gpu_hours"] for row in stages.values())
    limit = ledger["limits"]["total_gpu_hours"]
    violations = [f"{name} stage cap exceeded" for name, row in stages.items()
                  if row["committed_gpu_hours"] > row["limit_gpu_hours"]]
    if charged + reserved > limit:
        violations.append("total GPU-hour cap exceeded")
    for run_id, row in ledger["runs"].items():
        if row.get("allocation_mismatch"):
            violations.append(f"{run_id}: allocated GPU count differs from reservation")
    return {
        "limit_gpu_hours": limit, "charged_gpu_hours": charged,
        "reserved_gpu_hours": reserved, "committed_gpu_hours": charged + reserved,
        "remaining_gpu_hours": limit - charged - reserved,
        "stages": stages, "violations": violations,
    }


def _response(study: Path, ledger: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    result = {"study_path": str(study), "usage": _usage(ledger)}
    if run_id is None:
        result["ledger"] = ledger
    else:
        result["record"] = ledger["runs"][run_id]
    return copy.deepcopy(result)


def _save(study: Path, ledger: dict[str, Any]) -> None:
    ledger["revision"] += 1
    ledger["updated_at"] = _now()
    _validate(study, ledger)
    _atomic_write(study, ledger)


def initialize_study(study: Path, protocol: Path) -> dict[str, Any]:
    with _locked(study, initialize=True) as path:
        _require(not path.exists(), f"Study already exists: {path}")
        # Validate before persisting any protocol or ledger state.
        limits = _protocol_limits(_parse_json(Path(protocol).read_bytes()))
        snapshot = _snapshot(path, protocol)
        limits = _protocol_limits(_verify_snapshot(path, snapshot["snapshot_path"], snapshot["sha256"]))
        now = _now()
        ledger = {"schema_version": 1, "revision": 0, "created_at": now,
                  "updated_at": now, "protocol": snapshot, "limits": limits, "runs": {}}
        _validate(path, ledger)
        _atomic_write(path, ledger)
        return _response(path, ledger)


def reserve_run(study: Path, run_id: str, stage: str, gpus: int, seconds: int,
                config: Path, seed: int) -> dict[str, Any]:
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) is not None,
             "run_id must be 1-128 letters, digits, underscores, dots or hyphens")
    _require(stage in STAGE_GPU_HOURS, "Unknown research stage")
    _integer(gpus, "gpus", 1)
    _integer(seconds, "seconds", 1)
    _integer(seed, "seed")
    with _locked(study) as path:
        ledger = _read(path)
        _require(run_id not in ledger["runs"], f"Run ID already exists: {run_id}")
        usage = _usage(ledger)
        _require(not usage["violations"], "Budget cap already exceeded; new reservations are blocked")
        cost = gpus * seconds / 3600
        _require(cost <= usage["remaining_gpu_hours"], "Reservation exceeds total GPU-hour budget")
        _require(cost <= usage["stages"][stage]["remaining_gpu_hours"],
                 f"Reservation exceeds {stage} stage GPU-hour budget")
        snapshot = _snapshot(path, config)
        ledger["runs"][run_id] = {
            "run_id": run_id, "stage": stage, "gpus": gpus,
            "reserved_seconds": seconds, "reserved_gpu_seconds": gpus * seconds,
            "seed": seed, "status": "reserved", "job_id": None,
            "config_source_path": snapshot["source_path"],
            "config_snapshot_path": snapshot["snapshot_path"], "config_sha256": snapshot["sha256"],
            "protocol_snapshot_path": ledger["protocol"]["snapshot_path"],
            "protocol_sha256": ledger["protocol"]["sha256"], "created_at": _now(),
            "scheduler_state": None, "elapsed_seconds": None, "actual_gpu_seconds": None,
        }
        _save(path, ledger)
        return _response(path, ledger, run_id)


def _run(ledger: dict[str, Any], run_id: str) -> dict[str, Any]:
    _require(run_id in ledger["runs"], f"Unknown run ID: {run_id}")
    return ledger["runs"][run_id]


def attach_run(study: Path, run_id: str, job_id: str) -> dict[str, Any]:
    _require(isinstance(job_id, str) and re.fullmatch(r"\d+(?:_\d+)?", job_id) is not None,
             "job_id must be a Slurm numeric job ID (or numeric array task ID)")
    with _locked(study) as path:
        ledger = _read(path)
        row = _run(ledger, run_id)
        if row["status"] in {"submitted", "reconciled"} and row["job_id"] == job_id:
            return _response(path, ledger, run_id)
        _require(row["status"] == "reserved", "Only an unsubmitted reservation can attach a job")
        _require(all(other["job_id"] != job_id for other in ledger["runs"].values()),
                 "Slurm job ID is already attached to a run")
        row.update(status="submitted", job_id=job_id, submitted_at=_now())
        _save(path, ledger)
        return _response(path, ledger, run_id)


def reconcile_run(study: Path, run_id: str, state: str, elapsed_seconds: float,
                  allocated_gpus: int | None = None) -> dict[str, Any]:
    _require(isinstance(state, str) and state in TERMINAL_STATES,
             "Reconciliation requires an exact terminal Slurm state")
    elapsed = _seconds(elapsed_seconds)
    with _locked(study) as path:
        ledger = _read(path)
        row = _run(ledger, run_id)
        actual_gpus = row["gpus"] if allocated_gpus is None else allocated_gpus
        _integer(actual_gpus, "allocated_gpus")
        _require(math.isfinite(actual_gpus * elapsed), "Actual allocation GPU-seconds must be finite")
        if row["status"] == "reconciled":
            _require(row["scheduler_state"] == state and row["elapsed_seconds"] == elapsed
                     and row["allocated_gpus"] == actual_gpus,
                     "Conflicting reconciliation; an allocation can only be charged once")
            return _response(path, ledger, run_id)
        _require(row["status"] == "submitted", "Reconciliation requires an attached Slurm job")
        # Persist actual usage even when it exceeds any previously approved cap.
        row.update(status="reconciled", scheduler_state=state, elapsed_seconds=elapsed,
                   allocated_gpus=actual_gpus, actual_gpu_seconds=actual_gpus * elapsed,
                   allocation_mismatch=actual_gpus != row["gpus"],
                   wall_limit_exceeded=elapsed > row["reserved_seconds"], reconciled_at=_now())
        _save(path, ledger)
        return _response(path, ledger, run_id)


def release_run(study: Path, run_id: str) -> dict[str, Any]:
    with _locked(study) as path:
        ledger = _read(path)
        row = _run(ledger, run_id)
        if row["status"] == "released":
            return _response(path, ledger, run_id)
        _require(row["status"] == "reserved" and row["job_id"] is None,
                 "Only an unsubmitted reservation can be released; attached jobs must be reconciled")
        row.update(status="released", released_at=_now())
        _save(path, ledger)
        return _response(path, ledger, run_id)


def show_study(study: Path) -> dict[str, Any]:
    with _locked(study) as path:
        return _response(path, _read(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "reserve", "attach", "reconcile", "release", "show"):
        command = commands.add_parser(name)
        command.add_argument("--study", type=Path, required=True)
        if name == "init":
            command.add_argument("--protocol", type=Path, required=True)
        if name in {"reserve", "attach", "reconcile", "release"}:
            command.add_argument("--run-id", required=True)
        if name == "reserve":
            command.add_argument("--stage", choices=STAGE_GPU_HOURS, required=True)
            command.add_argument("--gpus", type=int, required=True)
            command.add_argument("--seconds", type=int, required=True)
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--seed", type=int, required=True)
        if name == "attach":
            command.add_argument("--job-id", required=True)
        if name == "reconcile":
            command.add_argument("--state", choices=sorted(TERMINAL_STATES), required=True)
            command.add_argument("--elapsed-seconds", type=float, required=True)
            command.add_argument("--allocated-gpus", type=int,
                                 help="Actual GPU count from sacct AllocTRES; defaults to reserved count")
    args = vars(parser.parse_args(argv))
    name = args.pop("command")
    functions = {"init": initialize_study, "reserve": reserve_run, "attach": attach_run,
                 "reconcile": reconcile_run, "release": release_run, "show": show_study}
    try:
        result = functions[name](**args)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        overrun = result.get("record", {}).get("wall_limit_exceeded", False)
        return 2 if result["usage"]["violations"] or (name == "reconcile" and overrun) else 0
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
