"""Controlled two-group LR profile, separate from the original optimizer search.

Two independent one-GPU workers share one two-GPU Slurm allocation. All
configurations are fixed before training; this diagnostic does not select or
evaluate on the confirmation set. A time-limited allocation can continue in a
new allocation without changing configurations or restarting completed runs.
"""

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from exp.common import DEFAULT, atomic_json

# Same twelve (matrix, embedding/norm) multipliers for both methods. The core
# is a 5 x 2 factorial grid, plus a lower matrix bracket and the SOTA BV point.
# BR/BU/BV are starting hypotheses from 1B models, not 150M results.
POINTS = (
    (1.0, 1.0), (1.25, 1.0), (1.25, 1.25), (1.0, 1.25),
    (0.825, 1.0), (1.5, 1.5), (1.5, 1.0), (2.0, 1.0),
    (0.5, 1.0), (0.825, 1.25), (1.5, 1.25), (2.0, 1.25),
)
METHODS = ("normuon", "hg_normuon")
TERMINAL = ("completed", "diverged")


def configurations():
    # Start with the anchor, the compensation hypothesis, and fixed shrinkage.
    # Ordering affects time-to-information, never selection or training budget.
    for i, (matrix_factor, embed_factor) in enumerate(POINTS):
        for method in METHODS:
            yield DEFAULT | {
                "method": method,
                "matrix_lr": DEFAULT["matrix_lr"] * matrix_factor,
                "embed_lr": DEFAULT["embed_lr"] * embed_factor,
                "meta_lr": 0.01 if method == "hg_normuon" else 0.0,
                "phase": "lr_diagnostic",
                "trial": i,
                "run_id": f"lr-{method}-m{i:02d}-s42",
                "wandb_group": "exp150m-lr-diagnostic-v1",
            }


def initialize(root, data_dir, cache_dir):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "study.json").exists():
        raise ValueError("Study exists; use submit/report")
    ids = []
    for cfg in configurations():
        rid = cfg["run_id"]
        ids.append(rid)
        atomic_json(root / "configs" / f"{rid}.json", cfg)
    state = {
        "version": 1,
        "purpose": "Conditional LR profile; not a new-optimizer success test",
        "code_dir": str(Path(__file__).resolve().parents[2]),
        "python": sys.executable,
        "data_dir": str(Path(data_dir).resolve()),
        "cache_dir": str(Path(cache_dir).resolve()),
        "runs": ids,
        "jobs": [],
        "max_allocations": 3,
        "lr_points": POINTS,
        "reference": "note/SOTA.md, note/BR.md, note/BU.md, note/BV.md",
        "reference_data_verified": "prepare-66ea6ae/data/bestfit (same as BU)",
        "seed": 42,
        "limits": [
            "A bounded two-group LR grid, with HG meta LR fixed at 0.01.",
            "1B-model SOTA scores cannot be compared to these 150M-model scores.",
            "One seed; no uncertainty estimate or optimizer superiority claim.",
            "A boundary minimum requires a subsequent preregistered extension.",
            "0.825 is an endpoint median, not an exact full-trajectory LR match.",
        ],
    }
    atomic_json(root / "study.json", state)
    return state


def read_result(root, rid):
    path = root / "runs" / rid / "result.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "missing"}


def report(root):
    state = json.loads((root / "study.json").read_text())
    rows = []
    for rid in state["runs"]:
        cfg = json.loads((root / "configs" / f"{rid}.json").read_text())
        result = read_result(root, rid)
        if result["status"] == "completed":
            if result["config"] != cfg:
                raise ValueError(f"Result/config mismatch: {rid}")
            if result["actual_tokens"] != result["total_tokens"]:
                raise ValueError(f"Incomplete token budget: {rid}")
        rows.append({
            "run_id": rid, "method": cfg["method"],
            "matrix_lr": cfg["matrix_lr"],
            "lr_factor": cfg["matrix_lr"] / DEFAULT["matrix_lr"],
            "embed_lr": cfg["embed_lr"],
            "embed_lr_factor": cfg["embed_lr"] / DEFAULT["embed_lr"],
            "status": result["status"],
            "dev_nll": result.get("metrics", {}).get("eval_loss"),
            "dev_ppl": result.get("metrics", {}).get("eval_perplexity"),
            "actual_tokens": result.get("actual_tokens"),
            "wandb_url": result.get("wandb_url"),
        })
    complete = [row for row in rows if row["status"] == "completed"]
    if len({row["actual_tokens"] for row in complete}) > 1:
        raise ValueError("Token budgets differ across runs")
    best = {}
    for method in METHODS:
        eligible = [r for r in complete if r["method"] == method]
        if eligible:
            winner = min(eligible, key=lambda row: row["dev_nll"])
            cfg = json.loads((root / "configs" / f"{winner['run_id']}.json").read_text())
            m, e = POINTS[cfg["trial"]]
            ms = [x for x, y in POINTS if y == e]
            es = [y for x, y in POINTS if x == m]
            best[method] = winner | {
                "matrix_lr_boundary": m in (min(ms), max(ms)),
                "embed_lr_boundary": e in (min(es), max(es)),
                "completed_grid_points": len(eligible),
            }
    result = {
        "complete": all(row["status"] in TERMINAL for row in rows),
        "best_observed": best, "rows": rows, "limitations": state["limits"],
        "claim": "Descriptive conditional LR comparison; no superiority conclusion",
    }
    if len(best) == 2:
        result["best_observed_hg_relative_ppl_percent"] = 100 * math.expm1(
            best["hg_normuon"]["dev_nll"] - best["normuon"]["dev_nll"]
        )
    atomic_json(root / "lr_report.json", result)
    with (root / "lr_profile.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def submit(root):
    # Serialize manual submissions and automatic continuations.
    with (root / "submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((root / "study.json").read_text())
        if report(root)["complete"]:
            return None
        live = subprocess.check_output(
            ["squeue", "--noheader", "--user", os.environ["USER"], "--format=%A"],
            text=True,
        ).splitlines()
        current = os.environ.get("SLURM_JOB_ID")
        if any(job in live and job != current for job in state["jobs"]):
            raise RuntimeError("A diagnostic allocation is already queued/running")
        if len(state["jobs"]) >= state["max_allocations"]:
            raise RuntimeError("Allocation cap reached; inspect unfinished trials")
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        command = [
            "sbatch", "--parsable", "--account=acd115198", "--partition=dev",
            "--gpus-per-node=2", "--cpus-per-task=24", "--mem=240G",
            "--time=04:00:00", "--job-name=exp150m-lr-profile",
            "--exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157,25a-hgpn146,25a-hgpn150,25a-hgpn071,25a-hgpn069,25a-hgpn173",
            f"--chdir={state['code_dir']}", f"--output={logs}/allocation-%j.out",
        ]
        if current:
            command += [f"--dependency=afterany:{current}"]
        command += [
            str(Path(state["code_dir"]) / "src/exp/lr_diagnostic.sbatch"), str(root)
        ]
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        state["jobs"].append(job)
        atomic_json(root / "study.json", state)
        print(json.dumps({"submitted_job": job, "root": str(root)}), flush=True)
        return job


def worker(root, tiny=False):
    state = json.loads((root / "study.json").read_text())
    # Do not let an accidental second allocation race on the same checkpoints.
    with (root / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        devices = ["cpu"] if tiny else os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if not tiny and len(devices) != 2:
            raise RuntimeError(f"Expected two allocated GPUs; got {devices}")
        ids = state["runs"]
        queues = [[] for _ in devices]
        for pair_start in range(0, len(ids), 2):
            # Alternate methods between physical GPUs across paired settings.
            for offset, rid in enumerate(ids[pair_start:pair_start + 2]):
                queues[(offset + pair_start // 2) % len(devices)].append(rid)
        stop = threading.Event()
        start = time.monotonic()
        errors = []

        def lane(device, queue):
            for rid in queue:
                existing = read_result(root, rid)
                if existing["status"] in TERMINAL:
                    continue
                if existing["status"] == "error":
                    stop.set()
                    errors.append(f"Software failure requires diagnosis: {rid}")
                    return
                # Reserve 45 minutes to finish a run and exit before Slurm's 4h.
                if stop.is_set() or time.monotonic() - start > (230 - 45) * 60:
                    return
                output = root / "runs" / rid
                command = [
                    state["python"], "-m", "exp.train", "--config",
                    str(root / "configs" / f"{rid}.json"), "--output", str(output),
                    "--data-dir", state["data_dir"], "--cache-dir", state["cache_dir"],
                    "--resume", "auto",
                ]
                if tiny:
                    command += ["--tiny-smoke", "--no-wandb"]
                env = os.environ.copy()
                if not tiny:
                    env["CUDA_VISIBLE_DEVICES"] = device
                logs = root / "logs"
                logs.mkdir(exist_ok=True)
                print(json.dumps({"run": rid, "device": device}), flush=True)
                trial_start = time.monotonic()
                with (logs / f"{rid}.log").open("a") as stream:
                    process = subprocess.run(command, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, check=False)
                result = read_result(root, rid)
                result["worker_gpu_hours"] = (time.monotonic() - trial_start) / 3600
                atomic_json(output / "result.json", result)
                if result["status"] not in TERMINAL:
                    errors.append(f"{rid}: exit={process.returncode}, {result['status']}")
                    stop.set()
                    return

        def guarded_lane(device, queue):
            try:
                lane(device, queue)
            except Exception:
                stop.set()
                raise

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(guarded_lane, device, queue)
                       for device, queue in zip(devices, queues, strict=True)]
            for future in futures:
                future.result()
        atomic_json(root / "allocations" / f"{os.environ.get('SLURM_JOB_ID', 'cpu')}.json",
                    {"elapsed_seconds": time.monotonic() - start,
                     "allocated_gpu_hours": 0 if tiny else
                     2 * (time.monotonic() - start) / 3600, "errors": errors})
        if errors:
            atomic_json(root / "needs_attention.json", errors)
            raise RuntimeError("; ".join(errors))
        if not tiny and not report(root)["complete"]:
            submit(root)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("init", "submit", "worker", "report"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-dir")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "init":
        if not args.data_dir or not args.cache_dir:
            parser.error("init requires --data-dir and --cache-dir")
        initialize(root, args.data_dir, args.cache_dir)
    elif args.command == "submit":
        submit(root)
    elif args.command == "worker":
        worker(root)
    else:
        result = report(root)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
