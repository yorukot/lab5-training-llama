"""Balanced Slurm search, selection, and locked confirmation campaign.

Initial search: 4 x 16 complete 300M-token runs. Expansion is symmetric (8 per
method, cap 32). Selection: top 3 x 2 additional seeds per method. Confirmation:
four selected configurations x five NEW paired seeds. No confirmation feedback
is used to select a configuration or extend a search.
"""

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from exp.common import DEFAULT, atomic_json
from exp.optimizer import METHODS

BOUNDS = {"matrix_lr": [1e-4, 1e-2], "embed_lr": [3e-4, 3e-2],
          "matrix_decay": [.05, 3.], "embed_decay": [.05, 3.], "meta_lr": [.001, .05]}
SEARCH_SEED = 20261016


def log_value(u, bounds):
    return math.exp(math.log(bounds[0]) * (1 - u) + math.log(bounds[1]) * u)


def configuration(method, trial, bounds=None):
    bounds = BOUNDS if bounds is None else bounds
    cfg = DEFAULT | {"method": method, "trial": trial}
    if trial == 0:
        if method == "adamw":
            cfg.update(matrix_lr=6e-4, embed_lr=6e-4)
    else:
        point = torch.quasirandom.SobolEngine(9, scramble=True, seed=SEARCH_SEED).draw(trial)[-1].tolist()
        for dim, key in ((0, "matrix_lr"), (1, "embed_lr"), (8, "meta_lr")):
            cfg[key] = log_value(point[dim], bounds[key])
        for dim, key in ((2, "matrix_decay"), (3, "embed_decay")):
            cfg[key] = 0. if point[dim] < .125 else log_value((point[dim] - .125) / .875, bounds[key])
        cfg.update(momentum=[.9, .95, .98][min(2, int(point[4] * 3))],
                   matrix_beta1=[.85, .9, .95][min(2, int(point[4] * 3))],
                   beta2=[.95, .9873, .999][min(2, int(point[5] * 3))],
                   warmup=[.01, .05][int(point[6] >= .5)],
                   decay=[.5, .8][int(point[7] >= .5)])
    if method != "adamw":
        cfg["matrix_beta1"] = .9  # unused for NorMuon matrices
    if not method.startswith("hg_"):
        cfg["meta_lr"] = 0.
    return cfg


def run_id(config):
    return f"{config['phase']}-{config['method']}-t{config['trial']:02d}-s{config['seed']}"


def write_config(root, config):
    config = config | {"run_id": run_id(config)}
    path = root / "configs" / f"{config['run_id']}.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f"Refusing to change an existing trial: {path}")
    atomic_json(path, config)
    return config["run_id"]


def initialize(root, data_dir, concurrency=4):
    if (root / "study.json").exists():
        raise ValueError("Study already exists; use advance/status")
    root.mkdir(parents=True, exist_ok=True)
    state = dict(version=1, phase="smoke", search_count=0, max_trials=32,
                 concurrency=concurrency, data_dir=str(Path(data_dir).resolve()),
                 python=sys.executable, code_dir=str(Path(__file__).resolve().parents[2]),
                 bounds={method: copy.deepcopy(BOUNDS) for method in METHODS},
                 batches=[], attempts={}, active_runs=[])
    for method in METHODS:
        state["active_runs"].append(write_config(root, configuration(method, 0) | {"phase": "smoke"}))
    atomic_json(root / "study.json", state)
    return state


def results(root, ids):
    return {rid: json.loads((root / "runs" / rid / "result.json").read_text())
            if (root / "runs" / rid / "result.json").exists() else {"status": "missing"} for rid in ids}


def submit(root, state, ids):
    # p_dev permits 20 submitted jobs per user, including array tasks. Leave
    # room for the current and next controller; keep the logical phase intact.
    ids = ids[:16]
    batch_number = len(state["batches"])
    batch = root / "batches" / f"{batch_number:03d}.json"
    atomic_json(batch, ids)
    script = Path(state["code_dir"]) / "exp" / "run.sbatch"
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    base = ["sbatch", "--parsable", "--account=acd115198", "--partition=dev",
            "--gpus-per-node=1", "--cpus-per-task=12", "--mem=120G", "--time=04:00:00",
            "--exclude=25a-hgpn001,25a-hgpn003,25a-hgpn062,25a-hgpn083,25a-hgpn157,25a-hgpn146,25a-hgpn150,25a-hgpn071,25a-hgpn069,25a-hgpn173",
            f"--chdir={Path(state['code_dir']).parent}", f"--output={logs}/%x-%A_%a.out"]
    job = subprocess.check_output(base + ["--job-name=exp150m-" + state["phase"],
                                          f"--array=0-{len(ids)-1}%{state['concurrency']}",
                                          str(script), "worker", str(root), str(batch)], text=True).strip().split(";")[0]
    # A short dependent controller executes even if a worker is killed or a node
    # fails, so missing status files cannot silently stall the campaign forever.
    record = dict(job_id=job, runs=ids, phase=state["phase"], submitted_at=time.time())
    state["batches"].append(record)
    atomic_json(root / "study.json", state)
    controller = subprocess.check_output(base + ["--job-name=exp150m-next", "--cpus-per-task=1",
                                                  "--mem=4G", "--time=00:10:00", f"--dependency=afterany:{job}",
                                                  str(script), "advance", str(root)], text=True).strip().split(";")[0]
    record["controller_job_id"] = controller
    atomic_json(root / "study.json", state)
    print(json.dumps(record, indent=2))


def maybe_expand(root, state):
    count = state["search_count"]
    need = False
    bests = {}
    for method in METHODS:
        rows = []
        for trial in range(count):
            rid = run_id(dict(method=method, trial=trial, phase="search", seed=42))
            result = results(root, [rid])[rid]
            if result["status"] == "completed":
                rows.append((result["metrics"]["eval_loss"], trial, result["config"]))
        if len(rows) < 3:
            need = True
            continue
        rows.sort()
        bests[method] = [row[2] for row in rows[:3]]
        best_loss, _, cfg = rows[0]
        previous = [loss for loss, trial, _ in rows if trial < count - 8]
        if previous and 1 - math.exp(best_loss - min(previous)) >= .002:
            need = True
        for key, bounds in state["bounds"][method].items():
            value = cfg[key]
            if value == 0 or key == "meta_lr" and not method.startswith("hg_"):
                continue
            position = math.log(value / bounds[0]) / math.log(bounds[1] / bounds[0])
            if position <= .1:
                bounds[0] /= 2
                need = True
            if position >= .9:
                bounds[1] *= 2
                need = True
    # Shared base-parameter coordinates/ranges for the three NorMuon variants.
    family = [m for m in METHODS if m != "adamw"]
    for key in BOUNDS:
        if key != "meta_lr":
            shared = [min(state["bounds"][m][key][0] for m in family),
                      max(state["bounds"][m][key][1] for m in family)]
            for method in family:
                state["bounds"][method][key] = shared.copy()
    if need and count < state["max_trials"]:
        state["active_runs"] = [write_config(root, configuration(m, t, state["bounds"][m]))
                                for t in range(count, count + 8) for m in METHODS]
        state["search_count"] += 8
        return
    if len(bests) != len(METHODS):
        raise RuntimeError("Fewer than three successful configurations for a method; inspect failures")
    state["search_not_converged"] = need
    state["shortlist"] = bests
    state["phase"] = "selection"
    state["active_runs"] = [write_config(root, cfg | {"seed": seed, "phase": "selection"})
                            for method in METHODS for cfg in bests[method] for seed in (43, 44)]


def select_finalists(root, state):
    winners, scores = {}, {}
    for method in METHODS:
        candidates = []
        for cfg in state["shortlist"][method]:
            ids = [run_id(cfg | {"phase": "search", "seed": 42})] + [
                run_id(cfg | {"phase": "selection", "seed": seed}) for seed in (43, 44)]
            rows = results(root, ids).values()
            score = sum(r["metrics"]["eval_loss"] for r in rows) / 3 if all(r["status"] == "completed" for r in rows) else math.inf
            candidates.append((score, cfg["trial"], cfg))
        score, _, winner = min(candidates, key=lambda row: (row[0], row[1]))
        if not math.isfinite(score):
            raise RuntimeError(f"No stable finalist for {method}")
        winners[method], scores[method] = winner, score
    state.update(winners=winners, selection_scores=scores,
                 primary_baseline=min(("adamw", "normuon"), key=lambda m: scores[m]), phase="confirm")
    atomic_json(root / "locked_finalists.json", dict(winners=winners, scores=scores,
                                                     primary_baseline=state["primary_baseline"]))
    state["active_runs"] = [write_config(root, winners[m] | {"phase": "confirm", "seed": seed})
                            for seed in range(101, 106) for m in METHODS]


def advance(root, do_submit):
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((root / "study.json").read_text())
        current = results(root, state["active_runs"])
        if state["phase"] == "complete":
            print("Campaign complete")
            return
        if not state["batches"]:
            if do_submit:
                submit(root, state, state["active_runs"])
            return
        outstanding = [rid for rid, r in current.items() if r["status"] not in ("completed", "diverged")]
        if outstanding:
            # Only retry failures attributable to infrastructure. Software errors
            # stop the study instead of being counted as bad hyperparameters.
            jobs = state["batches"][-1]["job_id"]
            live = subprocess.check_output(["squeue", "--noheader", "-j", jobs], text=True).strip()
            if live:
                print("Workers still active; no duplicate submission")
                return
            retry = []
            submitted_ids = {rid for batch in state["batches"] for rid in batch["runs"]}
            for rid in outstanding:
                if rid not in submitted_ids:
                    retry.append(rid)
                    continue
                r = current[rid]
                count = state["attempts"].get(rid, 0)
                if r["status"] in ("missing", "running", "infrastructure_error") and count < 2:
                    state["attempts"][rid] = count + 1
                    path = root / "runs" / rid / "result.json"
                    if path.exists():
                        atomic_json(path.with_name(f"attempt-{count}.json"), r)
                    retry.append(rid)
                else:
                    atomic_json(root / "needs_attention.json", current)
                    raise RuntimeError(f"Manual diagnosis required for {rid}: {r.get('error', r['status'])}")
            if do_submit:
                submit(root, state, retry)
            return
        if state["phase"] == "smoke":
            if any(r["status"] != "completed" for r in current.values()):
                raise RuntimeError("A smoke run failed; search will not start")
            timings = {r["config"]["method"]: r["median_step_seconds"] for r in current.values()}
            atomic_json(root / "smoke_summary.json", {"timings": timings, "runs": current})
            if max(timings[m] / timings["normuon"] for m in METHODS if m.startswith("hg_")) > 1.25:
                raise RuntimeError("HG overhead exceeds 25%; inspect before the full search")
            state.update(phase="search", search_count=16)
            state["active_runs"] = [write_config(root, configuration(m, t)) for t in range(16) for m in METHODS]
        elif state["phase"] == "search":
            maybe_expand(root, state)
        elif state["phase"] == "selection":
            select_finalists(root, state)
        elif state["phase"] == "confirm":
            state["phase"] = "complete"
            atomic_json(root / "study.json", state)
            from exp.report import report
            report(root)
            return
        atomic_json(root / "study.json", state)
        if do_submit:
            submit(root, state, state["active_runs"])


def worker(root, batch, index):
    state = json.loads((root / "study.json").read_text())
    rid = json.loads(Path(batch).read_text())[index]
    cfg = json.loads((root / "configs" / f"{rid}.json").read_text())
    output = root / "runs" / rid
    existing = results(root, [rid])[rid]
    if existing["status"] in ("completed", "diverged"):
        return
    command = [state["python"], "-m", "exp.train", "--config", str(root / "configs" / f"{rid}.json"),
               "--output", str(output), "--data-dir", state["data_dir"],
               "--cache-dir", str(root / "assets"), "--resume", "auto"]
    if cfg["phase"] == "smoke":
        command += ["--stop-after", "50"]
    start = time.perf_counter()
    process = subprocess.run(command, check=False)
    result = results(root, [rid])[rid]
    if result["status"] in ("missing", "running"):
        result.update(status="infrastructure_error" if process.returncode < 0 else "error",
                      error=f"Training exited {process.returncode} without a terminal record", config=cfg)
    result["worker_gpu_hours"] = (time.perf_counter() - start) / 3600
    result["array_job_id"] = os.environ.get("SLURM_ARRAY_JOB_ID")
    result["array_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID")
    atomic_json(output / "result.json", result)
    if result["status"] not in ("completed", "diverged"):
        raise SystemExit(process.returncode or 1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--root", required=True)
    init.add_argument("--data-dir", required=True)
    init.add_argument("--concurrency", type=int, default=4)
    init.add_argument("--submit", action="store_true")
    adv = sub.add_parser("advance")
    adv.add_argument("--root", required=True)
    adv.add_argument("--submit", action="store_true")
    work = sub.add_parser("worker")
    work.add_argument("--root", required=True)
    work.add_argument("--batch", required=True)
    work.add_argument("--index", type=int, required=True)
    status = sub.add_parser("status")
    status.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == "init":
        initialize(root, args.data_dir, args.concurrency)
        advance(root, args.submit)
    elif args.command == "advance":
        advance(root, args.submit)
    elif args.command == "worker":
        worker(root, args.batch, args.index)
    else:
        state = json.loads((root / "study.json").read_text())
        current = results(root, state["active_runs"])
        print(json.dumps({"phase": state["phase"], "search_count": state["search_count"],
                          "runs": {rid: r["status"] for rid, r in current.items()}}, indent=2))


if __name__ == "__main__":
    main()
