"""Summarize every trial, including failures; never select on confirmation loss."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from exp.common import atomic_json
from exp.optimizer import METHODS


def paired_summary(candidate, baseline):
    if len(candidate) != 5 or len(baseline) != 5:
        raise ValueError("The preregistered comparison requires all five paired seeds")
    delta = np.asarray(candidate) - np.asarray(baseline)
    mean = float(delta.mean())
    margin = 2.7764451051977987 * float(delta.std(ddof=1)) / math.sqrt(5)
    improvement = 1 - math.exp(mean)
    return {
        "delta_nll": delta.tolist(),
        "mean_delta_nll": mean,
        "ci95": [mean - margin, mean + margin],
        "geometric_mean_relative_ppl_improvement": improvement,
        "seeds_improved": int((delta < 0).sum()),
        "passes_primary_criterion": bool(
            mean + margin < 0 and improvement >= 0.003 and (delta < 0).sum() >= 4
        ),
    }


def report(root, upload=True):
    root = Path(root)
    state = json.loads((root / "study.json").read_text())
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "runs").glob("*/result.json")):
        result = json.loads(path.read_text())
        cfg = result.get("config", {})
        row = {
            "run_id": path.parent.name,
            "method": cfg.get("method"),
            "phase": cfg.get("phase"),
            "trial": cfg.get("trial"),
            "seed": cfg.get("seed"),
            "status": result["status"],
            "dev_nll": result.get("metrics", {}).get("eval_loss"),
            "confirm_nll": result.get("metrics", {}).get("confirm_loss"),
            "actual_tokens": result.get("actual_tokens"),
            "gpu_hours": result.get("worker_gpu_hours", result.get("gpu_hours", 0.0)),
            "peak_memory_bytes": result.get("peak_memory_bytes"),
            "median_step_seconds": result.get("median_step_seconds"),
            "wandb_url": result.get("wandb_url"),
            "error": result.get("error"),
        }
        for attempt in path.parent.glob("attempt-*.json"):
            previous = json.loads(attempt.read_text())
            row["gpu_hours"] += previous.get(
                "worker_gpu_hours", previous.get("gpu_hours", 0.0)
            )
        rows.append(row)
    curves = []
    for method in METHODS:
        best, hours = None, 0.0
        for number, row in enumerate(
            sorted(
                [r for r in rows if r["method"] == method and r["phase"] == "search"],
                key=lambda r: r["trial"],
            ),
            1,
        ):
            hours += row["gpu_hours"]
            if row["status"] == "completed":
                best = row["dev_nll"] if best is None else min(best, row["dev_nll"])
            curves.append(
                {
                    "method": method,
                    "trials": number,
                    "gpu_hours": hours,
                    "best_dev_ppl": math.exp(best) if best is not None else None,
                }
            )
    summary = {
        "phase": state["phase"],
        "search_not_converged": state.get("search_not_converged"),
        "total_worker_gpu_hours": sum(row["gpu_hours"] for row in rows),
        "total_recorded_training_tokens": sum(
            row["actual_tokens"] or 0 for row in rows
        ),
        "rows": rows,
        "tuning_curves": curves,
        "confirmation": None,
        "limitations": [
            "Evaluation documents were used by historical 1B experiments.",
            "Five training seeds give a small-sample t interval, not a general guarantee.",
            "Equal trial/token budgets; GPU-hour curves separately expose compute costs.",
            "Worker GPU-hours exclude Slurm launch/controller overhead; use sacct for allocation billing.",
        ],
    }
    confirmation = {
        method: sorted(
            [r for r in rows if r["method"] == method and r["phase"] == "confirm"],
            key=lambda r: r["seed"],
        )
        for method in METHODS
    }
    if state["phase"] == "complete":
        complete = all(
            len(v) == 5
            and [r["seed"] for r in v] == list(range(101, 106))
            and all(
                r["status"] == "completed" and r["confirm_nll"] is not None for r in v
            )
            for v in confirmation.values()
        )
        if complete:
            primary = state["primary_baseline"]
            candidate = [r["confirm_nll"] for r in confirmation["hg_normuon"]]
            summary["confirmation"] = {
                "primary_baseline": primary,
                "primary": paired_summary(
                    candidate, [r["confirm_nll"] for r in confirmation[primary]]
                ),
                "secondary_descriptive": {
                    m: paired_summary(
                        candidate, [r["confirm_nll"] for r in confirmation[m]]
                    )
                    for m in METHODS
                    if m != "hg_normuon"
                },
                "mean_ppl": {
                    m: float(np.mean([math.exp(r["confirm_nll"]) for r in v]))
                    for m, v in confirmation.items()
                },
            }
            if state.get("search_not_converged"):
                summary["confirmation"]["claim"] = (
                    "Search was not saturated; results are restricted to the tested budget."
                )
        else:
            summary["confirmation"] = {
                "claim": "No complete five-seed comparison; do not claim improvement."
            }
    atomic_json(root / "report.json", summary)
    if rows:
        with (root / "trials.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if upload:
        import os

        import wandb

        with wandb.init(
            project="lab-training-llama",
            entity=os.environ.get("WANDB_ENTITY", "cerulean-labs"),
            group="exp150m-hg-v1",
            job_type="report",
            name=root.name + "-report",
        ) as run:
            table = wandb.Table(
                columns=list(curves[0]) if curves else ["method"],
                data=[list(row.values()) for row in curves],
            )
            run.log(
                {
                    "tuning/all_trials": wandb.Table(
                        columns=list(rows[0]) if rows else ["run_id"],
                        data=[list(row.values()) for row in rows],
                    ),
                    "tuning/curves": table,
                }
            )
            if curves:
                run.log(
                    {
                        "tuning/best_ppl_by_trials": wandb.plot.line(
                            table, "trials", "best_dev_ppl", stroke="method"
                        ),
                        "tuning/best_ppl_by_gpu_hours": wandb.plot.line(
                            table, "gpu_hours", "best_dev_ppl", stroke="method"
                        ),
                    }
                )
            run.summary.update(
                {
                    "total_worker_gpu_hours": summary["total_worker_gpu_hours"],
                    "confirmation": summary["confirmation"],
                }
            )
            artifact = wandb.Artifact(root.name + "-results", type="optimizer-study")
            for filename in (
                "study.json",
                "report.json",
                "trials.csv",
                "locked_finalists.json",
            ):
                if (root / filename).exists():
                    artifact.add_file(str(root / filename))
            run.log_artifact(artifact)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    options = parser.parse_args()
    report(options.root, upload=not options.no_wandb)
