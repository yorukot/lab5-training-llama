"""Small paired CPU falsification study, never a language-model benchmark.

Run: PYTHONPATH=src python -m exp.research_probe --output /tmp/fresh-probe
The complete protocol is written before any optimization trial. Every method
uses the same generated initialization and gradient-noise sequence per seed.
An equal bounded LR grid is a diagnostic, not evidence of fully tuned methods.
"""

import argparse
import hashlib
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from exp.common import atomic_json, fingerprint
from exp.research_optimizers import make_matrix_optimizer, validate_research_config

METHODS = ("sgd", "matrix_adamw", "shampoo", "shampoo_shrink", "soft_polar")
CASES = (
    "aligned_noise",
    "rotated_noise",
    "weak_singular_signal",
    "abrupt_curvature_rotation",
)
LRS = (0.003, 0.01, 0.03, 0.1, 0.3)
TUNING_SEEDS = (11, 22)
TEST_SEEDS = (101, 202, 303)
DIMENSION = 4
NOISE_RMS = 0.25
FAILURE_LOSS = 1e12
CASE_DESCRIPTIONS = {
    "aligned_noise": {
        "curvature": "randomly rotated eigenvalues logspace(1, 30), dimension 16",
        "noise": "Gaussian covariance proportional to the Hessian, entry RMS .25",
        "prediction": "Reliable anisotropy may help structured preconditioning.",
    },
    "rotated_noise": {
        "curvature": "randomly rotated eigenvalues logspace(1, 30), dimension 16",
        "noise": "same covariance eigenvalues as aligned case, independently rotated",
        "prediction": "Gradient-noise geometry may disagree with loss curvature.",
    },
    "weak_singular_signal": {
        "curvature": "diagonal entries [1,.1,.01,.001]; other coordinate curvature 1",
        "initial": "diag(1,3,10,30), with independently sampled diagonal signs",
        "noise": "isotropic Gaussian, entry RMS .003",
        "prediction": "Small gradient singular modes still carry useful excess loss.",
    },
    "abrupt_curvature_rotation": {
        "curvature": "same spectrum, new independent basis halfway through training",
        "noise": "isotropic Gaussian, entry RMS .25",
        "prediction": "Stored momentum and geometry can lag after a curvature change.",
    },
}


@dataclass
class QuadraticProblem:
    initial: torch.Tensor
    noise: torch.Tensor
    first_hessian: torch.Tensor
    final_hessian: torch.Tensor
    switch_step: int

    def hessian(self, step):
        return self.final_hessian if step >= self.switch_step else self.first_hessian


def population_loss_and_gradient(parameter, hessian):
    """Exact F(W)=.5 vec(W)^T H vec(W), grad=reshape(H vec(W))."""
    flat = parameter.reshape(-1)
    gradient = hessian @ flat
    return 0.5 * torch.dot(flat, gradient), gradient.reshape_as(parameter)


def draw_problem(case, seed, steps):
    """Generate all samples up front, without optimizer-dependent RNG draws."""
    if case not in CASES:
        raise ValueError(f"Unknown case: {case}")
    if type(seed) is not int or seed < 0 or type(steps) is not int or steps < 1:
        raise ValueError("Expected a nonnegative seed and positive step count")
    # The first two cases additionally share their curvature and initial state,
    # isolating the effect of rotating noise. Python hash() is avoided.
    family_index = (
        0 if case in ("aligned_noise", "rotated_noise") else CASES.index(case)
    )
    generator = torch.Generator(device="cpu").manual_seed(
        seed + 100_000 * (family_index + 1)
    )
    dtype, dim, size = torch.float64, DIMENSION, DIMENSION**2

    def normal(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype)

    def basis():
        return torch.linalg.qr(normal(size, size)).Q

    initial = normal(dim, dim)
    eigenvalues = torch.logspace(0, math.log10(30), size, dtype=dtype)
    q = basis()
    first_hessian = (q * eigenvalues.unsqueeze(0)) @ q.T
    final_hessian = first_hessian.clone()
    switch_step = steps
    raw_noise = normal(steps, size)
    if case == "weak_singular_signal":
        signs = torch.where(normal(dim) >= 0, 1.0, -1.0)
        initial = torch.diag(torch.tensor([1, 3, 10, 30], dtype=dtype) * signs)
        curvature = torch.ones(dim, dim, dtype=dtype)
        curvature.diagonal().copy_(torch.tensor([1, 0.1, 0.01, 0.001], dtype=dtype))
        first_hessian = torch.diag(curvature.flatten())
        final_hessian = first_hessian.clone()
        noise = raw_noise * 0.003
    elif case in ("aligned_noise", "rotated_noise"):
        noise_basis = q if case == "aligned_noise" else basis()
        scale = (eigenvalues / eigenvalues.mean()).sqrt() * NOISE_RMS
        noise = (raw_noise * scale) @ noise_basis.T
    else:
        new_basis = basis()
        final_hessian = (new_basis * eigenvalues.unsqueeze(0)) @ new_basis.T
        switch_step = steps // 2
        noise = NOISE_RMS * raw_noise
    return QuadraticProblem(
        initial,
        noise.reshape(steps, dim, dim),
        first_hessian,
        final_hessian,
        switch_step,
    )


def method_settings(method):
    if method == "sgd":
        return {"momentum": 0.0}
    return validate_research_config(method, {})


def protocol(
    steps=64,
    seeds=TEST_SEEDS,
    *,
    tuning_seeds=TUNING_SEEDS,
    methods=METHODS,
    cases=CASES,
    lrs=LRS,
):
    """Resolve all choices before collecting a single development loss."""
    seeds, tuning_seeds = tuple(seeds), tuple(tuning_seeds)
    methods, cases, lrs = tuple(methods), tuple(cases), tuple(lrs)
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    for group in (seeds, tuning_seeds):
        if (
            not group
            or len(group) != len(set(group))
            or any(
                type(seed) is not int or not 0 <= seed < 2**63 - 1_000_000
                for seed in group
            )
        ):
            raise ValueError("Seed lists must contain distinct nonnegative integers")
    if set(seeds) & set(tuning_seeds):
        raise ValueError("Selection and test seeds must be disjoint")
    if not methods or len(methods) != len(set(methods)) or set(methods) - set(METHODS):
        raise ValueError("Invalid methods")
    if not cases or len(cases) != len(set(cases)) or set(cases) - set(CASES):
        raise ValueError("Invalid cases")
    if (
        not lrs
        or len(lrs) != len(set(lrs))
        or any(isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0 for lr in lrs)
    ):
        raise ValueError("Learning rates must be distinct positive finite numbers")
    lrs = tuple(sorted(lrs))
    files = [Path(__file__), Path(__file__).with_name("research_optimizers.py")]
    return {
        "version": 1,
        "kind": "synthetic_quadratic_falsification_not_language_model_training",
        "device": "cpu",
        "dtype": "float64",
        "torch_threads": 1,
        "torch_version": str(torch.__version__),
        "dimension": DIMENSION,
        "steps": steps,
        "tuning_seeds": list(tuning_seeds),
        "test_seeds": list(seeds),
        "methods": {method: method_settings(method) for method in methods},
        "cases": {case: CASE_DESCRIPTIONS[case] for case in cases},
        "learning_rates": list(lrs),
        "weight_decay": 0.0,
        "schedule": "constant learning rate",
        "gradient_convention": "one population gradient plus one precomputed noise sample per step",
        "divergence_loss_threshold": FAILURE_LOSS,
        "selection": "lowest arithmetic mean final population loss across all tuning seeds; any failure makes a configuration ineligible; ties choose smaller LR",
        "test": "freeze selected LR before test seeds; report every failure and require all test seeds for the aggregate",
        "boundary_policy": "flag lowest/highest LR choices; no automatic expansion and no superiority claim",
        "seed_derivation": "seed + 100000 * (family index + 1); aligned_noise and rotated_noise share index 0, other families use their fixed CASES index",
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
        "limitations": [
            "Tiny synthetic quadratics do not establish lower LM PPL or GPU cost.",
            "Equal LR counts do not establish equal tuning quality; boundary optima need expansion.",
            "Only LR is tuned; published optimizer defaults are not proven optimal here.",
            "The last case changes its population objective halfway through the run.",
            "Test seeds are held out from LR selection, but the preset problem families are designed.",
        ],
    }


@torch.no_grad()
def run_trial(case, method, lr, seed, steps):
    problem = draw_problem(case, seed, steps)
    parameter = torch.nn.Parameter(problem.initial.clone())
    optimizer = (
        torch.optim.SGD([parameter], lr=lr, momentum=0)
        if method == "sgd"
        else make_matrix_optimizer([parameter], method, lr=lr, weight_decay=0.0)
    )
    initial_loss = population_loss_and_gradient(parameter, problem.hessian(0))[0].item()
    result = {
        "case": case,
        "method": method,
        "lr": lr,
        "seed": seed,
        "steps_requested": steps,
        "steps_completed": 0,
        "status": "running",
        "initial_loss": initial_loss,
        "final_loss": None,
        "history": [],
    }
    start = time.perf_counter()
    try:
        for step in range(steps):
            hessian = problem.hessian(step)
            _, gradient = population_loss_and_gradient(parameter, hessian)
            parameter.grad = gradient + problem.noise[step]
            optimizer.step()
            result["steps_completed"] = step + 1
            loss = population_loss_and_gradient(parameter, hessian)[0].item()
            if not math.isfinite(loss) or loss > FAILURE_LOSS:
                result["status"] = "diverged"
                result["failure"] = (
                    f"Population loss nonfinite or above {FAILURE_LOSS:g}"
                )
                break
            if step == 0 or (step + 1) % max(1, steps // 4) == 0 or step + 1 == steps:
                result["history"].append({"step": step + 1, "population_loss": loss})
        else:
            result["status"] = "completed"
            result["final_loss"] = population_loss_and_gradient(
                parameter, problem.final_hessian
            )[0].item()
    except RuntimeError as error:
        result["status"] = "runtime_error"
        result["failure"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


def select_configuration(trials, learning_rates, tuning_seeds):
    """Any missing or failed seed invalidates that LR; test data never enter."""
    candidates = []
    for lr in sorted(learning_rates):
        rows = [row for row in trials if row["lr"] == lr]
        good = (
            len(rows) == len(tuning_seeds)
            and sorted(row["seed"] for row in rows) == sorted(tuning_seeds)
            and all(row["status"] == "completed" for row in rows)
        )
        candidates.append(
            {
                "lr": lr,
                "eligible": good,
                "mean_loss": statistics.mean(row["final_loss"] for row in rows)
                if good
                else None,
                "failed_or_missing_seeds": [
                    seed
                    for seed in tuning_seeds
                    if not any(
                        row["seed"] == seed and row["status"] == "completed"
                        for row in rows
                    )
                ],
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    selected = (
        min(eligible, key=lambda row: (row["mean_loss"], row["lr"]))
        if eligible
        else None
    )
    return {
        "selected_lr": selected["lr"] if selected else None,
        "selection_mean_loss": selected["mean_loss"] if selected else None,
        "lr_boundary": selected["lr"] in (min(learning_rates), max(learning_rates))
        if selected
        else None,
        "grid": candidates,
    }


def run_study(output, steps=64, seeds=TEST_SEEDS, **choices):
    spec = protocol(steps, seeds, **choices)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a fresh empty output directory")
    torch.set_num_threads(1)
    atomic_json(output / "protocol.json", spec)
    protocol_hash = fingerprint(spec)
    trials, locks = [], {}
    start = time.perf_counter()
    for case in spec["cases"]:
        locks[case] = {}
        for method in spec["methods"]:
            tuning = []
            for lr in spec["learning_rates"]:
                for seed in spec["tuning_seeds"]:
                    trial = run_trial(case, method, lr, seed, steps)
                    trial["phase"] = "selection"
                    trials.append(trial)
                    tuning.append(trial)
            locks[case][method] = select_configuration(
                tuning, spec["learning_rates"], spec["tuning_seeds"]
            )
        atomic_json(output / "trials.json", trials)
    # Persist every choice before drawing/optimizing any test-seed problem.
    atomic_json(
        output / "locked_configurations.json",
        {
            "protocol_sha256": protocol_hash,
            "configurations": locks,
        },
    )
    summary = {
        "protocol_sha256": protocol_hash,
        "kind": spec["kind"],
        "cases": {},
        "limitations": spec["limitations"],
    }
    for case in spec["cases"]:
        summary["cases"][case] = {}
        for method in spec["methods"]:
            selected = locks[case][method]
            test = []
            if selected["selected_lr"] is not None:
                for seed in spec["test_seeds"]:
                    trial = run_trial(
                        case, method, selected["selected_lr"], seed, steps
                    )
                    trial["phase"] = "test"
                    test.append(trial)
                    trials.append(trial)
            complete = len(test) == len(spec["test_seeds"]) and all(
                row["status"] == "completed" for row in test
            )
            losses = [row["final_loss"] for row in test] if complete else []
            summary["cases"][case][method] = selected | {
                "test_status": "completed" if complete else "incomplete_or_failed",
                "test_results": [
                    {key: row[key] for key in ("seed", "status", "final_loss")}
                    for row in test
                ],
                "mean_test_loss": statistics.mean(losses) if complete else None,
                "sample_sd_test_loss": statistics.stdev(losses)
                if len(losses) > 1
                else None,
            }
        baseline = summary["cases"][case].get("matrix_adamw")
        if baseline and baseline["test_status"] == "completed":
            baseline_losses = {
                row["seed"]: row["final_loss"] for row in baseline["test_results"]
            }
            for row in summary["cases"][case].values():
                if row["test_status"] == "completed":
                    differences = [
                        baseline_losses[test["seed"]] - test["final_loss"]
                        for test in row["test_results"]
                    ]
                    row["paired_adam_minus_method_losses"] = differences
                    row["mean_paired_adam_minus_method_loss"] = statistics.mean(
                        differences
                    )
        atomic_json(output / "trials.json", trials)
    summary["trial_counts"] = {
        "total": len(trials),
        "completed": sum(row["status"] == "completed" for row in trials),
        "failed": sum(row["status"] != "completed" for row in trials),
    }
    summary["elapsed_seconds"] = time.perf_counter() - start
    summary["h200_gpu_hours"] = 0.0
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh empty directory"
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(TEST_SEEDS),
        help="Fresh test-seed identifiers, separate from fixed selection seeds 11 and 22",
    )
    args = parser.parse_args()
    summary = run_study(args.output, args.steps, args.seeds)
    print(
        f"Synthetic probe: {summary['trial_counts']['total']} trials, "
        f"{summary['trial_counts']['failed']} failures; "
        f"{summary['elapsed_seconds']:.2f}s CPU; {args.output / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
