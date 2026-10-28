"""Preregistered CPU probes for temporal-uncertainty Shampoo shrinkage.

This is a separate study: it never edits the original research_probe defaults
or artifacts. Population quadratic losses and factor estimation errors are not
language-model PPL. Run with --output pointing to a fresh empty directory.
"""

import argparse
import hashlib
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from exp import research_probe as base
from exp.common import atomic_json, fingerprint
from exp.research_optimizers import (
    ema_weight_statistics,
    make_matrix_optimizer,
    trace_shrink,
    validate_research_config,
)

METHODS = ("shampoo", "shampoo_shrink", "shampoo_adaptive")
CASES = (
    "stationary_low_noise",
    "stationary_high_noise",
    "deterministic_weak_signal",
    "abrupt_curvature_drift",
)
STREAMS = (
    "constant_weak_signal",
    "anisotropic_low_noise",
    "anisotropic_high_noise",
    "isotropic_noise",
    "deterministic_drift",
)
LRS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
TEST_SEEDS = (1001, 2002, 3003)
STREAM_METHODS = {
    "shampoo": ("shampoo", {}),
    "shampoo_shrink": ("shampoo_shrink", {}),
    "shampoo_adaptive": ("shampoo_adaptive", {}),
    "adaptive_cap1_diagnostic": ("shampoo_adaptive", {"alpha_max": 1.0}),
}


@dataclass
class FactorStream:
    gradients: torch.Tensor
    left_truth: torch.Tensor
    right_truth: torch.Tensor
    switch_step: int


def draw_optimization_problem(case, seed, steps):
    if case in ("stationary_low_noise", "stationary_high_noise"):
        problem = base.draw_problem("aligned_noise", seed, steps)
        noise_rms = 0.03 if case == "stationary_low_noise" else 0.7
        problem.noise *= noise_rms / base.NOISE_RMS
    elif case == "deterministic_weak_signal":
        problem = base.draw_problem("weak_singular_signal", seed, steps)
        problem.noise.zero_()
    elif case == "abrupt_curvature_drift":
        problem = base.draw_problem("abrupt_curvature_rotation", seed, steps)
    else:
        raise ValueError(f"Unknown optimization case: {case}")
    return problem


def gaussian_factor_truth(mean, left_covariance, right_covariance, sigma):
    """For G=M+sigma L^.5 Z R^.5, IID standard normal Z, return E[GGt], E[GtG]."""
    return (
        mean @ mean.T + sigma**2 * right_covariance.trace() * left_covariance,
        mean.T @ mean + sigma**2 * left_covariance.trace() * right_covariance,
    )


def _sqrt_psd(matrix):
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(0)) @ vectors.T


def draw_factor_stream(case, seed, steps):
    if case not in STREAMS or type(steps) is not int or steps < 1:
        raise ValueError("Invalid factor stream or step count")
    if type(seed) is not int or not 0 <= seed < 2**63 - 8_100_000:
        raise ValueError("Factor stream seed is outside its safe range")
    generator = torch.Generator(device="cpu").manual_seed(seed + 8_100_000)
    dtype, dim = torch.float64, base.DIMENSION

    def normal(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype)

    left_basis = torch.linalg.qr(normal(dim, dim)).Q
    right_basis = torch.linalg.qr(normal(dim, dim)).Q
    left_spectrum = torch.tensor([0.1, 0.3, 1.0, 3.0], dtype=dtype)
    right_spectrum = torch.tensor([0.3, 0.7, 1.3, 2.0], dtype=dtype)
    left_spectrum *= dim / left_spectrum.sum()
    right_spectrum *= dim / right_spectrum.sum()
    left_cov = (left_basis * left_spectrum.unsqueeze(0)) @ left_basis.T
    right_cov = (right_basis * right_spectrum.unsqueeze(0)) @ right_basis.T
    signal = torch.diag(torch.tensor([1.0, 0.3, 0.1, 0.03], dtype=dtype))
    mean = left_basis @ signal @ right_basis.T
    second_mean = mean
    sigma = 0.01 if case == "anisotropic_low_noise" else 0.5
    switch_step = steps
    if case == "constant_weak_signal":
        sigma = 0.0
    elif case == "isotropic_noise":
        mean = torch.zeros_like(mean)
        second_mean = mean
        left_cov = right_cov = torch.eye(dim, dtype=dtype)
    elif case == "deterministic_drift":
        sigma = 0.0
        second_left = torch.linalg.qr(normal(dim, dim)).Q
        second_right = torch.linalg.qr(normal(dim, dim)).Q
        second_mean = second_left @ signal @ second_right.T
        switch_step = steps // 2
    white = normal(steps, dim, dim)
    colored = _sqrt_psd(left_cov) @ white @ _sqrt_psd(right_cov)
    means = mean.unsqueeze(0).repeat(steps, 1, 1)
    means[switch_step:] = second_mean
    gradients = means + sigma * colored
    first_left, first_right = gaussian_factor_truth(mean, left_cov, right_cov, sigma)
    second_left, second_right = gaussian_factor_truth(
        second_mean, left_cov, right_cov, sigma
    )
    left_truth = first_left.unsqueeze(0).repeat(steps, 1, 1)
    right_truth = first_right.unsqueeze(0).repeat(steps, 1, 1)
    left_truth[switch_step:] = second_left
    right_truth[switch_step:] = second_right
    return FactorStream(gradients, left_truth, right_truth, switch_step)


def oracle_shrinkage(estimate, truth, alpha_max=1.0):
    """Clairvoyant same-dimension Frobenius oracle, never used for training.

    Minimizes ||(1-a) C_hat + a tr(C_hat)/d I - C_truth||_F^2
    for 0 <= a <= alpha_max, for this one realized estimate.
    It is not an optimizer-step oracle or an unbiased estimator of optimal a.
    """
    if estimate.ndim != 2 or estimate.shape != truth.shape or estimate.shape[0] != estimate.shape[1]:
        raise ValueError("Oracle requires equally shaped square factors")
    if not math.isfinite(alpha_max) or not 0 <= alpha_max <= 1:
        raise ValueError("alpha_max must be in [0,1]")
    anisotropy = estimate - torch.eye(
        estimate.shape[0], dtype=estimate.dtype, device=estimate.device
    ) * estimate.diagonal().mean()
    denominator = anisotropy.square().sum()
    if denominator.item() == 0:
        return 0.0
    return (
        ((estimate - truth) * anisotropy).sum() / denominator
    ).clamp(0, alpha_max).item()


def _squared_error(estimate, truth):
    return (estimate - truth).square().sum().item()


def _all_finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def factor_metrics(estimate, truth, ema_truth, alpha):
    """Compare all policies in exactly the same factor dimension, before ridge."""
    transformed = trace_shrink(estimate, alpha)
    record = {}
    for label, target in (("current_truth", truth), ("expected_ema_truth", ema_truth)):
        energy = target.square().sum().item()
        raw_error = _squared_error(estimate, target)
        shrunk_error = _squared_error(transformed, target)
        record[label] = {
            "target_squared_norm": energy,
            "raw_squared_error": raw_error,
            "shrunk_squared_error": shrunk_error,
            "raw_relative_squared_error": raw_error / energy if energy > 0 else None,
            "shrunk_relative_squared_error": shrunk_error / energy if energy > 0 else None,
        }
        for cap in (0.1, 1.0):
            oracle = oracle_shrinkage(estimate, target, cap)
            record[label][f"oracle_cap_{cap:g}"] = {
                "alpha": oracle,
                "squared_error": _squared_error(trace_shrink(estimate, oracle), target),
            }
    return record


def protocol(steps=128, seeds=TEST_SEEDS, *, tuning_seeds=base.TUNING_SEEDS,
             lrs=LRS, cases=CASES, streams=STREAMS, methods=METHODS):
    # Reuse established seed/LR validation without changing its old registry.
    checked = base.protocol(
        steps, seeds, tuning_seeds=tuning_seeds, lrs=lrs,
        methods=("shampoo",), cases=("aligned_noise",),
    )
    if max(checked["test_seeds"]) >= 2**63 - 8_100_000:
        raise ValueError("Test seed is outside the factor-stream safe range")
    for chosen, allowed, label in ((cases, CASES, "cases"), (streams, STREAMS, "streams"),
                                   (methods, METHODS, "methods")):
        if not chosen or len(chosen) != len(set(chosen)) or set(chosen) - set(allowed):
            raise ValueError(f"Invalid {label}")
    files = [Path(__file__), Path(base.__file__), Path(__file__).with_name("research_optimizers.py")]
    return checked | {
        "version": 1,
        "kind": "adaptive_shrinkage_synthetic_mechanism_study_not_ppl",
        "methods": {method: validate_research_config(method, {}) for method in methods},
        "cases": list(cases),
        "streams": list(streams),
        "stream_methods": {
            label: {"method": method, "settings": validate_research_config(method, kwargs)}
            for label, (method, kwargs) in STREAM_METHODS.items()
        },
        "noise_pairing": "Low/high optimization noise share Hessian, initialization and standardized draws; low/high factor streams share means, covariance geometry and white draws.",
        "seed_derivation": {
            "stationary_low_noise": "base aligned_noise: seed + 100000",
            "stationary_high_noise": "base aligned_noise: seed + 100000",
            "deterministic_weak_signal": "base weak_singular_signal: seed + 300000; noise replaced with zero",
            "abrupt_curvature_drift": "base abrupt_curvature_rotation: seed + 400000",
            "all_factor_streams": "seed + 8100000; low/high anisotropic cases share every draw",
        },
        "factor_targets": "Current E[GGt]/E[GtG] and normalized EMA of these expectations; no curvature substitution.",
        "factor_noise_model": "G=M+sigma L^.5 Z R^.5; Z IID N(0,1); exact E[GGt]=MMt+sigma^2 tr(R)L.",
        "factor_protocol": "Replay identical fixed-parameter gradient streams at lr=0 for every method; score actual factor/alpha state only at root-refresh steps.",
        "factor_oracle": "Clairvoyant per-realization scalar Frobenius minimizer in the same factor dimension, caps .1 and 1; never feeds optimizer or selection.",
        "alpha_defaults": "Production candidate cap .1; cap1 is a separately labeled factor-stream diagnostic only.",
        "selection": "Select LR only on tuning seeds, using final quadratic loss; any failed seed makes LR ineligible; lock every case/method before all test trials and factor diagnostics.",
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "limitations": checked["limitations"] + [
            "Temporal factor variability combines stochastic noise, serial correlation, and genuine drift.",
            "Under IID stationary factor observations, q*V/(1-q) estimates the tracefree EMA-mean squared error; the ratio giving alpha is a plug-in heuristic, not an unbiased oracle coefficient.",
            "A smaller factor Frobenius error does not guarantee a better inverse root, optimizer direction, or final loss.",
            "The cap1 stream diagnostic is not an additional tuned optimizer comparison.",
            "Constant deterministic factor streams differ from deterministic optimization trajectories whose gradients change.",
        ],
    }


def _applied_alpha(block, side, method, settings):
    if method == "shampoo_adaptive":
        return block[f"{side}_alpha"].reshape(-1)[0].item()
    return settings["shrinkage"]


@torch.no_grad()
def run_trial(case, method, lr, seed, steps):
    problem = draw_optimization_problem(case, seed, steps)
    parameter = torch.nn.Parameter(problem.initial.clone())
    optimizer = make_matrix_optimizer([parameter], method, lr=lr, weight_decay=0.0)
    settings = validate_research_config(method, {})
    initial = base.population_loss_and_gradient(parameter, problem.hessian(0))[0].item()
    result = {
        "case": case, "method": method, "lr": lr, "seed": seed,
        "steps_requested": steps, "steps_completed": 0, "initial_loss": initial,
        "status": "running", "final_loss": None, "refresh_history": [],
    }
    start = time.perf_counter()
    try:
        for step in range(steps):
            hessian = problem.hessian(step)
            _, gradient = base.population_loss_and_gradient(parameter, hessian)
            parameter.grad = gradient + problem.noise[step]
            optimizer.step()
            result["steps_completed"] = step + 1
            loss = base.population_loss_and_gradient(parameter, hessian)[0].item()
            if not math.isfinite(loss) or loss > base.FAILURE_LOSS:
                result["status"] = "diverged"
                result["failure"] = "Nonfinite population loss or divergence threshold exceeded"
                break
            if step % settings["update_frequency"] == 0:
                block = optimizer.state[parameter]["blocks"][0]
                result["refresh_history"].append({
                    "step": step + 1, "population_loss": loss,
                    "left_alpha": _applied_alpha(block, "left", method, settings),
                    "right_alpha": _applied_alpha(block, "right", method, settings),
                })
        else:
            result["status"] = "completed"
            result["final_loss"] = base.population_loss_and_gradient(
                parameter, problem.final_hessian
            )[0].item()
    except RuntimeError as error:
        result["status"] = "runtime_error"
        result["failure"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


@torch.no_grad()
def run_factor_diagnostic(case, label, seed, steps):
    stream = draw_factor_stream(case, seed, steps)
    method, kwargs = STREAM_METHODS[label]
    settings = validate_research_config(method, kwargs)
    parameter = torch.nn.Parameter(torch.zeros_like(stream.gradients[0]))
    optimizer = make_matrix_optimizer([parameter], method, lr=0.0, weight_decay=0.0, **kwargs)
    beta = settings["beta2"]
    expected = {"left": torch.zeros_like(parameter), "right": torch.zeros_like(parameter)}
    result = {
        "case": case, "method": label, "seed": seed, "status": "running",
        "steps_completed": 0, "records": [],
    }
    try:
        for index, gradient in enumerate(stream.gradients):
            step = index + 1
            parameter.grad = gradient.clone()
            optimizer.step()
            result["steps_completed"] = step
            current = {"left": stream.left_truth[index], "right": stream.right_truth[index]}
            for side, expected_value in expected.items():
                expected_value.mul_(beta).add_(current[side], alpha=1 - beta)
            if index % settings["update_frequency"] != 0:
                continue
            block = optimizer.state[parameter]["blocks"][0]
            _, q = ema_weight_statistics(beta, step)
            for side, expected_value in expected.items():
                estimate = block[side][0] / (1 - beta**step)
                alpha = _applied_alpha(block, side, method, settings)
                scatter = block.get(f"{side}_scatter")
                scatter = scatter.reshape(-1)[0].item() if scatter is not None else None
                if (
                    not math.isfinite(alpha) or not 0 <= alpha <= 1
                    or not torch.isfinite(estimate).all()
                    or (scatter is not None and not math.isfinite(scatter))
                ):
                    result["status"] = "nonfinite_factor"
                    result["failure"] = "Invalid factor, scatter, or shrinkage coefficient"
                    return result
                record = {
                    "step": step, "side": side, "alpha": alpha,
                    "after_drift": index >= stream.switch_step,
                    "weight_square_sum": q, "effective_sample_count": 1 / q,
                    "scatter": scatter,
                    "temporal_uncertainty_estimate": q * scatter / (1 - q)
                    if scatter is not None and q < 1 else None,
                    "uncertainty_estimate_defined": scatter is not None and q < 1,
                    "anisotropy_energy": (
                        estimate - torch.eye(estimate.shape[0], dtype=estimate.dtype)
                        * estimate.diagonal().mean()
                    ).square().sum().item(),
                }
                if not _all_finite(record):
                    result["status"] = "nonfinite_diagnostic"
                    result["failure"] = "Nonfinite derived scalar diagnostic"
                    return result
                record.update(factor_metrics(
                    estimate, current[side], expected_value / (1 - beta**step), alpha
                ))
                if not _all_finite(record):
                    result["status"] = "nonfinite_diagnostic"
                    result["failure"] = "Nonfinite factor error or oracle diagnostic"
                    return result
                result["records"].append(record)
        result["status"] = "completed"
    except (RuntimeError, ValueError) as error:
        result["status"] = "runtime_error"
        result["failure"] = f"{type(error).__name__}: {error}"
    return result


def _test_summary(selected, trials, seeds):
    complete = len(trials) == len(seeds) and all(row["status"] == "completed" for row in trials)
    losses = [row["final_loss"] for row in trials] if complete else []
    return selected | {
        "test_status": "completed" if complete else "incomplete_or_failed",
        "test_results": [
            {key: row[key] for key in ("seed", "status", "final_loss")} for row in trials
        ],
        "mean_test_loss": statistics.mean(losses) if complete else None,
        "sample_sd_test_loss": statistics.stdev(losses) if len(losses) > 1 else None,
    }


def _factor_summary(diagnostics):
    summary = {}
    for case in dict.fromkeys(row["case"] for row in diagnostics):
        summary[case] = {}
        for method in STREAM_METHODS:
            runs = [row for row in diagnostics if row["case"] == case and row["method"] == method]
            sides = {}
            for side in ("left", "right"):
                final = [
                    [record for record in row["records"] if record["side"] == side][-1]
                    for row in runs if row["status"] == "completed"
                ]
                complete = len(final) == len(runs) and bool(runs)
                sides[side] = {
                    "status": "completed" if complete else "incomplete_or_failed",
                    "mean_final_refresh_alpha": statistics.mean(row["alpha"] for row in final) if complete else None,
                }
                for target in ("current_truth", "expected_ema_truth"):
                    sides[side][target] = {
                        key: statistics.mean(row[target][key] for row in final) if complete else None
                        for key in ("raw_squared_error", "shrunk_squared_error")
                    }
                    for cap in (0.1, 1.0):
                        key = f"oracle_cap_{cap:g}"
                        sides[side][target][key] = {
                            metric: statistics.mean(row[target][key][metric] for row in final) if complete else None
                            for metric in ("alpha", "squared_error")
                        }
            summary[case][method] = sides
    return summary


def run_study(output, steps=128, seeds=TEST_SEEDS, **choices):
    spec = protocol(steps, seeds, **choices)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a fresh empty output directory")
    torch.set_num_threads(1)
    atomic_json(output / "protocol.json", spec)
    protocol_hash = fingerprint(spec)
    start = time.perf_counter()
    trials, locked = [], {}
    for case in spec["cases"]:
        locked[case] = {}
        for method in spec["methods"]:
            selection = []
            for lr in spec["learning_rates"]:
                for seed in spec["tuning_seeds"]:
                    row = run_trial(case, method, lr, seed, steps) | {"phase": "selection"}
                    trials.append(row)
                    selection.append(row)
            locked[case][method] = base.select_configuration(
                selection, spec["learning_rates"], spec["tuning_seeds"]
            )
        atomic_json(output / "trials.json", trials)
    atomic_json(output / "locked_configurations.json", {
        "protocol_sha256": protocol_hash, "configurations": locked,
    })
    summary = {
        "protocol_sha256": protocol_hash, "kind": spec["kind"],
        "cases": {}, "limitations": spec["limitations"],
    }
    for case in spec["cases"]:
        summary["cases"][case] = {}
        for method in spec["methods"]:
            selected, test = locked[case][method], []
            if selected["selected_lr"] is not None:
                for seed in spec["test_seeds"]:
                    row = run_trial(case, method, selected["selected_lr"], seed, steps)
                    row["phase"] = "test"
                    test.append(row)
                    trials.append(row)
            summary["cases"][case][method] = _test_summary(selected, test, spec["test_seeds"])
        parent = summary["cases"][case].get("shampoo")
        if parent and parent["test_status"] == "completed":
            parent_losses = {row["seed"]: row["final_loss"] for row in parent["test_results"]}
            for method in summary["cases"][case].values():
                if method["test_status"] == "completed":
                    differences = [parent_losses[row["seed"]] - row["final_loss"] for row in method["test_results"]]
                    method["paired_parent_minus_method_loss"] = differences
                    method["mean_paired_parent_minus_method_loss"] = statistics.mean(differences)
        atomic_json(output / "trials.json", trials)
    diagnostics = []
    for case in spec["streams"]:
        for method in STREAM_METHODS:
            for seed in spec["test_seeds"]:
                diagnostics.append(run_factor_diagnostic(case, method, seed, steps))
        atomic_json(output / "factor_diagnostics.json", diagnostics)
    summary["factor_diagnostics"] = _factor_summary(diagnostics)
    summary["trial_counts"] = {
        "optimization_total": len(trials),
        "optimization_failed": sum(row["status"] != "completed" for row in trials),
        "factor_stream_total": len(diagnostics),
        "factor_stream_failed": sum(row["status"] != "completed" for row in diagnostics),
    }
    summary["elapsed_seconds"] = time.perf_counter() - start
    summary["h200_gpu_hours"] = 0.0
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TEST_SEEDS))
    args = parser.parse_args()
    summary = run_study(args.output, args.steps, args.seeds)
    print(f"Adaptive shrinkage CPU probe: {summary['trial_counts']}; {summary['elapsed_seconds']:.3f}s; {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
