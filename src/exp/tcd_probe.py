"""Balanced synthetic falsification study for the exact TCD reference rule.

These are 4x4 quadratic objectives, NOT language-model training results.
Each method gets 24 configurations, three tuning seeds, then eight new seeds.
All consume exactly four independent gradient samples per optimization step.
"""

import argparse
import hashlib
import math
import shutil
import time
from pathlib import Path

import torch

from exp.common import atomic_json
from exp.tcd import crossfit_direction, precondition, precondition_first_order
from muon import zeropower_via_newtonschulz5

METHODS = ("sgd", "adam", "normuon", "tcd", "no_coupling", "same_fold", "no_trace")
CASES = ("common_mode_noise", "isotropic_noise", "rotated_curvature")
EXTRA_CASES = ("common_mode_symmetric_signal",)
LRS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
TUNE_SEEDS = (11, 12, 13)
TEST_SEEDS = tuple(range(101, 109))


def grid(method):
    secondary = (
        (0.0, 0.9, 0.95)
        if method in ("sgd", "normuon")
        else ((0.9, 0.99, 0.999) if method == "adam" else (0.3, 3.0, 30.0))
    )
    return [{"lr": lr, "secondary": value} for lr in LRS for value in secondary]


def rank_one_noise_control(samples, rho):
    """Simpler noise-covariance inverse via Sherman--Morrison, not a novelty claim.

    Same fold split, gate, spectral range and trace compensation as TCD.
    This tests whether the Gram penalty adds anything over vector noise filtering.
    """
    g1, g2, g3, g4 = samples.unbind(-3)
    a, b = (g1 + g2) / 2, (g3 + g4) / 2
    ea, eb = (g1 - g2) / 2, (g3 - g4) / 2

    def solve(g, e, mean):
        energy = e.square().sum((-2, -1), keepdim=True)
        tiny = torch.finfo(e.dtype).tiny
        gate = energy / (energy + mean.square().sum((-2, -1), keepdim=True)).clamp_min(tiny)
        alpha = rho * gate / (1 + rho * gate)
        q = g - alpha * e * (e * g).sum((-2, -1), keepdim=True) / energy.clamp_min(tiny)
        return q / (1 - alpha / (e.shape[-2] * e.shape[-1]))

    return (solve(b, ea, a) + solve(a, eb, b)) / 2


def draw_problem(case, seed, steps):
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    initial = normal(4, 4)
    if case in ("common_mode_noise", "common_mode_symmetric_signal"):
        if case == "common_mode_noise":
            initial = (initial - initial.T) / 2
        else:
            initial = (initial + initial.T) / 2
            initial -= torch.eye(4, dtype=initial.dtype) * initial.trace() / 4
        noise = 0.03 * normal(steps, 4, 4, 4)
        noise += 3 * normal(steps, 4, 1, 1) * torch.eye(4, dtype=torch.float64)
        hessian = torch.eye(16, dtype=torch.float64)
    elif case == "isotropic_noise":
        noise = 0.5 * normal(steps, 4, 4, 4)
        hessian = torch.eye(16, dtype=torch.float64)
    else:
        noise = 0.2 * normal(steps, 4, 4, 4)
        q, _ = torch.linalg.qr(normal(16, 16))
        hessian = q @ torch.diag(torch.logspace(0, 2, 16, dtype=torch.float64)) @ q.T
    return initial, noise, hessian


@torch.no_grad()
def simulate(case, method, configs, seeds, steps, solver="exact"):
    problems = [draw_problem(case, seed, steps) for seed in seeds]
    initial = torch.stack([p[0] for p in problems])
    noise = torch.stack([p[1] for p in problems], dim=1)
    hessian = torch.stack([p[2] for p in problems])
    w = initial.unsqueeze(0).repeat(len(configs), 1, 1, 1)
    momentum, variance = torch.zeros_like(w), torch.zeros_like(w)
    row_sq = torch.zeros_like(w[..., :1])
    bad = torch.zeros(w.shape[:2], dtype=torch.bool)
    rates = torch.tensor([c["lr"] for c in configs], dtype=w.dtype)[:, None, None, None]
    secondary = torch.tensor([c["secondary"] for c in configs], dtype=w.dtype)[
        :, None, None, None
    ]
    history = []
    start = time.perf_counter()
    for step in range(steps):
        population = torch.einsum("sij,csj->csi", hessian, w.flatten(-2)).reshape_as(w)
        samples = population.unsqueeze(-3) + noise[step].unsqueeze(0)
        g = samples.mean(-3)
        if method == "sgd":
            momentum = secondary * momentum + (1 - secondary) * g
            direction = momentum / (1 - secondary ** (step + 1))
        elif method == "adam":
            momentum.lerp_(g, 0.1)
            variance = secondary * variance + (1 - secondary) * g.square()
            direction = (momentum / (1 - 0.9 ** (step + 1))) / (
                (variance / (1 - secondary ** (step + 1))).sqrt() + 1e-8
            )
        elif method == "normuon":
            momentum = secondary * momentum + (1 - secondary) * g
            nest = (1 - secondary) * g + secondary * momentum
            polar = zeropower_via_newtonschulz5(nest, polar_express=True).double()
            norm = polar.norm(dim=(-2, -1), keepdim=True)
            row_sq.lerp_(polar.square().mean(-1, keepdim=True), 0.05)
            direction = polar / (row_sq.sqrt() + 1e-10)
            direction *= norm / (direction.norm(dim=(-2, -1), keepdim=True) + 1e-10)
            direction *= 0.2 * math.sqrt(4)
        else:
            direction = torch.stack(
                [
                    rank_one_noise_control(samples[i], cfg["secondary"])
                    if method == "rank_one_noise"
                    else crossfit_direction(
                        samples[i],
                        cfg["secondary"],
                        coupled=method != "no_coupling",
                        normalize_trace=method != "no_trace",
                        crossfit=method != "same_fold",
                        solver="first_order" if method == "tcd_first_order" else solver,
                    )
                    for i, cfg in enumerate(configs)
                ]
            )
        w -= rates * direction
        invalid = ~w.isfinite().all((-2, -1)) | (w.abs().amax((-2, -1)) > 1e20)
        bad |= invalid
        w[bad] = 0  # Preserve failed outcomes, prevent contamination of batched SVD.
        if step % 10 == 0 or step == steps - 1:
            value = 0.5 * torch.einsum(
                "csi,sij,csj->cs", w.flatten(-2), hessian, w.flatten(-2)
            )
            history.append(
                {
                    "step": step + 1,
                    "loss": value.masked_fill(bad, float("inf")).tolist(),
                }
            )
    losses = history[-1]["loss"]
    return losses, history, time.perf_counter() - start


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def run(root, steps=200, solver="exact", seed_offset=0, extended_controls=False, cases=CASES):
    root.mkdir(parents=True, exist_ok=True)
    tuning_seeds = tuple(seed + seed_offset for seed in TUNE_SEEDS)
    test_seeds = tuple(seed + seed_offset for seed in TEST_SEEDS)
    methods = METHODS + (("rank_one_noise", "tcd_first_order") if extended_controls else ())
    if extended_controls and solver != "exact":
        raise ValueError("Use exact solver with extended controls to compare both solvers")
    protocol = {
        "cases": cases,
        "methods": methods,
        "steps": steps,
        "tuning_seeds": tuning_seeds,
        "test_seeds": test_seeds,
        "tcd_solver": solver,
        "grid": {method: grid(method) for method in methods},
        "gradient_samples_per_step": 4,
        "decision": "Mechanism diagnostic only; no LLM performance inference",
        "schedule": "constant LR, no weight decay; true quadratic objective",
    }
    if (root / "protocol.json").exists():
        raise ValueError("Use a fresh output directory to preserve prior evidence")
    source = root / "source"
    source.mkdir()
    here = Path(__file__).resolve().parent
    paths = [here / name for name in ("tcd.py", "tcd_probe.py", "test_tcd.py")]
    paths += [here.parent / "muon.py"]
    manifest = {}
    for path in paths:
        shutil.copyfile(path, source / path.name)
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol["source_sha256"] = manifest
    atomic_json(root / "protocol.json", protocol)
    summary = {}
    for case in cases:
        tuning = {}
        for method in methods:
            configs = grid(method)
            losses, _, seconds = simulate(case, method, configs, tuning_seeds, steps, solver)
            scores = [sum(row) / len(row) for row in losses]
            best = min(range(len(configs)), key=lambda i: scores[i])
            tuning[method] = {
                "configs": configs,
                "losses": losses,
                "selected": configs[best],
                "seconds": seconds,
                "selected_lr_boundary": configs[best]["lr"] in (min(LRS), max(LRS)),
                "selected_secondary_boundary": configs[best]["secondary"]
                in (min(c["secondary"] for c in configs), max(c["secondary"] for c in configs)),
            }
        # Lock all methods before evaluating any held-out seed in this case.
        atomic_json(root / f"{case}-selection.json", finite_json(tuning))
        confirmation = {}
        for method in methods:
            loss, history, seconds = simulate(
                case, method, [tuning[method]["selected"]], test_seeds, steps, solver
            )
            confirmation[method] = {
                "losses": loss[0],
                "mean_loss": sum(loss[0]) / len(loss[0]),
                "seconds": seconds,
                "history": history,
            }
        atomic_json(root / f"{case}-test.json", finite_json(confirmation))
        summary[case] = {
            method: {
                "mean_loss": confirmation[method]["mean_loss"],
                "selected": tuning[method]["selected"],
                "lr_boundary": tuning[method]["selected_lr_boundary"],
                "secondary_boundary": tuning[method]["selected_secondary_boundary"],
            }
            for method in methods
        }
        print(case, finite_json(summary[case]), flush=True)
        atomic_json(root / "summary.json", finite_json(summary))
    e = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    g = torch.tensor([[0.0, -1.0], [0.0, 0.0]], dtype=torch.float64)
    solve = precondition if solver == "exact" else precondition_first_order
    d, info = solve(g, e, torch.zeros_like(e), 3)
    after = -0.1 * d
    failure = {
        "initial_loss": 0.5,
        "step_size": 0.1,
        "direction": d.tolist(),
        "gradient_alignment": (g * d).sum().item(),
        "final_loss": (0.5 * (after[0, 1] - 1) ** 2 + 5000 * after[1, 0] ** 2).item(),
        "trace_mean": info.get("inverse_trace_mean", info.get("trace_mean")).item(),
        "solver": solver,
        "interpretation": "Positive first-order alignment does not control unobserved curvature",
    }
    atomic_json(root / "curvature_counterexample.json", failure)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--solver", choices=("exact", "first_order"), default="exact")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--extended-controls", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=CASES + EXTRA_CASES, default=CASES)
    options = parser.parse_args()
    torch.set_num_threads(2)
    run(
        options.output,
        options.steps,
        options.solver,
        options.seed_offset,
        options.extended_controls,
        options.cases,
    )
