"""Enumerate a finite distribution to audit the conditional descent derivation.

This is an exact finite-distribution check, not a language-model benchmark.
It does not use or inspect development/held-out text.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

from exp.covariance_revision.covariance import covariance_direction


def probe():
    torch.set_num_threads(2)
    choices = torch.tensor(
        [[[1.0, 2.0], [0.5, -0.5]], [[0.5, 1.0], [-0.5, 0.25]]],
        dtype=torch.float64,
    )
    mu = choices.mean(0)
    old = 0.9 * mu
    beta, rho, k, fan_in = 0.95, 3.0, 4, 2
    nu = beta**2 * old + (1 - beta**2) * mu
    halves = [choices[list(ids)] for ids in itertools.product(range(2), repeat=k)]
    operators, scales = [], []
    for half in halves:
        mean = half.mean(0)
        residual = (half - mean).flatten(1)
        covariance = residual.T @ residual / (k * (k - 1))
        variance = covariance.trace()
        gate = variance / (variance + mean.square().sum())
        # Dense inverse here is independent of the implementation's Woodbury solve.
        penalty = rho * gate * covariance / variance if variance > 0 else covariance
        inverse = torch.linalg.inv(torch.eye(mu.numel()) + penalty)
        numerator = beta**2 * old + (1 - beta**2) * mean
        scale = numerator.square().mean().sqrt().clamp_min(1e-8).item()
        scales.append(scale)
        operators.append(inverse / (math.sqrt(fan_in) * scale))
    directions = torch.stack(
        [
            covariance_direction(torch.cat([a, b]), old, rho, beta)[0]
            for a, b in itertools.product(halves, repeat=2)
        ]
    )
    actual_mean = directions.mean(0)
    predicted_mean = (torch.stack(operators).mean(0) @ nu.flatten()).reshape_as(mu)
    mean_error = (actual_mean - predicted_mean).abs().max().item()
    assert mean_error < 1e-12
    lower = 1 / (math.sqrt(fan_in) * max(scales) * (1 + rho))
    upper = 1 / (math.sqrt(fan_in) * min(scales))
    c = lower * mu.square().sum() - beta**2 * upper * mu.norm() * (old - mu).norm()
    noise_trace = (choices - mu).square().sum() / len(choices)
    second_bound = upper**2 * (
        nu.square().sum() + (1 - beta**2) ** 2 * noise_trace / k
    )
    second_actual = directions.square().sum((1, 2)).mean()
    assert second_actual <= second_bound + 1e-12
    assert (mu * actual_mean).sum() >= c - 1e-12
    smoothness = 2.0
    eta_limit = (2 * c / (smoothness * second_bound)).item()
    assert eta_limit > 0
    eta = eta_limit / 2
    actual_change = (
        -eta * (directions * mu).sum((1, 2))
        + smoothness * eta**2 / 2 * directions.square().sum((1, 2))
    ).mean()
    bound = -eta * c + smoothness * eta**2 / 2 * second_bound
    assert actual_change <= bound + 1e-12 and bound < 0
    return {
        "scope": "finite-distribution theorem audit; not LLM performance evidence",
        "uniform_equiprobable_eight_sample_outcomes": len(directions),
        "conditional_mean_max_abs_error": mean_error,
        "second_moment_actual": second_actual.item(),
        "second_moment_upper_bound": second_bound.item(),
        "descent_coefficient_lower_bound": c.item(),
        "sufficient_lr_upper_bound": eta_limit,
        "tested_lr": eta,
        "expected_quadratic_loss_change": actual_change.item(),
        "loss_change_upper_bound": bound.item(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
