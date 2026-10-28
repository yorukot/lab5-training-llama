"""Loss-variance correction using regularized covariance and Woodbury.

This is a diagnostic replacement for TCD's Gram risk, not a novelty claim for
covariance inversion or cross-fitting. Unobserved-noise directions are fixed.
"""

import math

import torch


def noise_factors(half):
    """Four equal-weight mean gradients; covariance of their mean.

    For iid samples with covariance Sigma, E[Z.T@Z/12] = Sigma/4.
    Return mean, trace-one factor B, and estimated covariance trace v.
    """
    if half.shape[0] != 4 or half.ndim != 3:
        raise ValueError("Expected four mean matrix gradients")
    mean = half.mean(0)
    residual = (half - mean).flatten(1)
    energy = residual.square().sum()
    factors = residual / energy.sqrt().clamp_min(torch.finfo(half.dtype).tiny)
    return mean, factors, energy / 12


def covariance_precondition(numerator, half, rho=3.0):
    """Minimize .5||d-u||² + .5*lambda*||B d||², up to roundoff.

    lambda=rho*v/(v+||mean||²). Inverse eigenvalues lie in [1/(1+rho),1].
    Zero noise returns identity; no trace compensation amplifies the nullspace.
    """
    if rho == 0:
        return numerator
    if rho < 0 or not math.isfinite(rho):
        raise ValueError("rho must be finite and nonnegative")
    mean, b, variance = noise_factors(half)
    gate = variance / (variance + mean.square().sum()).clamp_min(
        torch.finfo(half.dtype).tiny
    )
    strength = rho * gate
    system = torch.eye(4, device=half.device, dtype=half.dtype) + strength * (b @ b.T)
    rhs = b @ numerator.flatten()
    # SPD with condition number <=1+rho in exact arithmetic. No per-matrix CPU
    # sync; tests check this solve against the full-dimensional covariance inverse.
    coefficients, _ = torch.linalg.solve_ex(system, rhs, check_errors=False)
    return numerator - (strength * (b.T @ coefficients)).reshape_as(numerator)


def covariance_direction(rank_means, old_momentum, rho=3.0, mu=0.95):
    """Eight means; opposite-fold covariance and opposite-fold RMS scale."""
    if rank_means.shape[0] != 8 or rank_means.ndim != 3:
        raise ValueError("Expected eight mean matrix gradients")
    ha, hb = rank_means[:4], rank_means[4:]
    a, b = ha.mean(0), hb.mean(0)
    na = mu * mu * old_momentum + (1 - mu * mu) * a
    nb = mu * mu * old_momentum + (1 - mu * mu) * b
    da = covariance_precondition(nb, ha, rho)
    db = covariance_precondition(na, hb, rho)
    da = da / na.square().mean().sqrt().clamp_min(1e-8)
    db = db / nb.square().mean().sqrt().clamp_min(1e-8)
    direction = (da + db) * (0.5 / math.sqrt(rank_means.shape[-1]))
    next_momentum = mu * old_momentum + (1 - mu) * (a + b) / 2
    return direction, next_momentum
