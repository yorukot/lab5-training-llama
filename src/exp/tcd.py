"""Transpose-Coupled Descent: exact, deliberately experimental reference core.

This module does not call Adam, Muon, polar decomposition, or Newton--Schulz.
It solves a quadratic Gram-disagreement penalty in a noise residual's SVD
basis, including transpose coupling. Cross-fitting itself is prior work.
See the TCD research section in README.md for assumptions and counterexamples.
"""

import math

import torch


def gram_risk(noise, direction):
    left = noise.mT @ direction + direction.mT @ noise
    right = noise @ direction.mT + direction @ noise.mT
    return (left.square().sum((-2, -1)) + right.square().sum((-2, -1))) / 4


def risk_operator(noise, direction, coupled=True):
    value = noise @ (noise.mT @ direction) + (direction @ noise.mT) @ noise
    if coupled:
        value = value + 2 * (noise @ direction.mT) @ noise
    return value


def precondition(
    gradient, noise, fold_mean, rho=3.0, *, coupled=True, normalize_trace=True
):
    """Apply a linear operator to gradient, with geometry from another fold.

    Leading batch dimensions are supported. The return is (direction, info).
    `fold_mean` and `noise` must come from the same fold, independent of the
    numerator gradient for the conditional expectation argument to apply.
    All arithmetic uses the input dtype; reference tests use float64.
    """
    if gradient.shape != noise.shape or noise.shape != fold_mean.shape:
        raise ValueError("Gradient, noise, and fold mean must have identical shapes")
    if gradient.ndim < 2 or not math.isfinite(rho) or rho < 0:
        raise ValueError("Expected matrix inputs and finite nonnegative rho")
    if gradient.dtype not in (torch.float32, torch.float64):
        raise ValueError("The exact reference solver requires float32 or float64")
    if rho == 0:
        one = gradient.new_ones(gradient.shape[:-2])
        return gradient.clone(), {
            "inverse_trace_mean": one,
            "gate": one * 0,
            "gamma": one * 0,
        }
    transpose = gradient.shape[-2] > gradient.shape[-1]
    g, e, mean = (x.mT if transpose else x for x in (gradient, noise, fold_mean))
    m, n = e.shape[-2:]
    u, singular, vh = torch.linalg.svd(e, full_matrices=False)
    tiny = torch.finfo(e.dtype).tiny
    noise_energy = e.square().sum((-2, -1))
    gate = noise_energy / (noise_energy + mean.square().sum((-2, -1))).clamp_min(tiny)
    gamma = rho * gate / (4 * singular[..., 0].square()).clamp_min(tiny)
    gamma = torch.where(singular[..., 0] > 0, gamma, torch.zeros_like(gamma))
    s, t = singular.unsqueeze(-1), singular.unsqueeze(-2)
    if coupled:
        plus = 1 / (1 + gamma[..., None, None] * (s + t).square())
        minus = 1 / (1 + gamma[..., None, None] * (s - t).square())
        trace = (
            plus.sum((-2, -1))
            + minus.sum((-2, -1))
            + plus.diagonal(dim1=-2, dim2=-1).sum(-1)
            - m
        ) / 2
    else:
        plus = 1 / (1 + gamma[..., None, None] * (s.square() + t.square()))
        minus = plus
        trace = plus.sum((-2, -1))
    perpendicular = 1 / (1 + gamma[..., None] * singular.square())
    trace = (trace + (n - m) * perpendicular.sum(-1)) / (m * n)
    rotated = u.mT @ g
    core = rotated @ vh.mT
    solved_core = (core + core.mT) * plus / 2 + (core - core.mT) * minus / 2
    # Thin SVD: the right-nullspace complement is handled without a full V.
    solved = solved_core @ vh + (rotated - core @ vh) * perpendicular.unsqueeze(-1)
    result = u @ solved
    if normalize_trace:
        result = result / trace[..., None, None]
    return (result.mT if transpose else result), {
        "inverse_trace_mean": trace,
        "gate": gate,
        "gamma": gamma,
    }


def precondition_first_order(
    gradient, noise, fold_mean, rho=3.0, *, coupled=True, normalize_trace=True
):
    """One conservative gradient step on the direction objective, without SVD.

    This is a distinct approximation, not the exact inverse above. A Schatten
    norm bounds the noise operator; no iteration assumes a known eigenvector.
    The trace and positive-definiteness bounds hold in exact arithmetic.
    """
    if gradient.shape != noise.shape or noise.shape != fold_mean.shape:
        raise ValueError("Gradient, noise, and fold mean must have identical shapes")
    if gradient.ndim < 2 or not math.isfinite(rho) or rho < 0:
        raise ValueError("Expected matrix inputs and finite nonnegative rho")
    if gradient.dtype not in (torch.float32, torch.float64):
        raise ValueError("The reference solver requires float32 or float64")
    transpose = gradient.shape[-2] > gradient.shape[-1]
    g, raw, mean = (x.mT if transpose else x for x in (gradient, noise, fold_mean))
    m, n = raw.shape[-2:]
    tiny = torch.finfo(raw.dtype).tiny
    energy = raw.square().sum((-2, -1))
    gate = energy / (energy + mean.square().sum((-2, -1))).clamp_min(tiny)
    e = raw / energy.sqrt().clamp_min(tiny)[..., None, None]
    a = e @ e.mT
    a2 = a @ a
    a4 = a2 @ a2
    # ||A^4||_F^(1/4) >= lambda_max(A) for symmetric PSD A.
    bound = 4 * a4.norm(dim=(-2, -1)).pow(0.25)
    alpha = rho * gate / (1 + rho * gate)
    coefficient = alpha / bound.clamp_min(tiny)
    t = g @ e.mT
    k = a @ g + (t + 2 * t.mT if coupled else t) @ e
    result = g - coefficient[..., None, None] * k
    trace_k = (m + n + (2 if coupled else 0)) * e.square().sum((-2, -1))
    trace = 1 - coefficient * trace_k / (m * n)
    if normalize_trace:
        result = result / trace[..., None, None]
    return (result.mT if transpose else result), {
        "trace_mean": trace,
        "gate": gate,
        "coefficient": coefficient,
        "operator_bound": bound,
    }


def crossfit_direction(
    samples,
    rho=3.0,
    *,
    coupled=True,
    normalize_trace=True,
    crossfit=True,
    solver="exact",
):
    """Four equal-size, same-parameter microbatch MEAN gradients (...,4,m,n).

    `crossfit=False` is an explicit same-fold ablation, not the proposed rule.
    No per-step norm grafting or current-batch global clipping is hidden here.
    """
    if samples.ndim < 3 or samples.shape[-3] != 4:
        raise ValueError("Exactly four microbatch mean gradients are required")
    if solver not in ("exact", "first_order"):
        raise ValueError("Unknown TCD solver")
    if rho == 0:
        return samples.mean(dim=-3)
    g1, g2, g3, g4 = samples.unbind(dim=-3)
    a, b = (g1 + g2) / 2, (g3 + g4) / 2
    ea, eb = (g1 - g2) / 2, (g3 - g4) / 2
    solve = precondition if solver == "exact" else precondition_first_order
    da, _ = solve(
        b if crossfit else a,
        ea,
        a,
        rho,
        coupled=coupled,
        normalize_trace=normalize_trace,
    )
    db, _ = solve(
        a if crossfit else b,
        eb,
        b,
        rho,
        coupled=coupled,
        normalize_trace=normalize_trace,
    )
    return (da + db) / 2


class TCD(torch.optim.Optimizer):
    """Reference optimizer for matrix parameters supplied with four gradients.

    Usage: opt.set_samples({parameter: tensor_of_shape_4_m_n}); opt.step().
    The samples replace .grad as the source of the update. This explicit API
    avoids silently treating accumulated gradients as independent samples.
    Momentum is optional; the basic expectation theorem assumes momentum=0.
    This is NOT wired into the production/150M Trainer yet.
    """

    def __init__(
        self, params, lr=1e-3, rho=3.0, momentum=0.0, weight_decay=0.0, solver="exact"
    ):
        if not all(math.isfinite(v) for v in (lr, rho, momentum, weight_decay)):
            raise ValueError("Hyperparameters must be finite")
        if min(lr, rho, weight_decay) < 0 or not 0 <= momentum < 1:
            raise ValueError("Invalid hyperparameters")
        if solver not in ("exact", "first_order"):
            raise ValueError("Unknown TCD solver")
        super().__init__(
            params,
            {
                "lr": lr,
                "rho": rho,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "solver": solver,
            },
        )
        self._samples = None

    def set_samples(self, samples):
        if self._samples is not None:
            raise RuntimeError("A sample set is already pending")
        params = [p for group in self.param_groups for p in group["params"]]
        if set(params) != set(samples):
            raise ValueError("Provide samples for every optimized matrix exactly once")
        for p in params:
            sample = samples[p]
            if p.ndim != 2 or sample.shape != (4, *p.shape):
                raise ValueError("Expected four matrix gradients per parameter")
            if sample.dtype != p.dtype or sample.device != p.device:
                raise ValueError("Sample dtype/device must match the parameter")
            if not sample.isfinite().all():
                raise FloatingPointError("Non-finite gradient samples")
        self._samples = {p: samples[p].detach().clone() for p in params}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None or self._samples is None:
            raise RuntimeError("Call set_samples before step; closures are unsupported")
        # Finish all solves before mutating any parameters.
        directions = {}
        for group in self.param_groups:
            for p in group["params"]:
                directions[p] = crossfit_direction(
                    self._samples[p], group["rho"], solver=group["solver"]
                )
                if not directions[p].isfinite().all():
                    raise FloatingPointError("Non-finite TCD direction")
        for group in self.param_groups:
            for p in group["params"]:
                direction = directions[p]
                if group["momentum"]:
                    state = self.state[p]
                    state.setdefault("momentum", torch.zeros_like(p))
                    state["step"] = state.get("step", 0) + 1
                    state["momentum"].lerp_(direction, 1 - group["momentum"])
                    direction = state["momentum"] / (
                        1 - group["momentum"] ** state["step"]
                    )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(direction, alpha=-group["lr"])
        self._samples = None

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._samples = None
