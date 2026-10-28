"""Explicit matrix-optimizer research variants, consuming global SUM gradients.

These algorithms are hypotheses, not claims of lower perplexity or new methods.
The immutable upstream implementation remains the Muon reference. This module
only replaces hidden-matrix updates; embeddings and auxiliary AdamW stay outside.

Primary sources:
* AdamW: https://arxiv.org/abs/1711.05101
* Shampoo (two inverse fourth powers): https://arxiv.org/abs/1802.09568
* Blocking/grafting: https://arxiv.org/abs/2002.09018 and
  https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo
* Matrix iterations: https://eprints.maths.manchester.ac.uk/694/1/high86p.pdf
* SoftMuon (the same regularized-polar family): https://arxiv.org/abs/2605.31371

``shampoo`` below is a declared blocked, EMA, bias-corrected, Adam-grafted
variant, not a reproduction of every choice in the original Shampoo paper.
``shampoo_adaptive`` estimates temporal uncertainty of each factor's trace-free
part using scalar weighted Welford statistics. Independence and stationarity
justify its noise-energy estimate, not an unconditional risk or PPL guarantee;
deterministic drift also increases this statistic. Its alpha cap defaults to .1.
``soft_polar`` is a SoftMuon-family control with Frobenius-relative temperature.
It uses s/sqrt(s*s+tau*tau), not s/(s+tau): the former admits the bounded
polynomial iteration implemented here without a training-time SVD.
"""

import copy
import math
from collections import defaultdict

import torch
import torch.distributed as dist


_ADAM_DEFAULTS = {"beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8}
_SHAMPOO_DEFAULTS = {
    **_ADAM_DEFAULTS,
    "precondition_epsilon": 1e-6,
    "block_size": 128,
    "update_frequency": 10,
    "shrinkage": 0.0,
    "graft": "adam",
}
RESEARCH_METHODS = {
    "matrix_adamw": {
        "description": "Bias-corrected AdamW on the existing hidden matrices",
        "defaults": _ADAM_DEFAULTS,
    },
    "shampoo": {
        "description": "Blocked EMA Shampoo with optional tensorwise Adam grafting",
        "defaults": _SHAMPOO_DEFAULTS,
    },
    "shampoo_shrink": {
        "description": "Same Shampoo with trace-preserving shrinkage at inversion",
        "defaults": {**_SHAMPOO_DEFAULTS, "shrinkage": 0.1},
    },
    "shampoo_adaptive": {
        "description": "Shampoo with capped trace-free temporal-uncertainty shrinkage",
        "defaults": {**_SHAMPOO_DEFAULTS, "alpha_max": 0.1},
    },
    "soft_polar": {
        "description": "Frobenius-relative regularized polar matrix update",
        "defaults": {
            "mu": 0.95,
            "tau": 0.1,
            "iterations": 12,
            "match_update_norm": False,
        },
    },
}


def validate_research_config(method, optimizer_kwargs=None):
    """Return all effective settings; reject unknown keys and ignored settings."""
    if method not in RESEARCH_METHODS:
        raise ValueError(f"Unknown matrix research method: {method}")
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if not isinstance(optimizer_kwargs, dict):
        raise ValueError("optimizer_kwargs must be a dictionary")  # noqa: TRY004 - config errors share ValueError
    defaults = RESEARCH_METHODS[method]["defaults"]
    unknown = set(optimizer_kwargs) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {method} settings: {sorted(unknown)}")
    settings = defaults | optimizer_kwargs
    for key, value in settings.items():
        if key in ("graft", "match_update_norm"):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite number")  # noqa: TRY004 - config errors share ValueError
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    for key in ("beta1", "beta2", "mu"):
        if key in settings and not 0 <= settings[key] < 1:
            raise ValueError(f"{key} must be in [0, 1)")
    for key in ("epsilon", "precondition_epsilon"):
        if key in settings and settings[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("block_size", "update_frequency", "iterations"):
        if key in settings and (
            type(settings[key]) is not int or settings[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    if method.startswith("shampoo"):
        if not 0 <= settings["shrinkage"] <= 1:
            raise ValueError("shrinkage must be in [0, 1]")
        if method in ("shampoo", "shampoo_adaptive") and settings["shrinkage"] != 0:
            raise ValueError("Only shampoo_shrink accepts fixed nonzero shrinkage")
        if method == "shampoo_adaptive" and not 0 <= settings["alpha_max"] <= 1:
            raise ValueError("alpha_max must be in [0, 1]")
        if settings["graft"] not in ("none", "adam"):
            raise ValueError("graft must be 'none' or 'adam'")
    if method == "soft_polar":
        # This ridge limits the condition number to at most 10001. The reference
        # supports tau=0, but finite-precision production must not approach an
        # unregularized inverse of a rank-deficient Gram matrix.
        if not 0.01 <= settings["tau"] <= 1:
            raise ValueError("Production tau must be in [0.01, 1]")
        if not 1 <= settings["iterations"] <= 20:
            raise ValueError("iterations must be in [1, 20]")
        if type(settings["match_update_norm"]) is not bool:
            raise ValueError("match_update_norm must be a boolean")
        # At the smallest possible Gram eigenvalue, track the scalar
        # sqrt(lambda)*Z gain. This is a rigorous exact-arithmetic relative
        # approximation bound, not an empirical finite-precision guarantee.
        tau = settings["tau"]
        gain = tau / math.sqrt(1 + tau * tau)
        for _ in range(settings["iterations"]):
            gain *= 1.5 - 0.5 * gain * gain
        if 1 - gain > 1e-3:
            raise ValueError(
                "iterations insufficient for tau: increase iterations until "
                "the worst-case inverse-root relative error is <= 0.001"
            )
    return settings


def _work_dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _matrix(tensor):
    if tensor.ndim != 2 or not tensor.is_floating_point() or tensor.is_complex():
        raise ValueError("Expected a real floating-point matrix")
    if min(tensor.shape) == 0:
        raise ValueError("Matrix dimensions must be nonzero")
    return tensor.to(dtype=_work_dtype(tensor))


def _relative_matrix(matrix):
    """Frobenius normalization without overflow for finite FP32 SUM gradients."""
    matrix = _matrix(matrix)
    scale = matrix.abs().amax().clamp_min(torch.finfo(matrix.dtype).tiny)
    relative = matrix / scale
    return relative / relative.norm().clamp_min(torch.finfo(matrix.dtype).tiny)


def soft_polar_reference(matrix, tau=0.1):
    """Exact small-matrix SVD reference, NEVER called by the optimizer.

    For X=M/||M||F, return U diag(s/sqrt(s**2+tau**2)) V.T.
    At tau=0 use the rank-aware polar pseudofactor (zero on zero singular values).
    """
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative")
    x = _relative_matrix(matrix)
    u, singular, vh = torch.linalg.svd(x, full_matrices=False)
    if tau == 0:
        cutoff = max(x.shape) * torch.finfo(x.dtype).eps * singular.max()
        filtered = (singular > cutoff).to(x.dtype)
    else:
        filtered = singular / (singular.square() + tau * tau).sqrt()
    return (u * filtered.unsqueeze(0)) @ vh


def _soft_polar_kernel(matrix, tau, iterations):
    """Pure tensor kernel, separated from validation for CUDA compilation."""
    x = _relative_matrix(matrix)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    identity = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
    normalization = 1 + tau * tau
    y = (x @ x.mT + tau * tau * identity) / normalization
    z = identity
    for _ in range(iterations):
        correction = 0.5 * (3 * identity - z @ y)
        y = y @ correction
        z = correction @ z
    result = (z @ x) / math.sqrt(normalization)
    return result.mT if transposed else result


_compiled_soft_polar_kernel = torch.compile(_soft_polar_kernel, fullgraph=True)


def soft_polar_direction(matrix, tau=0.1, iterations=12):
    """SVD-free bounded approximation to the regularized polar direction.

    A=(XX.T+tau**2 I)/(1+tau**2), Y0=A, Z0=I;
    Tk=(3I-ZkYk)/2, Y{k+1}=YkTk, Z{k+1}=TkZk.
    Return Zk X / sqrt(1+tau**2), transposing tall inputs to use the smaller
    Gram matrix. The eigenvalues of A lie in (0,1]. In exact arithmetic each
    inverse-root iterate underestimates A**(-1/2), so each resulting singular
    value is between zero and the exact filter's value (at most one).
    Configuration requires a worst-case exact-arithmetic relative inverse-root
    error <= 0.001; low tau therefore needs more iterations than the default.
    Finite-precision accuracy is tested separately; iteration count is explicit.
    CUDA uses a compiled pure tensor kernel; CPU uses the same eager operations.
    """
    validate_research_config("soft_polar", {"tau": tau, "iterations": iterations})
    kernel = _compiled_soft_polar_kernel if matrix.is_cuda else _soft_polar_kernel
    return kernel(matrix, tau, iterations)


def _square_factor(factor):
    if (
        factor.ndim < 2
        or factor.shape[-2] != factor.shape[-1]
        or factor.shape[-1] == 0
        or not factor.is_floating_point()
    ):
        raise ValueError("Expected a real square factor or batch of square factors")
    return factor.to(dtype=_work_dtype(factor))


def _trace_shrink(factor, shrinkage, validate_values):
    factor = _square_factor(factor)
    if isinstance(shrinkage, torch.Tensor):
        if not shrinkage.is_floating_point() or shrinkage.shape not in (
            torch.Size([]), factor.shape[:-2]
        ):
            raise ValueError("Tensor shrinkage must be scalar or match the factor batch")
        shrinkage = shrinkage.to(device=factor.device, dtype=factor.dtype)
        # Public tensor inputs are checked. The internal adaptive path already
        # produces bounded coefficients and avoids synchronizing GPU values.
        if validate_values and not (
            torch.isfinite(shrinkage) & (shrinkage >= 0) & (shrinkage <= 1)
        ).all():
            raise ValueError("Tensor shrinkage must be finite and in [0, 1]")
        shrinkage = shrinkage[..., None, None]
    elif (
        isinstance(shrinkage, bool)
        or not isinstance(shrinkage, (int, float))
        or not math.isfinite(shrinkage)
        or not 0 <= shrinkage <= 1
    ):
        raise ValueError("shrinkage must be finite and in [0, 1]")
    symmetric = 0.5 * (factor + factor.mT)
    mean_eigenvalue = symmetric.diagonal(dim1=-2, dim2=-1).mean(-1)
    identity = torch.eye(factor.shape[-1], dtype=factor.dtype, device=factor.device)
    return (1 - shrinkage) * symmetric + shrinkage * mean_eigenvalue[..., None, None] * identity


def trace_shrink(factor, shrinkage):
    """(1-rho) C + rho trace(C)/dim I, without changing stored statistics.

    ``shrinkage`` can be a number, a scalar tensor, or a tensor with the same
    leading batch dimensions as ``factor``. Public tensor values are validated.
    """
    return _trace_shrink(factor, shrinkage, validate_values=True)


def inverse_fourth_root(
    factor, shrinkage=0.0, epsilon=1e-6, *, validate_shrinkage=True
):
    """Inverse fourth root after shrinkage and an explicit relative ridge.

    D=(1-rho) C + rho trace(C)/d I + eps max(trace(C)/d,1e-12) I.
    Eigensolver roundoff below the same positive ridge is clamped. Stored C is
    never regularized in place. Both left and right Shampoo factors use -1/4.
    The adaptive optimizer passes validate_shrinkage=False only for coefficients
    constructed by its bounded estimator, avoiding a GPU-to-host synchronization.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    regularized = _trace_shrink(factor, shrinkage, validate_values=validate_shrinkage)
    diagonal = regularized.diagonal(dim1=-2, dim2=-1)
    ridge = epsilon * diagonal.mean(-1).clamp_min(1e-12)
    diagonal.add_(ridge.unsqueeze(-1))
    values, vectors = torch.linalg.eigh(regularized)
    values = values.clamp_min(ridge.unsqueeze(-1)).pow(-0.25)
    return (vectors * values.unsqueeze(-2)) @ vectors.mT


def ema_weight_statistics(beta2, step):
    """Normalized EMA newest weight w and squared-weight sum q, in float64.

    Weights are a_i=(1-beta2)*beta2**(step-i)/(1-beta2**step). Their squared
    sum is q=(1-beta2)/(1+beta2)*(1+beta2**step)/(1-beta2**step).
    expm1 avoids cancellation near beta2=1. At one effective observation q=1.
    """
    if (
        isinstance(beta2, bool)
        or not isinstance(beta2, (int, float))
        or not math.isfinite(beta2)
        or not 0 <= beta2 < 1
        or type(step) is not int
        or step < 1
    ):
        raise ValueError("Expected beta2 in [0, 1) and a positive integer step")
    if step == 1 or beta2 == 0:
        return 1.0, 1.0
    normalizer = -math.expm1(step * math.log(beta2))
    newest_weight = (1 - beta2) / normalizer
    squared_weight_sum = (1 - beta2) / (1 + beta2) * (2 - normalizer) / normalizer
    return newest_weight, min(1.0, squared_weight_sum)


def _tracefree_squared_norm(matrix):
    # Explicit centering avoids cancellation in ||C||^2 - trace(C)^2/d when
    # anisotropy is weak but reliable. Only a temporary matrix is allocated.
    centered = matrix.clone()
    diagonal = centered.diagonal(dim1=-2, dim2=-1)
    diagonal.sub_(diagonal.mean(-1, keepdim=True))
    return centered.square().sum(dim=(-2, -1))


def ema_anisotropy_scatter(
    sample_factor, previous_mean, previous_scatter, beta2, step
):
    """Update scalar weighted Welford scatter for trace-free factor samples.

    Inputs have shape (..., d, d); previous_scatter has shape (...). The previous
    mean is the bias-corrected stored factor at step-1. Return
    V_t=(1-w_t)*(V_{t-1}+w_t*||B_sample-B_previous_mean||_F^2), where B removes
    the isotropic trace component. The first step returns zero scatter.
    No sample history or additional persistent matrix is required.
    """
    sample_factor = _square_factor(sample_factor)
    previous_mean = _square_factor(previous_mean)
    if (
        sample_factor.shape != previous_mean.shape
        or previous_scatter.shape != sample_factor.shape[:-2]
        or sample_factor.device != previous_mean.device
        or sample_factor.device != previous_scatter.device
    ):
        raise ValueError("Incompatible factor mean or scalar scatter shape/device")
    weight, _ = ema_weight_statistics(beta2, step)
    if step == 1 or beta2 == 0:
        return sample_factor.new_zeros(sample_factor.shape[:-2])
    innovation_energy = _tracefree_squared_norm(sample_factor - previous_mean)
    return (1 - weight) * (
        previous_scatter.to(dtype=sample_factor.dtype) + weight * innovation_energy
    )


def ema_shrinkage_alpha(mean_factor, scatter, beta2, step, alpha_max=0.1):
    """Capped plug-in shrinkage of the mean factor's trace-free component.

    N=q*V/(1-q) estimates the mean's temporal uncertainty under stationary,
    independent Gram observations with finite second moment. Return
    clip(N/||tracefree(mean_factor)||_F^2, 0, alpha_max). This sample-dependent
    ratio has no general risk-domination guarantee. At step1 or beta2=0 return0.
    If both numerator and anisotropy vanish return0; positive uncertainty with
    zero anisotropy returns the cap (shrinking an isotropic factor is a no-op).
    """
    mean_factor = _square_factor(mean_factor)
    if (
        scatter.shape != mean_factor.shape[:-2]
        or scatter.device != mean_factor.device
        or isinstance(alpha_max, bool)
        or not isinstance(alpha_max, (int, float))
        or not math.isfinite(alpha_max)
        or not 0 <= alpha_max <= 1
    ):
        raise ValueError("Expected matching scalar scatter and alpha_max in [0, 1]")
    _, squared_weight_sum = ema_weight_statistics(beta2, step)
    if squared_weight_sum == 1 or alpha_max == 0:
        return mean_factor.new_zeros(mean_factor.shape[:-2])
    noise_energy = scatter.to(dtype=mean_factor.dtype).clamp_min(0) * (
        squared_weight_sum / (1 - squared_weight_sum)
    )
    anisotropy_energy = _tracefree_squared_norm(mean_factor)
    denominator = torch.where(
        anisotropy_energy > 0, anisotropy_energy, torch.ones_like(anisotropy_energy)
    )
    ratio = noise_energy / denominator
    zero_signal_alpha = torch.where(noise_energy > 0, alpha_max, 0.0)
    return torch.where(anisotropy_energy > 0, ratio, zero_signal_alpha).clamp(
        min=0, max=alpha_max
    )


def _block_regions(matrix, block_size):
    """At most four rectangles with homogeneous tile shapes, including edges."""
    def split(size):
        full = size // block_size * block_size
        regions = [(slice(0, full), block_size)] if full else []
        if full < size:
            regions.append((slice(full, size), size - full))
        return regions

    for row, tile_rows in split(matrix.shape[0]):
        for col, tile_cols in split(matrix.shape[1]):
            yield (row, col), tile_rows, tile_cols


def _pack_region(matrix, index, tile_rows, tile_cols):
    region = matrix[index]
    rows, cols = region.shape
    return (
        region.reshape(rows // tile_rows, tile_rows, cols // tile_cols, tile_cols)
        .permute(0, 2, 1, 3)
        .reshape(-1, tile_rows, tile_cols)
    )


def _unpack_region(output, index, tiles):
    rows, cols = output[index].shape
    tile_rows, tile_cols = tiles.shape[-2:]
    output[index] = (
        tiles.reshape(rows // tile_rows, cols // tile_cols, tile_rows, tile_cols)
        .permute(0, 2, 1, 3)
        .reshape(rows, cols)
    )


def _adam_direction(gradient, state, group):
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(gradient)
        state["exp_avg_sq"] = torch.zeros_like(gradient)
    beta1, beta2 = group["beta1"], group["beta2"]
    state["exp_avg"].lerp_(gradient, 1 - beta1)
    state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    mean = state["exp_avg"] / (1 - beta1 ** state["step"])
    variance = state["exp_avg_sq"] / (1 - beta2 ** state["step"])
    return mean, mean / (variance.sqrt() + group["epsilon"])


def _shampoo_direction(gradient, state, group):
    mean, adam = _adam_direction(gradient, state, group)
    regions = list(_block_regions(gradient, group["block_size"]))
    if "blocks" not in state:
        state["blocks"] = []
        for index, rows, cols in regions:
            count = gradient[index].numel() // (rows * cols)
            state["blocks"].append(
                {
                    "left": gradient.new_zeros(count, rows, rows),
                    "right": gradient.new_zeros(count, cols, cols),
                }
            )
    direction = torch.empty_like(gradient)
    beta2, step = group["beta2"], state["step"]
    refresh = step == 1 or (step - 1) % group["update_frequency"] == 0
    adaptive = group["method"] == "shampoo_adaptive"
    for (index, rows, cols), block in zip(regions, state["blocks"], strict=True):
        sample = _pack_region(gradient, index, rows, cols)
        for side, statistic in (
            ("left", sample @ sample.mT),
            ("right", sample.mT @ sample),
        ):
            if adaptive:
                previous_mean = (
                    block[side] / (1 - beta2 ** (step - 1))
                    if step > 1 else block[side]
                )
                scatter_key = f"{side}_scatter"
                if scatter_key not in block:
                    block[scatter_key] = gradient.new_zeros(statistic.shape[:-2])
                block[scatter_key] = ema_anisotropy_scatter(
                    statistic, previous_mean, block[scatter_key], beta2, step
                )
            # This arithmetic is shared verbatim with the unshrunk parent;
            # neither adaptive shrinkage nor its diagnostics modify raw factors.
            block[side].mul_(beta2).add_(statistic, alpha=1 - beta2)
        if refresh:
            if adaptive:
                block["last_refresh_step"] = step
            for side in ("left", "right"):
                factor_mean = block[side] / (1 - beta2**step)
                shrinkage = group["shrinkage"]
                if adaptive:
                    shrinkage = ema_shrinkage_alpha(
                        factor_mean,
                        block[f"{side}_scatter"],
                        beta2,
                        step,
                        group["alpha_max"],
                    )
                    block[f"{side}_alpha"] = shrinkage
                block[f"{side}_root"] = inverse_fourth_root(
                    factor_mean,
                    shrinkage,
                    group["precondition_epsilon"],
                    validate_shrinkage=not adaptive,
                )
        tiles = block["left_root"] @ _pack_region(mean, index, rows, cols) @ block["right_root"]
        _unpack_region(direction, index, tiles)
    if group["graft"] == "adam":
        direction.mul_(adam.norm() / direction.norm().clamp_min(1e-30))
    return direction


class MatrixResearchOptimizer(torch.optim.Optimizer):
    """Owner-sharded matrix updates with homogeneous-shape parameter all-gather.

    The caller MUST SUM-reduce each gradient before step; this optimizer neither
    reduces gradients nor divides by tokens/world size. Owner state is persisted
    on each rank, so resumption requires all rank checkpoints and the same world
    size/rank mapping. Single-process use needs no distributed process group.
    Parameters may be FP32, BF16, or FP64; state/linear algebra are FP32 except
    for FP64 reference parameters. Checkpoint loading preserves that precision.
    """

    def __init__(self, params, method, lr=None, weight_decay=0.0, **kwargs):
        settings = validate_research_config(method, kwargs)
        if lr is None:
            lr = 0.025 if method == "soft_polar" else 0.001
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in (lr, weight_decay)
        ):
            raise ValueError("lr and weight_decay must be finite and nonnegative")
        params = list(params)
        if not params or any(
            p.ndim != 2
            or min(p.shape) == 0
            or p.dtype not in (torch.float32, torch.bfloat16, torch.float64)
            for p in params
        ):
            raise ValueError("Expected nonempty FP32, BF16, or FP64 matrix parameters")
        if len({id(p) for p in params}) != len(params):
            raise ValueError("Duplicate parameters")
        if len({p.device for p in params}) != 1:
            raise ValueError("Matrix parameters must share one device")
        params = sorted(params, key=lambda p: tuple(p.shape), reverse=True)
        super().__init__(
            params,
            {"method": method, "lr": lr, "weight_decay": weight_decay, **settings},
        )
        self.matrix_parameters = set(params)
        self._topology = self._distributed_topology()

    @staticmethod
    def _distributed_topology():
        if dist.is_available() and dist.is_initialized():
            return {"world_size": dist.get_world_size(), "rank": dist.get_rank()}
        return {"world_size": 1, "rank": 0}

    def state_dict(self):
        result = super().state_dict()
        result["research_topology"] = dict(self._topology)
        return result

    def load_state_dict(self, state_dict):
        copied = copy.deepcopy(state_dict)
        if copied.pop("research_topology", None) != self._topology:
            raise ValueError("Matrix state requires the same saved world size and rank")
        if len(copied.get("param_groups", [])) != len(self.param_groups):
            raise ValueError("Incompatible optimizer groups")
        for saved, current in zip(copied["param_groups"], self.param_groups, strict=True):
            if saved.get("method") != current["method"]:
                raise ValueError("Cannot restore state for a different optimizer method")
            defaults = RESEARCH_METHODS[current["method"]]["defaults"]
            validate_research_config(
                current["method"], {key: saved[key] for key in defaults}
            )
            if (
                current["method"].startswith("shampoo")
                and saved["block_size"] != current["block_size"]
            ):
                raise ValueError("Checkpoint block_size must match the constructor")
        # torch Optimizer casts floating state to parameter dtype. Preserve the
        # original checkpoint tensors, rather than upcasting BF16-rounded state.
        raw_states = copied["state"]
        saved_groups = copied["param_groups"]
        super().load_state_dict(copied)

        def move(value, parameter):
            if isinstance(value, torch.Tensor):
                return value.to(device=parameter.device, dtype=_work_dtype(parameter))
            if isinstance(value, dict):
                return {key: move(item, parameter) for key, item in value.items()}
            if isinstance(value, list):
                return [move(item, parameter) for item in value]
            return value

        for saved, current in zip(saved_groups, self.param_groups, strict=True):
            for key, parameter in zip(saved["params"], current["params"], strict=True):
                if key in raw_states:
                    self.state[parameter] = move(raw_states[key], parameter)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Supply already SUM-reduced gradients; closures unsupported")
        if self._distributed_topology() != self._topology:
            raise RuntimeError("Initialize the process group before creating the optimizer")
        world_size, rank = self._topology["world_size"], self._topology["rank"]
        for group in self.param_groups:
            buckets = defaultdict(list)
            for parameter in group["params"]:
                if parameter.grad is None or parameter.grad.is_sparse:
                    raise ValueError("Every matrix needs a dense globally SUM gradient")
                buckets[(tuple(parameter.shape), parameter.dtype)].append(parameter)
            for bucket in buckets.values():
                for base in range(0, len(bucket), world_size):
                    chunk = bucket[base : base + world_size]
                    if rank < len(chunk):
                        parameter = chunk[rank]
                        gradient = parameter.grad.to(dtype=_work_dtype(parameter))
                        state = self.state[parameter]
                        state["step"] = state.get("step", 0) + 1
                        if group["method"] == "soft_polar":
                            if "momentum" not in state:
                                state["momentum"] = torch.zeros_like(gradient)
                            mu = group["mu"]
                            state["momentum"].lerp_(gradient, 1 - mu)
                            nesterov = gradient.lerp(state["momentum"], mu)
                            update = soft_polar_direction(
                                nesterov, group["tau"], group["iterations"]
                            )
                            rows, cols = gradient.shape
                            update.mul_(math.sqrt(max(1, rows / cols)))
                            if group["match_update_norm"]:
                                update.mul_(math.sqrt(rows) / update.norm().clamp_min(1e-30))
                        elif group["method"].startswith("shampoo"):
                            update = _shampoo_direction(gradient, state, group)
                        else:
                            _, update = _adam_direction(gradient, state, group)
                        parameter.mul_(1 - group["lr"] * group["weight_decay"])
                        parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
                    if world_size > 1:
                        # Same-shape collectives work on Gloo as well as NCCL.
                        padded = chunk + [
                            torch.empty_like(chunk[0]) for _ in range(world_size - len(chunk))
                        ]
                        dist.all_gather(padded, padded[rank])


def make_matrix_optimizer(params, method, lr=None, weight_decay=0.0, **kwargs):
    """Factory used by the checked upstream AST integration."""
    return MatrixResearchOptimizer(params, method, lr, weight_decay, **kwargs)
