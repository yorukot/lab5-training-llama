"""Experimental cross-fitted TCD1 integration; no claim of optimizer superiority.

Eight rank-local SUM gradients are paired into four equal-token MEAN gradients.
Only matrix optimization and explicitly declared search budgets are changed.
The immutable upstream architecture, data order and auxiliary AdamW are retained.
"""

import ast
import math

import torch
import torch.distributed as dist

from exp.covariance_revision.covariance import covariance_direction
from exp.tcd import precondition_first_order


def fold_direction(samples, old_momentum, rho=3.0, mu=0.95):
    """Other-fold geometry AND scale; historical momentum is shared past state.

    samples has shape (4, rows, cols), each a mean over 131072 target tokens.
    The returned direction has Muon's nominal 1/sqrt(fan_in) entry scale.
    This scaling is an experimental integration choice, not part of old TCD.
    """
    g1, g2, g3, g4 = samples.unbind(0)
    a, b = (g1 + g2) / 2, (g3 + g4) / 2
    # Matches upstream EMA-Nesterov convention before preconditioning.
    na = mu * mu * old_momentum + (1 - mu * mu) * a
    nb = mu * mu * old_momentum + (1 - mu * mu) * b
    if rho:
        da, _ = precondition_first_order(nb, (g1 - g2) / 2, a, rho)
        db, _ = precondition_first_order(na, (g3 - g4) / 2, b, rho)
    else:
        da, db = nb, na
    # Each numerator is scaled by the opposite (geometry) fold. Current-fold
    # self-normalization would invalidate even the conditional mean identity.
    da = da / na.square().mean().sqrt().clamp_min(1e-8)
    db = db / nb.square().mean().sqrt().clamp_min(1e-8)
    direction = (da + db) * (0.5 / math.sqrt(samples.shape[-1]))
    next_momentum = mu * old_momentum + (1 - mu) * (a + b) / 2
    return direction, next_momentum


def matrix_update_eager(rank_sums, momentum, rho=3.0, mu=0.95, geometry="gram"):
    if geometry == "covariance":
        direction, next_momentum = covariance_direction(
            rank_sums / 65536, momentum, rho, mu
        )
    else:
        samples = rank_sums.reshape(4, 2, *rank_sums.shape[1:]).mean(1) / 65536
        direction, next_momentum = fold_direction(samples, momentum, rho, mu)
    momentum.copy_(next_momentum)
    return direction


matrix_update = torch.compile(matrix_update_eager)


class MatrixTCD(torch.optim.Optimizer):
    """Shard matrix optimizer work across eight ranks, preserving local gradients.

    Communication differs from Muon and is included in wall-clock cost. Each
    parameter owner gathers eight local gradients before the global reduction;
    parameters are then shared using the upstream round-robin all-gather layout.
    """

    def __init__(
        self, params, lr=0.025, weight_decay=0.05, rho=3.0, mu=0.95, geometry="gram"
    ):
        if geometry not in ("gram", "covariance"):
            raise ValueError("Unknown geometry")
        if not all(math.isfinite(x) for x in (lr, weight_decay, rho, mu)):
            raise ValueError("Nonfinite optimizer hyperparameters")
        if min(lr, weight_decay, rho) < 0 or not 0 <= mu < 1:
            raise ValueError("Invalid optimizer hyperparameters")
        params = sorted(params, key=lambda p: p.size(), reverse=True)
        if not params or any(p.ndim != 2 or p.dtype != torch.float32 for p in params):
            raise ValueError("Expected nonempty FP32 matrix parameters")
        super().__init__(
            params,
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "rho": rho,
                "mu": mu,
                "geometry": geometry,
            },
        )
        self.matrix_parameters = set(params)
        self._buffers = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None or dist.get_world_size() != 8:
            raise ValueError("This experimental collector requires eight ranks")
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            padded = params + [torch.empty_like(params[-1])] * (-len(params) % 8)
            for base in range(0, len(params), 8):
                owned_samples = None
                for owner, p in enumerate(params[base : base + 8]):
                    if p.grad is None:
                        raise ValueError("Missing matrix gradient")
                    if owner == rank:
                        shape = tuple(p.shape)
                        if shape not in self._buffers:
                            self._buffers[shape] = torch.empty(
                                (8, *shape), device=p.device, dtype=p.dtype
                            )
                        owned_samples = self._buffers[shape]
                        gather_list = list(owned_samples.unbind(0))
                    else:
                        gather_list = None
                    dist.gather(p.grad, gather_list=gather_list, dst=owner)
                if base + rank < len(params):
                    assert owned_samples is not None
                    p = params[base + rank]
                    state = self.state[p]
                    if not state:
                        state["momentum"] = torch.zeros_like(p)
                    update = matrix_update(
                        owned_samples,
                        state["momentum"],
                        group["rho"],
                        group["mu"],
                        group["geometry"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(padded[base : base + 8], padded[base + rank])


def reduce_gradient(parameter, optimizer):
    if parameter not in optimizer.matrix_parameters:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def experiment_source(source, config):
    """Apply a narrow, checked AST patch to the hash-verified upstream source."""
    required = {"method", "matrix_lr", "matrix_decay", "steps", "rho"}
    if set(config) != required or config["method"] not in (
        "muon",
        "tcd1",
        "fold_rms",
        "covariance",
    ):
        raise ValueError("Invalid experiment configuration")
    if (
        not isinstance(config["steps"], int)
        or not 1 <= config["steps"] <= 3250
        or not all(
            math.isfinite(config[k]) for k in ("matrix_lr", "matrix_decay", "rho")
        )
        or config["matrix_lr"] <= 0
        or min(config["matrix_decay"], config["rho"]) < 0
        or (config["method"] == "fold_rms" and config["rho"] != 0)
    ):
        raise ValueError("Invalid experiment budget or hyperparameters")
    tree = ast.parse(source)
    changed = {"steps": 0, "optimizer": 0, "reduction": 0}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "train_steps":
                node.value = ast.Constant(config["steps"])
                changed["steps"] += 1
            if isinstance(target, ast.Name) and target.id == "optimizer2":
                call = node.value
                if not isinstance(call, ast.Call) or ast.unparse(call.func) != "Muon":
                    raise ValueError("Upstream optimizer structure changed")
                for keyword in call.keywords:
                    if keyword.arg == "lr":
                        keyword.value = ast.Constant(config["matrix_lr"])
                    elif keyword.arg == "weight_decay":
                        keyword.value = ast.Constant(config["matrix_decay"])
                if config["method"] != "muon":
                    call.func = ast.Name("MatrixTCD", ast.Load())
                    call.keywords.append(
                        ast.keyword("rho", ast.Constant(config["rho"]))
                    )
                    if config["method"] == "covariance":
                        call.keywords.append(
                            ast.keyword("geometry", ast.Constant("covariance"))
                        )
                changed["optimizer"] += 1
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "dist.all_reduce"
            and len(node.value.args) == 1
            and ast.unparse(node.value.args[0]) == "p.grad"
        ):
            if config["method"] != "muon":
                node.value = ast.Call(
                    ast.Name("reduce_gradient", ast.Load()),
                    [ast.Name("p", ast.Load()), ast.Name("optimizer2", ast.Load())],
                    [],
                )
            changed["reduction"] += 1
    if changed != {"steps": 1, "optimizer": 1, "reduction": 1}:
        raise ValueError(f"Ambiguous upstream patch: {changed}")
    if config["method"] != "muon":
        tree.body.insert(
            1,
            ast.ImportFrom(
                "exp.covariance_revision.candidate",
                [ast.alias("MatrixTCD"), ast.alias("reduce_gradient")],
                0,
            ),
        )
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
