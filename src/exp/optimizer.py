"""Bounded one-step hypergradient adaptation of the existing NorMuon update."""

import math

import torch

from adamw import adamw_update
from muon import muon_update

METHODS = ("adamw", "normuon", "hg_normuon_global", "hg_normuon")


class ExperimentOptimizer(torch.optim.Optimizer):
    def __init__(self, groups, *, method="hg_normuon", meta_lr=0.01,
                 factor_min=0.5, factor_max=2.0, warmup_steps=0):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if not (math.isfinite(meta_lr) and meta_lr >= 0
                and 0 < factor_min <= 1 <= factor_max < math.inf):
            raise ValueError("Invalid meta learning rate or factor bounds")
        defaults = dict(lr=1e-3, weight_decay=0., beta1=0.9, beta2=0.9873,
                        momentum=0.95, method=method, meta_lr=meta_lr,
                        factor_min=factor_min, factor_max=factor_max,
                        warmup_steps=warmup_steps, completed_steps=0)
        super().__init__(groups, defaults)
        self._observed = False
        self._pending = {}
        self._last_h = None

    @property
    def matrices(self):
        return [p for g in self.param_groups if g["kind"] == "matrix" for p in g["params"]]

    @property
    def adaptive(self):
        return self.param_groups[0]["method"].startswith("hg_")

    def _meta(self, p, key="meta"):
        return self.state[p].setdefault(key, {
            "factor": torch.ones((), device=p.device, dtype=torch.float32),
            "sum_sq": torch.zeros((), device=p.device, dtype=torch.float32),
        })

    @torch.no_grad()
    def observe_unclipped_gradients(self):
        """Call once, AFTER accumulation/DDP synchronization and BEFORE clipping.

        This observes the derivative w.r.t. the PREVIOUS multiplier, treating its
        base update and all preceding history as constants. It is not a derivative
        through the full training trajectory. Missing gradients invalidate history.
        """
        if not self.adaptive:
            return
        if self._observed:
            raise RuntimeError("Observation already pending; call step before observing again")
        self._pending = {}
        for p in self.matrices:
            state = self.state[p]
            if p.grad is None:
                state.pop("previous_base_step", None)
                continue
            if p.grad.is_sparse:
                raise ValueError("Sparse gradients are unsupported")
            if "previous_base_step" in state:
                self._pending[p] = -(p.grad.float() * state["previous_base_step"]).sum()
        if self._pending:
            values = torch.stack(list(self._pending.values()))
            if not values.isfinite().all():
                raise FloatingPointError("Non-finite hypergradient")
            self._last_h = (values.mean(), values.square().mean().sqrt())
        self._observed = True

    @staticmethod
    def _adapt(meta, h, group):
        meta["sum_sq"].add_(h.square())
        meta["factor"].sub_(group["meta_lr"] * h / (meta["sum_sq"].sqrt() + 1e-12))
        meta["factor"].clamp_(group["factor_min"], group["factor_max"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            raise ValueError("Observe gradients explicitly; closures are unsupported")
        control = self.param_groups[0]
        if self.adaptive and not self._observed:
            raise RuntimeError("observe_unclipped_gradients() must precede each adaptive step")
        active = control["completed_steps"] >= control["warmup_steps"]
        global_meta = None
        if control["method"] == "hg_normuon_global" and self.matrices:
            global_meta = self._meta(self.matrices[0], "global_meta")
            if active and self._pending:
                self._adapt(global_meta, torch.stack(list(self._pending.values())).sum(), control)
        for group in self.param_groups:
            lr = group["lr"]
            group["last_lr"] = lr
            for p in group["params"]:
                state = self.state[p]
                if p.grad is None:
                    state.pop("previous_base_step", None)
                    continue
                if p.grad.is_sparse:
                    raise ValueError("Sparse gradients are unsupported")
                base = state.setdefault("base", {})
                matrix = group["kind"] == "matrix"
                if matrix and group["method"] != "adamw":
                    update = muon_update(
                        p.grad, base, "normuon", momentum=group["momentum"]
                    )
                else:
                    update = adamw_update(p.grad, base, group["beta2"], group["beta1"])
                factor = None
                if matrix and self.adaptive:
                    meta = global_meta if global_meta is not None else self._meta(p)
                    if global_meta is None and active and p in self._pending:
                        self._adapt(meta, self._pending[p], group)
                    factor = meta["factor"]
                    if lr:
                        state["previous_base_step"] = update.float().mul(lr)
                    else:
                        state.pop("previous_base_step", None)
                p.mul_(1 - lr * group["weight_decay"])
                # Multiplication by exactly one preserves the baseline update.
                if factor is not None:
                    update = update * factor
                p.add_(update.to(p.dtype), alpha=-lr)
            group["completed_steps"] += 1
        self._observed = False
        self._pending = {}
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._observed = False
        self._pending = {}
        self._last_h = None

    @torch.no_grad()
    def diagnostics(self):
        if not self.adaptive or not self.matrices:
            return {}
        key = "global_meta" if self.param_groups[0]["method"] == "hg_normuon_global" else "meta"
        params = self.matrices[:1] if key == "global_meta" else self.matrices
        factors = torch.stack([self._meta(p, key)["factor"] for p in params])
        g = self.param_groups[0]
        diagnostics = {
            "meta/factor_min": factors.min().item(),
            "meta/factor_median": factors.median().item(),
            "meta/factor_max": factors.max().item(),
            "meta/boundary_fraction": ((factors <= g["factor_min"]) | (factors >= g["factor_max"])).float().mean().item(),
            "meta/effective_lr_min": factors.min().item() * g.get("last_lr", g["lr"]),
            "meta/effective_lr_max": factors.max().item() * g.get("last_lr", g["lr"]),
        }
        if self._last_h is not None:
            diagnostics.update({"meta/hypergradient_mean": self._last_h[0].item(),
                                "meta/hypergradient_rms": self._last_h[1].item()})
        return diagnostics


def build_optimizer(model, config, matrix_wd, embed_wd, warmup_steps):
    buckets = {"matrix": [], "embedding": [], "norm": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = "matrix" if p.ndim == 2 and ".layers." in name else "embedding" if p.ndim == 2 else "norm"
        buckets[kind].append(p)
    groups = []
    for kind, params in buckets.items():
        matrix = kind == "matrix"
        groups.append(dict(params=params, kind=kind,
                           lr=config["matrix_lr"] if matrix else config["embed_lr"],
                           weight_decay=matrix_wd if matrix else embed_wd if kind == "embedding" else 0.,
                           momentum=config["momentum"], beta2=config["beta2"],
                           beta1=config["matrix_beta1"] if matrix else 0.9))
    return ExperimentOptimizer(groups, method=config["method"],
                               meta_lr=config["meta_lr"], warmup_steps=warmup_steps)
