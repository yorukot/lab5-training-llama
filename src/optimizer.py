"""One optimizer for the whole model. Each param group names its update rule ("muon", "adamw" or
"sophia", each in its own file); the weight decay and the lr step are shared, and so is the
optional EMA-Nesterov lookahead (ema_nesterov.py)."""

import torch

import sophia
from adamw import adamw_update
from ema_nesterov import ema_nesterov_step, lookahead
from muon import muon_update

# OPTIMIZER -> the update rules for (the block matrices, the tied embedding and the norms).
ALGORITHMS = {
    "muon": ("muon", "adamw"),
    "adamw": ("adamw", "adamw"),
    "sophia+adamw": ("sophia", "adamw"),
    "sophia": ("sophia", "sophia"),
}


class LabOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        param_groups,
        adam_beta2,
        muon_variant,
        cautious_wd,
        total_steps,
        ema_nesterov,
    ):
        super().__init__(param_groups, {})
        self.adam_beta2 = adam_beta2
        self.muon_variant = muon_variant
        self.cautious_wd = cautious_wd
        self.total_steps = total_steps
        self.ema_nesterov = ema_nesterov
        # Sophia's share of entries left unclipped in the last step, for the log.
        self.win_rate = float("nan")

    @torch.no_grad()
    def step(self, closure=None):
        wins = entries = 0
        for group in self.param_groups:
            lr, weight_decay = group["lr"], group["weight_decay"]
            # Steps done, counted in the group so that a checkpoint keeps it.
            group["step"] = group.get("step", 0) + 1
            progress = group["step"] / self.total_steps
            beta_prev = group.get("lookahead", 0.0)
            if self.ema_nesterov:
                group["lookahead"] = lookahead(
                    self.ema_nesterov, progress, lr / group.get("initial_lr", lr)
                )
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["algorithm"] == "muon":
                    update = muon_update(
                        p.grad, state, self.muon_variant, progress=progress
                    )
                elif group["algorithm"] == "adamw":
                    update = adamw_update(p.grad, state, self.adam_beta2)
                else:
                    update = sophia.sophia_update(p.grad, state)
                    wins += (update.abs() < 1).sum()
                    entries += update.numel()
                if self.ema_nesterov:
                    # This iteration's change of p (update and weight decay), which
                    # EMA-Nesterov averages; p itself steps as usual below.
                    step = -lr * (update.to(p.dtype) + weight_decay * p)
                if self.cautious_wd:
                    # Cautious weight decay (arXiv 2510.12402): decay only the entries whose
                    # update has their sign, so the decay never pulls against the update.
                    p.sub_(p * (update * p >= 0), alpha=lr * weight_decay)
                else:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update.to(p.dtype), alpha=-lr)
                if self.ema_nesterov:
                    ema_nesterov_step(p, step, state, beta_prev, group["lookahead"])
        if entries:
            self.win_rate = wins / entries

    @torch.no_grad()
    def update_hessian(self, tokens):
        for group in self.param_groups:
            if group["algorithm"] == "sophia":
                for p in group["params"]:
                    if p.grad is not None:
                        sophia.update_hessian(p.grad, self.state[p], tokens)


def build_optimizer(
    model,
    name,
    lr,
    weight_decay,
    embed_lr,
    embed_wd,
    norm_lr,
    adam_beta2,
    muon_variant,
    cautious_wd,
    total_steps,
    ema_nesterov,
):
    """The 2D weights inside the transformer blocks get lr and weight_decay (in two groups,
    "attn" and "mlp"); the (tied) embedding embed_lr and embed_wd; the norms norm_lr and no
    weight decay. name: a key of ALGORITHMS; muon_variant: see muon.muon_update; ema_nesterov:
    the largest lookahead β of EMA-Nesterov around every update (ema_nesterov.py), 0 for none."""
    matrix_algorithm, embed_algorithm = ALGORITHMS[name]
    attn, mlp, embedding, norms = [], [], [], []
    for param_name, p in model.named_parameters():
        if p.ndim == 2 and ".layers." in param_name:
            (mlp if ".mlp." in param_name else attn).append(p)
        elif p.ndim == 2:
            embedding.append(p)
        else:
            norms.append(p)
    groups = [
        {
            "name": block,
            "params": params,
            "algorithm": matrix_algorithm,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        for block, params in (("attn", attn), ("mlp", mlp))
    ] + [
        {
            "name": "embedding",
            "params": embedding,
            "algorithm": embed_algorithm,
            "lr": embed_lr,
            "weight_decay": embed_wd,
        },
        {
            "name": "norms",
            "params": norms,
            "algorithm": embed_algorithm,
            "lr": norm_lr,
            "weight_decay": 0.0,
        },
    ]
    return LabOptimizer(
        groups, adam_beta2, muon_variant, cautious_wd, total_steps, ema_nesterov
    )
