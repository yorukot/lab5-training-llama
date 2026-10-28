"""Muon (Keller Jordan, kellerjordan.github.io/posts/muon), for the 2D weights.

Take the Nesterov momentum of a matrix's gradient and replace it with the nearest orthogonal
matrix (every singular value 1, found with a few Newton-Schulz steps), so the update moves every
direction by the same amount instead of being dominated by a few. The result is scaled as in
Moonlight (arXiv 2502.16982) to AdamW's update RMS of 0.2, so the two share lr and weight decay.
The weight decay and the lr step are applied in optimizer.py.
"""

import os

import torch

from head_damping import damp_head

# Nesterov momentum; a run can set another with sbatch --export=...,MUON_MOMENTUM=0.98.
MOMENTUM = float(os.environ.get("MUON_MOMENTUM", "0.95"))
# NorMuon's EMA of each row's mean squared update; another with NORMUON_BETA2=0.9873.
NORMUON_BETA2 = float(os.environ.get("NORMUON_BETA2", "0.95"))
JORDAN_COEFFS = [(3.4445, -4.7750, 2.0315)] * 5
# Polar Express (arXiv 2505.16932): a different quintic per step, so the singular values land
# closer to 1. The first 5 of optimal_composition(l=1e-3, num_iters=10, safety_factor_eps=1e-2,
# cushion=0.02) from github.com/NoahAmsel/PolarExpress, safety factor already applied.
POLAR_EXPRESS_COEFFS = [
    (8.237312490495555, -23.157747414558187, 16.6805684114459),
    (4.082441999064831, -2.8930477353325843, 0.5252849256975642),
    (3.926347992254648, -2.854746803476524, 0.531802242289498),
    (3.298218713308519, -2.42454198102671, 0.48632008358844137),
    (2.297036943455259, -1.636625581259032, 0.4002628455953629),
]


def zeropower_via_newtonschulz5(G, polar_express=False):
    coeffs = POLAR_EXPRESS_COEFFS if polar_express else JORDAN_COEFFS
    margin = 1.01 if polar_express else 1.0
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * margin + 1e-7)
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def normuon_equalize(update, row_sq, beta2=NORMUON_BETA2):
    """NorMuon (arXiv 2510.05491, github.com/zichongli5/NorMuon): divide each output row by the
    root of an EMA of its mean squared update, then restore the matrix's norm (same step size).
    A row_sq of shape [1, n] does the same per input column instead."""
    norm = update.norm()
    dim = -1 if row_sq.size(-1) == 1 else -2
    row_sq.lerp_(update.square().mean(dim, keepdim=True), 1 - beta2)
    update = update / (row_sq.sqrt() + 1e-10)
    return update * (norm / (update.norm() + 1e-10))


def muon_update(grad, state, variant, momentum=None, progress=0.0):
    """variant: "muon"; "polar_express" (Polar Express's coefficients only); "normuon" (NorMuon
    plus Polar Express, the pair modded-nanogpt adopted together); two NorMuon candidates from
    note/optimizer-research/03-candidates.md: "normuon_long" (a wide matrix's rows are already
    equal after the polar step, so equalize its columns instead) and "normuon_damped" (see
    head_damping.py). progress: the share of the run done."""
    normuon = variant.startswith("normuon")
    if not state:
        state["momentum_buffer"] = torch.zeros_like(grad)
        if normuon:
            columns = variant == "normuon_long" and grad.size(-2) < grad.size(-1)
            state["row_sq"] = torch.zeros_like(
                grad[..., :1, :] if columns else grad[..., :1]
            )
    momentum = MOMENTUM if momentum is None else momentum
    buf = state["momentum_buffer"]
    buf.lerp_(grad, 1 - momentum)
    nesterov = grad.lerp(buf, momentum)
    update = zeropower_via_newtonschulz5(nesterov, polar_express=variant != "muon")
    if normuon:
        update = normuon_equalize(update.float(), state["row_sq"])
    if variant == "normuon_damped":
        update = damp_head(update, nesterov, state, progress)
    return update * (0.2 * max(grad.shape) ** 0.5)
