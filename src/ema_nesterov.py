"""EMA-Nesterov (Yau et al., arXiv 2605.25395), as in modded-nanogpt Track 3 record #39.

The model's weights are a lookahead point x + β·m, where x is the iterate and m an EMA (rate GAMMA)
of x's change per step; the usual update is taken from that point, for every parameter. Changes
that persist much longer than 1/(1 − GAMMA) = 100 steps move up to 1/(1 − β) times as far, while
faster ones (back and forth across a valley) keep their usual size. β = β_max·η_t/η_peak inside
WINDOW (fractions of the run: Track 3's steps 500 to 2,400 of 3,250), 0 outside it, so the end of
the decay is the plain optimizer. Track 3 used β_max 0.6 (record #39) and 0.3 (#40). Card S4-A
in note/optimizer-research/03-candidates.md.
"""

import os

import torch

# The EMA's rate; another with sbatch --export=...,EMA_NESTEROV_GAMMA=0.997.
GAMMA = float(os.environ.get("EMA_NESTEROV_GAMMA", "0.99"))
WINDOW = (0.15, 0.74)


def lookahead(beta_max, progress, lr_ratio):
    """β for the next step; progress: the share of the run done, lr_ratio: η_t/η_peak."""
    return beta_max * lr_ratio if WINDOW[0] < progress <= WINDOW[1] else 0.0


def ema_nesterov_step(p, step, state, beta_prev, beta):
    """p has just taken this iteration's step (its update and weight decay); move it on to the
    next lookahead point. beta_prev: the β p was moved by last time; beta: the next one."""
    if "lookahead" not in state:
        state["lookahead"] = torch.zeros_like(p)
    m = state["lookahead"]
    # The iterate x moved by step plus the last lookahead, beta_prev·m.
    m.mul_(GAMMA + (1 - GAMMA) * beta_prev).add_(step, alpha=1 - GAMMA)
    p.add_(m, alpha=beta)
