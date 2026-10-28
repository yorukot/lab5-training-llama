"""Head damping (SAMuon-lite, arXiv 2608.25990), on top of NorMuon.

Muon gives every singular direction of the momentum the same step. The top one (the "head") is
about the sharpest direction and already at the edge of stability, so its step is divided by
GAMMA and the rest keep theirs. The head comes from POWER_STEPS power iterations warm-started from
the last step's, a few matrix-vector products. GAMMA is reached by a cosine ramp over the first
RAMP of the run (the paper's spectral warmup). Card S5-A in note/optimizer-research/03-candidates.md.
"""

import math

import torch

GAMMA = 5.0
RAMP = 0.3
POWER_STEPS = 2


def damp_head(update, momentum, state, progress):
    """update: the NorMuon update of the Nesterov input momentum; progress: the share of the run
    done."""
    if "head" not in state:
        # The same start on every rank, so the replicas stay identical.
        state["head"] = torch.ones_like(momentum[0]) / momentum.size(-1) ** 0.5
    v = state["head"]
    for _ in range(POWER_STEPS):
        v = momentum.mT @ (momentum @ v)
        v = v / (v.norm() + 1e-30)
    state["head"] = v
    u = momentum @ v
    u = u / (u.norm() + 1e-30)
    gamma = 1 + (GAMMA - 1) * (1 - math.cos(math.pi * min(progress / RAMP, 1))) / 2
    return update - (1 - 1 / gamma) * (u @ update @ v) * torch.outer(u, v)
