"""Diagnostics for the optimizer research (note/optimizer-research/03-candidates.md).

Probe is a callback for a 150M baseline run. It logs every block matrix's norm, saves extra
checkpoints 20 and 100 steps after the 20/50/80% saves (the path-coherence probe needs pairs),
and keeps two EMAs of the weights over the last 10% of the run, read out at the end.
"""

import json
from contextlib import contextmanager

import torch
from transformers import TrainerCallback

# The weight EMAs' windows, as fractions of the run (tail averaging, card S4-B).
EMA_WINDOWS = (0.01, 0.04)


class Probe(TrainerCallback):
    def __init__(self, model, steps, path):
        self.model, self.steps, self.path = model, steps, path
        self.params = dict(model.named_parameters())
        self.matrices = {
            n: p for n, p in self.params.items() if p.ndim == 2 and ".layers." in n
        }
        every = max(1, steps // 10)
        self.extra_saves = {k * every + d for k in (2, 5, 8) for d in (20, 100)}
        self.emas = {}

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step % 10 == 0:
            norms = torch.stack([p.norm() for p in self.matrices.values()]).tolist()
            with open(self.path, "a") as stream:
                stream.write(
                    json.dumps({"step": step, "norms": dict(zip(self.matrices, norms))})
                )
                stream.write("\n")
        if step in self.extra_saves:
            control.should_save = True
        if step >= 0.9 * self.steps:
            for window in EMA_WINDOWS:
                ema = self.emas.setdefault(
                    window, {n: p.detach().clone() for n, p in self.params.items()}
                )
                for n, p in self.params.items():
                    ema[n].lerp_(p, 1 / (window * self.steps))

    @contextmanager
    def ema_weights(self, window):
        """The EMA's weights in the model; each matrix scaled back to the current weights' norm,
        since averaging rotating noise shrinks it (modded-nanogpt's "decontraction")."""
        with torch.no_grad():
            backup = {n: p.detach().clone() for n, p in self.params.items()}
            for n, p in self.params.items():
                ema = self.emas[window][n]
                p.copy_(ema * (p.norm() / ema.norm()) if p.ndim == 2 else ema)
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.params.items():
                    p.copy_(backup[n])
