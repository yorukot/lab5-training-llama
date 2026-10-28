"""AdamW (Loshchilov & Hutter, arXiv 1711.05101).

Each entry steps by the running mean of its gradient divided by the root of the running mean of
its squared gradient, so every entry moves about the same amount whatever its gradient's scale.
The weight decay and the lr step are applied in optimizer.py.
"""

import torch

BETA1 = 0.9
EPS = 1e-8


def adamw_update(grad, state, beta2, beta1=BETA1):
    if not state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(grad)
        state["exp_avg_sq"] = torch.zeros_like(grad)
    state["step"] += 1
    state["exp_avg"].lerp_(grad, 1 - beta1)
    state["exp_avg_sq"].lerp_(grad.square(), 1 - beta2)
    m_hat = state["exp_avg"] / (1 - beta1 ** state["step"])
    v_hat = state["exp_avg_sq"] / (1 - beta2 ** state["step"])
    return m_hat / (v_hat.sqrt() + EPS)
