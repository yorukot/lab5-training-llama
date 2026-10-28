"""Sophia (Liu et al., arXiv 2305.14342; SophiaG in github.com/Liuhong99/Sophia).

Like Adam, each entry steps by the running mean of its gradient, but divides it by an estimate of
the loss's curvature in that entry (the Hessian's diagonal) instead of the gradient's size: flat
directions take bigger steps, sharp ones smaller. Each entry's step is clipped to at most 1 (times
lr), which guards against tiny or stale curvature estimates.

The curvature comes from the Gauss-Newton-Bartlett estimator: every HESSIAN_INTERVAL steps,
training.py samples labels from the model's own predictions (sample_labels), backpropagates the
loss on them, and calls update_hessian, which keeps an EMA of tokens * grad**2.

Changes from the official code, for our short runs (763 estimates in 1B tokens) at a small batch:
- The curvature EMA is bias-corrected, as Adam's is; without it the curvature is underestimated
  for the first ~1,000 steps, which clips nearly every entry.
- Its beta2 is 0.95, not 0.99: Wen et al. 2025 (arXiv 2509.02046) found 0.9-0.95 best and 0.99
  divergent at 520M.
- The curvature pass's gradient is not clipped (Levanter and Semenov et al. 2025 don't either):
  once the gradient grows, clipping it shrinks the curvature, which makes the steps bigger.
The weight decay and the lr step are applied in optimizer.py.
"""

import torch

BETA1, BETA2 = 0.965, 0.95
RHO = 0.04  # Entries with |m| > RHO * curvature are clipped; 10-50% should stay unclipped.
HESSIAN_INTERVAL = 10


def sophia_update(grad, state):
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(grad)
    state["exp_avg"].lerp_(grad, 1 - BETA1)
    hessian = state["hessian"] / (1 - BETA2 ** state["hessian_updates"])
    ratio = (state["exp_avg"].abs() / (RHO * hessian + 1e-15)).clamp(max=1)
    return state["exp_avg"].sign() * ratio


def update_hessian(grad, state, tokens):
    """grad: of the mean loss on sampled labels over `tokens` tokens."""
    if "hessian" not in state:
        state["hessian"] = torch.zeros_like(grad)
        state["hessian_updates"] = 0
    state["hessian"].lerp_(tokens * grad.square(), 1 - BETA2)
    state["hessian_updates"] += 1


@torch.no_grad()
def sample_labels(hidden, lm_head_weight, labels):
    """Labels drawn from the model's own predictions (hidden: its last hidden states), wherever
    the training loss has a label; elsewhere -100."""
    sampled = torch.empty_like(labels)
    flat_hidden, flat_sampled = hidden.flatten(0, 1), sampled.view(-1)
    weight = lm_head_weight.float()
    for i in range(0, len(flat_hidden), 1024):  # 1024 tokens of fp32 logits at a time
        logits = flat_hidden[i : i + 1024].float() @ weight.T
        # Gumbel-max: argmax(logits + Gumbel noise) is a sample from softmax(logits).
        noise = torch.empty_like(logits).exponential_().log()
        flat_sampled[i : i + 1024] = (logits - noise).argmax(-1)
    # Position t predicts the label at t + 1, as in the training loss.
    sampled = torch.cat([torch.full_like(labels[:, :1], -100), sampled[:, :-1]], 1)
    return sampled.masked_fill(labels == -100, -100)
