"""CP 的權重更新：矩陣用 NorMuon，embedding / norm 用 AdamW，外面加 EMA。

lr 和 weight decay 分開設定：矩陣 0.0012 / 0.0729，embedding 0.0068 / 0.0129。
norm 的 lr 也是 0.0068，但沒有 weight decay。
以 45,759 步估算，1 / (最高 lr * wd * 步數) 約為 0.25。
"""

import math

import settings as s
import torch

# Polar Express：5 次矩陣多項式，近似把奇異值拉到 1 附近。
POLAR_COEFFICIENTS = [
    (8.237312490495555, -23.157747414558187, 16.6805684114459),
    (4.082441999064831, -2.8930477353325843, 0.5252849256975642),
    (3.926347992254648, -2.854746803476524, 0.531802242289498),
    (3.298218713308519, -2.42454198102671, 0.48632008358844137),
    (2.297036943455259, -1.636625581259032, 0.4002628455953629),
]


def lr_ratio(completed_steps, total_steps):
    """前 1% 線性暖身 → 維持最高 lr → 最後 80% 線性降到 0。"""
    warmup = math.ceil(s.WARMUP_FRACTION * total_steps)
    decay = int(s.DECAY_FRACTION * total_steps)
    if completed_steps < warmup:
        return completed_steps / warmup
    if completed_steps < total_steps - decay:
        return 1.0
    return max(0.0, (total_steps - completed_steps) / decay)


def normuon_update(grad, state):
    if not state:
        state["momentum"] = torch.zeros_like(grad)
        state["row_square"] = torch.zeros_like(grad[:, :1])

    # 先用快速 Nesterov momentum 決定方向。
    momentum = state["momentum"]
    momentum.lerp_(grad, 1 - s.MUON_MOMENTUM)
    direction = grad.lerp(momentum, s.MUON_MOMENTUM)

    # Muon：讓更新不要被少數很大的奇異值主導。
    x = direction.bfloat16()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.norm() * 1.01 + 1e-7)
    for a, b, c in POLAR_COEFFICIENTS:
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transpose:
        x = x.T
    update = x.float()

    # NorMuon：每個輸出 row 再除以自己的歷史 RMS，然後還原矩陣總長度。
    old_norm = update.norm()
    row_square = state["row_square"]
    row_square.lerp_(update.square().mean(-1, keepdim=True), 1 - s.NORMUON_BETA2)
    update = update / (row_square.sqrt() + 1e-10)
    update = update * (old_norm / (update.norm() + 1e-10))
    return update * (0.2 * max(grad.shape) ** 0.5)


def adamw_update(grad, state):
    if not state:
        state["step"] = 0
        state["mean"] = torch.zeros_like(grad)
        state["square"] = torch.zeros_like(grad)
    beta1, beta2 = s.ADAM_BETAS
    state["step"] += 1
    state["mean"].lerp_(grad, 1 - beta1)
    state["square"].lerp_(grad.square(), 1 - beta2)
    mean = state["mean"] / (1 - beta1 ** state["step"])
    square = state["square"] / (1 - beta2 ** state["step"])
    return mean / (square.sqrt() + 1e-8)


class SOTAOptimizer:
    def __init__(self, model):
        self.parameters = []
        self.previous_beta = 0.0
        for name, p in model.named_parameters():
            # tied embedding / lm_head 是同一份參數，named_parameters 只列一次。
            is_matrix = p.ndim == 2 and ".layers." in name
            lr = s.MUON_LR if is_matrix else s.ADAM_LR
            wd = s.MUON_WD if is_matrix else s.EMBED_WD if p.ndim == 2 else 0.0
            self.parameters.append((p, is_matrix, lr, wd, {}))

    @torch.no_grad()
    def step(self, step, total_steps):
        # step 從 1 開始；與原本 scheduler 一樣，第一次更新的 lr 是 0。
        ratio = lr_ratio(step - 1, total_steps)
        progress = step / total_steps
        beta = s.EMA_BETA * ratio if s.EMA_START < progress <= s.EMA_END else 0.0

        for p, is_matrix, peak_lr, wd, state in self.parameters:
            if p.grad is None:
                continue
            update = (
                normuon_update(p.grad, state)
                if is_matrix
                else adamw_update(p.grad, state)
            )
            lr = peak_lr * ratio
            displacement = -lr * (update.to(p.dtype) + wd * p)

            # 一般更新：先縮小權重，再沿更新方向走一步。
            p.mul_(1 - lr * wd)
            p.add_(update.to(p.dtype), alpha=-lr)

            # EMA-Nesterov：平均最近約 100 步的位移，再朝這個方向多走一點。
            # 這不是「最後把模型換成平均權重」；它在訓練時直接影響下次 forward。
            # previous_beta 那一項也要留下，因為上一步的 lookahead 已包含在 p 裡。
            if "slow_momentum" not in state:
                state["slow_momentum"] = torch.zeros_like(p)
            slow = state["slow_momentum"]
            slow.mul_(s.EMA_GAMMA + (1 - s.EMA_GAMMA) * self.previous_beta)
            slow.add_(displacement, alpha=1 - s.EMA_GAMMA)
            p.add_(slow, alpha=beta)

        self.previous_beta = beta
