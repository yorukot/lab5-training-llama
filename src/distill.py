"""Distillation from cached teacher predictions (src/distillation/generate_logits.py).

At each position the target mixes the teacher with the data: the teacher's top-128 tokens get
KD_ALPHA times their teacher probability, and the rest (1 - KD_ALPHA, plus the teacher's mass
beyond its top 128) goes to the actual next token. The loss is the cross-entropy from that target
to the student, so KD_ALPHA = 0 is the usual loss. The data's share matters: the top 128 hold
96.5% of the teacher's mass on average, and the actual next token is outside them 3.7% of the time.
With unbiased_tail, the actual next token instead stands in for the teacher's mass beyond its
top 128, as a sample of it: it gets weight 1 when it is outside them and 1 - KD_ALPHA when it is
inside. The target then sums to 1 only on average, and the gradient is sum(target) * softmax
minus the target.
"""

import mmap
from pathlib import Path

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

# Llama 3.1 8B's top-128 for the rows of the 6B data, in the order training visits them.
CACHE_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/distillation/llama31-8b-top128"
TOP_K = 128
CHUNK = 2048  # tokens whose logits [CHUNK, vocab] exist at once


class Teacher:
    """The cached top-128 ids and probabilities of each training row, by dataset row."""

    def __init__(self):
        self.shards = []
        self.where = {}  # dataset row -> (shard, row in the shard)
        # Shared maps: a private map counts the whole cache (183 GB at 1B tokens) toward the
        # kernel's commit limit in every rank and DataLoader worker, so fork() fails. Shared maps
        # write through to the files, but nothing here writes to them.
        with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
            for path in sorted(Path(CACHE_DIR).glob("shard-*.pt")):
                shard = torch.load(
                    path, map_location="cpu", mmap=True, weights_only=True
                )
                for i, row in enumerate(shard["rows"].tolist()):
                    self.where[row] = (len(self.shards), i)
                self.shards.append(shard)

    def targets(self, rows, seq_len):
        """The teacher's prediction at each position of each row (of the token after it)."""
        ids = torch.zeros(len(rows), seq_len, TOP_K, dtype=torch.long)
        probs = torch.zeros(len(rows), seq_len, TOP_K)
        for b, row in enumerate(rows):
            if row not in self.where:
                raise KeyError(f"Row {row} is not in the teacher cache {CACHE_DIR}")
            index, i = self.where[row]
            shard = self.shards[index]
            start, stop = shard["row_offsets"][i : i + 2].tolist()
            logits = shard["topk_logits"][start:stop].float()
            lse = shard["logsumexp"][start:stop, None]
            ids[b, : stop - start] = shard["topk_ids"][start:stop]
            probs[b, : stop - start] = (logits - lse).exp()
        return {"kd_ids": ids, "kd_probs": probs}


class MixedCrossEntropy(torch.autograd.Function):
    """The summed cross-entropy from the mixed target to softmax(hidden @ weight.T). As in Liger's
    fused linear cross-entropy, the logits exist one chunk at a time and the gradients are
    computed in the forward pass."""

    @staticmethod
    def forward(ctx, hidden, weight, labels, kd_ids, kd_probs, alpha, unbiased_tail):
        on = torch.is_autocast_enabled(hidden.device.type)
        dtype = torch.get_autocast_dtype(hidden.device.type) if on else hidden.dtype
        h, w = hidden.to(dtype), weight.to(dtype)
        grad_h = torch.empty_like(h)
        grad_w = torch.zeros_like(weight, dtype=torch.float32)
        loss = torch.zeros((), device=hidden.device)
        for s in range(0, len(h), CHUNK):
            hc, label = h[s : s + CHUNK], labels[s : s + CHUNK, None]
            ids, teacher = kd_ids[s : s + CHUNK], alpha * kd_probs[s : s + CHUNK]
            if unbiased_tail:
                data = torch.where((ids != label).all(1, keepdim=True), 1.0, 1 - alpha)
            else:
                data = 1 - teacher.sum(1, keepdim=True)
            logits = (hc @ w.T).float()
            lse = logits.logsumexp(1, keepdim=True)
            # sum(target * log softmax) with the target nonzero only at ids and the label
            log_p = logits.sub_(lse)
            loss -= (teacher * log_p.gather(1, ids)).sum()
            loss -= (data * log_p.gather(1, label)).sum()
            grad = log_p.exp_()  # sum(target) * softmax - target
            if unbiased_tail:
                grad.mul_(teacher.sum(1, keepdim=True) + data)
            grad.scatter_add_(1, ids, -teacher).scatter_add_(1, label, -data)
            grad = grad.to(dtype)
            grad_h[s : s + CHUNK] = grad @ w
            grad_w += (grad.T @ hc).float()
        ctx.save_for_backward(grad_h.to(hidden.dtype), grad_w.to(weight.dtype))
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        grad_h, grad_w = ctx.saved_tensors
        return grad_h * grad_loss, grad_w * grad_loss, None, None, None, None, None


def use_mixed_loss(model, alpha, unbiased_tail=False):
    """Make model.forward return the mixed loss for batches that carry the teacher's targets
    (training); other batches (eval) keep the usual forward."""
    base = model.forward

    def forward(labels=None, kd_ids=None, kd_probs=None, **inputs):
        if kd_ids is None:
            return base(labels=labels, **inputs)
        # The Trainer passes the batch's label count, as Liger's loss uses it.
        count = inputs.pop("num_items_in_batch", None)
        hidden = model.model(**inputs).last_hidden_state[:, :-1]
        labels = labels[:, 1:]  # position t predicts token t + 1
        keep = labels != -100
        loss = MixedCrossEntropy.apply(
            hidden[keep],
            model.lm_head.weight,
            labels[keep],
            kd_ids[:, :-1][keep],
            kd_probs[:, :-1][keep],
            alpha,
            unbiased_tail,
        )
        return CausalLMOutputWithPast(loss=loss / (count or keep.sum()))

    model.forward = forward
