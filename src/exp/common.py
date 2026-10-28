"""Pure experiment configuration, schedules, and atomic records."""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT: dict[str, Any] = {
    "method": "normuon",
    "matrix_lr": 1.2e-3,
    "embed_lr": 6.8e-3,
    "matrix_decay": 1.5,
    "embed_decay": 1.5,
    "momentum": 0.95,
    "matrix_beta1": 0.9,
    "beta2": 0.9873,
    "warmup": 0.01,
    "decay": 0.8,
    "meta_lr": 0.01,
    "seed": 42,
    "max_tokens": 300_000_000,
    "global_batch": 16,
    "micro_batch": 4,
    "phase": "search",
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def schedule(steps, warmup, decay):
    """LR actually used at optimizer step i; final step is nonzero, then zero.

    Starts at 1/warmup_steps, avoiding a wasted zero-LR optimizer step. The
    returned values are used both by LambdaLR and by exact decay calibration.
    """
    if steps < 1 or not (0 <= warmup < 1 and 0 < decay <= 1 and warmup + decay <= 1):
        raise ValueError("Invalid schedule")
    nw = min(steps, math.ceil(steps * warmup))
    nd = min(steps - nw, max(1, math.ceil(steps * decay)))
    values = []
    for i in range(steps):
        if i < nw:
            value = (i + 1) / nw
        elif nd and i >= steps - nd:
            value = (steps - i) / nd
        else:
            value = 1.0
        values.append(value)
    return values, nw


def calibrated_decay(peak_lr, multipliers, strength):
    if strength < 0 or peak_lr <= 0:
        raise ValueError("Invalid decay calibration")
    if strength == 0:
        return 0.0
    rates = peak_lr * np.asarray(multipliers, dtype=np.float64)
    lo, hi = 0.0, (1 - 1e-12) / rates.max()
    for _ in range(80):
        mid = (lo + hi) / 2
        value = -np.log1p(-rates * mid).sum()
        if value < strength:
            lo = mid
        else:
            hi = mid
    # Keep optimizer checkpoints compatible with torch.load(weights_only=True).
    return float((lo + hi) / 2)


def model_config(base=None, tiny=False):
    from transformers import LlamaConfig

    cfg = base if base is not None else LlamaConfig()
    fields = {
        "vocab_size": 128256,
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 22,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "max_position_embeddings": 8192,
        "tie_word_embeddings": True,
        "use_cache": False,
    }
    if tiny:
        fields.update(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    for key, value in fields.items():
        setattr(cfg, key, value)
    return cfg
