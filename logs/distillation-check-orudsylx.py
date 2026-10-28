import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/u9603854/lab5-training-llama/src/distillation")
import generate_logits as cache

cache.MAX_TOKENS = 1_000_000
cache.OUTPUT_DIR = cache.WORK_DIR / "data/distillation" / ("smoke-" + os.environ["SLURM_JOB_ID"])
started = time.perf_counter()
cache.main()
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
paths = sorted(cache.OUTPUT_DIR.glob("*.pt"))
assert paths, "No cache files were written"
n_tokens = 0
next_row = 0
for path in paths:
    saved = torch.load(path, mmap=True, weights_only=True)
    metadata = saved["metadata"]
    assert metadata["row_start"] == next_row
    next_row = metadata["row_stop"]
    offsets = saved["row_offsets"]
    tokens = int(offsets[-1])
    assert offsets[0] == 0
    assert torch.all(offsets[1:] > offsets[:-1])
    assert saved["topk_ids"].shape == (tokens, cache.TOP_K)
    assert saved["topk_logits"].shape == (tokens, cache.TOP_K)
    assert saved["logsumexp"].shape == (tokens,)
    assert torch.all((saved["topk_ids"] >= 0) & (saved["topk_ids"] < 128256))
    assert torch.isfinite(saved["topk_logits"]).all()
    assert torch.isfinite(saved["logsumexp"]).all()
    mass = (saved["topk_logits"].float() - saved["logsumexp"][:, None]).exp().sum(-1)
    assert torch.all(mass <= 1.0001), ("invalid probability mass", float(mass.max()))
    assert torch.all(mass > 0)
    print("shard verified:", path.name, tokens, "tokens; mean Top-32 mass:", float(mass.mean()))
    n_tokens += tokens
print("SMOKE_OK", "rows=", next_row, "tokens=", n_tokens, "seconds=", round(elapsed, 2),
      "peak_GPU_GB=", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
