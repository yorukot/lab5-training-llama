"""Cache Llama 3.1 8B's Top-128 next-token logits for offline distillation.

Rows are cached in the order training.py's sampler visits them (training_order), so a run
over the first N tokens finds every row it trains on.

Edit the constants below, then run scripts/distillation.sbatch on nano4.
Each GPU loads its own teacher and writes different shards; no DDP is needed.
"""

import getpass
import json
import os
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import model_info
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

TEACHER_ID = "meta-llama/Llama-3.1-8B"
TOKENIZER_ID = "meta-llama/Llama-3.2-1B"  # The tokenizer used by prepare.py.
WORK_DIR = Path("/work") / getpass.getuser() / "lab5-training-llama"
DATA_DIR = WORK_DIR / "prepare-66ea6ae/data/bestfit/train"
OUTPUT_DIR = WORK_DIR / "prepare-66ea6ae/data/distillation/llama31-8b-top128"

TOP_K = 128
# Cache the rows training visits first, within this budget; None = all rows. Leave a margin:
# the DataLoader fetches one batch past a run's last step.
MAX_TOKENS = None
ROWS_PER_SHARD = 128  # ~800 MB per file at Top-128; supports torch.load(mmap=True).
BATCH_ROWS = 4  # Packed rows per GPU forward pass, each at most 8192 tokens.
# Only project this many hidden states to the vocabulary at once.
LOGIT_CHUNK_TOKENS = 512


def training_order(n_rows):
    """The row order of training.py's first epoch: accelerate's SeedableRandomSampler seeds
    torch.randperm with the Trainer seed 42 (checked against BU's tokens seen at steps 10-30)."""
    generator = torch.Generator()
    generator.manual_seed(42)
    return torch.randperm(n_rows, generator=generator).numpy()


def rows_within_budget(token_counts, max_tokens):
    """Keep whole rows, in their original order, without exceeding the token budget."""
    cumulative = np.cumsum(token_counts, dtype=np.int64)
    if max_tokens is None:
        return len(cumulative)
    return int(np.searchsorted(cumulative, max_tokens, side="right"))


def pack_rows(rows, device):
    """Flatten rows without padding. FA3's offsets keep attention inside each document."""
    for row in rows:
        if (
            not row["doc_lens"]
            or min(row["doc_lens"]) < 2
            or sum(row["doc_lens"]) != len(row["input_ids"])
            or row["n_tokens"] != len(row["input_ids"])
        ):
            raise ValueError("Expected whole documents from src/prepare.py.")
    doc_lens = list(chain.from_iterable(row["doc_lens"] for row in rows))
    input_ids = list(chain.from_iterable(row["input_ids"] for row in rows))
    positions = list(chain.from_iterable(range(length) for length in doc_lens))
    offsets = torch.tensor([0] + doc_lens, dtype=torch.int32, device=device)
    offsets = offsets.cumsum(0, dtype=torch.int32)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "position_ids": torch.tensor([positions], dtype=torch.long, device=device),
        "cu_seq_lens_q": offsets,
        "cu_seq_lens_k": offsets,
        "max_length_q": max(doc_lens),
        "max_length_k": max(doc_lens),
    }


@torch.inference_mode()
def generate_batch(teacher, rows):
    inputs = pack_rows(rows, teacher.device)
    hidden = teacher.model(**inputs, use_cache=False).last_hidden_state[0]
    ids, values, normalizers = [], [], []
    for chunk in hidden.split(LOGIT_CHUNK_TOKENS):
        logits = teacher.lm_head(chunk).float()
        top_values, top_ids = logits.topk(TOP_K, dim=-1)
        ids.append(top_ids.to(device="cpu", dtype=torch.int32))
        values.append(top_values.to(device="cpu", dtype=torch.float16))
        # Normalize over ALL vocabulary entries, not just the retained Top-K.
        normalizers.append(logits.logsumexp(dim=-1).cpu())
        del logits
    return {
        "topk_ids": torch.cat(ids),
        "topk_logits": torch.cat(values),
        "logsumexp": torch.cat(normalizers),
    }


def generate_shard(teacher, rows, row_ids):
    batches = [
        generate_batch(teacher, rows[start : start + BATCH_ROWS])
        for start in range(0, len(rows), BATCH_ROWS)
    ]
    result = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
    # Row i's cache is [row_offsets[i]:row_offsets[i + 1]], in original token order.
    result["row_offsets"] = torch.tensor(
        [0] + [row["n_tokens"] for row in rows]
    ).cumsum(0)
    # The dataset row of each cached row.
    result["rows"] = torch.tensor(row_ids, dtype=torch.int64)
    return result


def load_shard(path, metadata, row_start):
    """Reuse the saved prefix, even when the requested token budget has increased."""
    if not path.exists():
        return None
    cached = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    expected = {
        **metadata,
        "row_start": row_start,
        "row_stop": cached["metadata"]["row_stop"],
    }
    if cached["metadata"] != expected:
        raise ValueError(f"Cache settings differ in {path}; choose a new OUTPUT_DIR.")
    return cached


def save_manifest(metadata, n_rows, n_tokens):
    """Record the requested coverage; existing shards may cover more or fewer rows."""
    path = OUTPUT_DIR / "manifest.json"
    if path.exists() and json.loads(path.read_text())["metadata"] != metadata:
        raise ValueError(f"Cache settings differ in {path}; choose a new OUTPUT_DIR.")
    manifest = {
        "metadata": metadata,
        "requested_rows": n_rows,
        "requested_tokens": n_tokens,
    }
    # Workers write identical manifests. Unique temporary names avoid write races.
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def save_shard(path, shard):
    # A cancelled job leaves only a .tmp file; a .pt file is always fully written.
    temporary = path.with_suffix(".tmp")
    torch.save(shard, temporary)
    temporary.replace(path)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Run this on a CUDA GPU with Flash Attention 3 installed.")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    dataset = Dataset.load_from_disk(str(DATA_DIR))
    order = training_order(len(dataset))
    token_counts = np.asarray(dataset["n_tokens"], dtype=np.int64)[order]
    n_rows = rows_within_budget(token_counts, MAX_TOKENS)
    if n_rows == 0:
        raise ValueError("No complete rows fit MAX_TOKENS.")

    # A teacher's token IDs must mean the same thing as the existing student data.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_ID)
    revision = model_info(TEACHER_ID).sha
    if revision is None:
        raise RuntimeError("Could not resolve the teacher's Hugging Face revision.")
    teacher_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        TEACHER_ID, revision=revision
    )
    if (
        tokenizer.get_vocab() != teacher_tokenizer.get_vocab()
        or tokenizer.bos_token_id != teacher_tokenizer.bos_token_id
    ):
        raise ValueError("Teacher and student token IDs do not match.")
    metadata = {
        "format_version": 2,
        "teacher_id": TEACHER_ID,
        "teacher_revision": revision,
        "teacher_dtype": "bfloat16",
        "tokenizer_id": TOKENIZER_ID,
        "dataset_path": str(DATA_DIR.resolve()),
        "dataset_fingerprint": dataset._fingerprint,
        "rows_per_shard": ROWS_PER_SHARD,
        "row_order": "training_order: torch.randperm(rows, seed 42)",
        "top_k": TOP_K,
        "temperature": 1.0,
        "alignment": "position t predicts token t+1; ignore document-ending positions",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_tokens = int(token_counts[:n_rows].sum())
    save_manifest(metadata, n_rows, n_tokens)
    if rank == 0:
        print(f"Caching {n_rows:,} rows / {n_tokens:,} tokens with {world_size} GPU(s)")
        print(f"Output: {OUTPUT_DIR} (~{n_tokens * (TOP_K * 6 + 4) / 1e9:.1f} GB)")

    # A shard covers positions [row_start, row_stop) of the training order; "rows" holds their
    # dataset rows. GPU 0 takes shards 0, 8, 16, ...; GPU 1 takes 1, 9, 17, ... on 8 GPUs.
    teacher = None  # Load weights only if this worker has uncached rows.
    starts = range(rank * ROWS_PER_SHARD, n_rows, world_size * ROWS_PER_SHARD)
    for start in tqdm(starts, desc="Cache shards (GPU 0)", disable=rank != 0):
        stop = min(start + ROWS_PER_SHARD, n_rows)
        path = OUTPUT_DIR / f"shard-{start:07d}.pt"
        cached = load_shard(path, metadata, start)
        next_row = start if cached is None else cached["metadata"]["row_stop"]
        if next_row >= stop:
            continue
        if teacher is None:
            teacher = AutoModelForCausalLM.from_pretrained(
                TEACHER_ID,
                revision=revision,
                dtype=torch.bfloat16,
                device_map={"": local_rank},
                attn_implementation="flash_attention_3",
            ).eval()
            teacher.requires_grad_(False)
        row_ids = order[next_row:stop].tolist()
        shard = generate_shard(teacher, list(dataset.select(row_ids)), row_ids)
        if cached is not None:
            # Extend the old final shard without running its already cached rows again.
            for key in ("topk_ids", "topk_logits", "logsumexp", "rows"):
                shard[key] = torch.cat([cached[key], shard[key]])
            shard["row_offsets"] = torch.cat(
                [
                    cached["row_offsets"],
                    shard["row_offsets"][1:] + cached["row_offsets"][-1],
                ]
            )
        shard["metadata"] = {**metadata, "row_start": start, "row_stop": stop}
        save_shard(path, shard)
    print(f"GPU {rank}: finished")


if __name__ == "__main__":
    main()
