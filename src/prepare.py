import bisect
import hashlib
import os
from itertools import chain

import numpy as np
from datasets import Features, List, Value, load_dataset
from transformers import PreTrainedTokenizerFast

MODEL_ID = "meta-llama/Llama-3.2-1B"
SEQ_LEN = 8192
# Absolute, so saving never needs the working directory: a /home hiccup once made
# os.getcwd() fail inside save_to_disk after an hour of tokenizing.
OUT_DIR = os.path.realpath("data/bestfit")
TRAIN_DOCS = 7_000_000  # candidates; ~7B tokens are left after dedup and truncation
EVAL_DOCS = 10_000  # 20% of the 50k held-out docs
PACK_DOCS = 10_000  # documents best-fit together
NUM_PROC = 12  # = --cpus-per-task in script/prepare_data.sbatch
FEATURES = Features(
    {
        "input_ids": List(Value("int32")),
        "doc_lens": List(Value("int16")),
        "n_tokens": Value("int32"),
    }
)

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)


def doc_hash(text):
    """Hash of the text with case and whitespace normalized, for exact deduplication."""
    text = " ".join(text.lower().split()).encode()
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), signed=True)


def hashes(split):
    hashed = split.map(
        lambda batch: {"hash": [doc_hash(text) for text in batch["text"]]},
        batched=True,
        remove_columns=split.column_names,
        num_proc=NUM_PROC,
        desc="hash",
    )
    return np.array(hashed["hash"], dtype=np.int64)


def pack(docs):
    """Best-fit docs, longest first, into rows of at most SEQ_LEN tokens; returns each row's docs."""
    rows = []
    free = []  # sorted (free space, row index)
    for doc in sorted(docs, key=len, reverse=True):
        # The row with the least space that fits.
        i = bisect.bisect_left(free, (len(doc), -1))
        if i == len(free):
            rows.append([])
            space, row = SEQ_LEN, len(rows) - 1
        else:
            space, row = free.pop(i)
        rows[row].append(doc)
        bisect.insort(free, (space - len(doc), row))
    return rows


def tokenize_and_pack(batch):
    """Whole documents only: no document is split across rows, and rows are not padded here.

    Tokenized like the OJ: BOS + text, truncated to SEQ_LEN, no EOS. A document of only the
    BOS has nothing to score, so it is dropped.
    """
    docs = tokenizer(batch["text"], truncation=True, max_length=SEQ_LEN)["input_ids"]
    rows = pack([doc for doc in docs if len(doc) > 1])
    return {
        "input_ids": [list(chain.from_iterable(row)) for row in rows],
        "doc_lens": [[len(doc) for doc in row] for row in rows],
        "n_tokens": [sum(len(doc) for doc in row) for row in rows],
    }


# Split on documents first (shuffle seed 42, last 50k held out), then tokenize.
dataset = load_dataset("allenai/dolma3_mix-150B-1025", split="train", num_proc=NUM_PROC)
print(f"total docs: {len(dataset):,}")

# Same permutation as dataset.shuffle(seed=42). Each split's indices are sorted
# so rows are read from disk in order instead of randomly (much faster).
order = np.random.default_rng(42).permutation(len(dataset))
held_out = order[-50_000:]

# The dataset repeats documents: keep the first copy of each text in train, and drop any
# copy of a held-out document, case and whitespace aside.
candidates = dataset.select(np.sort(order[:TRAIN_DOCS]))
train_hashes = hashes(candidates)
first = np.sort(np.unique(train_hashes, return_index=True)[1])
keep = first[~np.isin(train_hashes[first], hashes(dataset.select(np.sort(held_out))))]
print(
    f"train docs: {len(keep):,} of {TRAIN_DOCS:,} "
    f"({TRAIN_DOCS - len(first):,} duplicates, {len(first) - len(keep):,} held-out copies)"
)
splits = {
    "train": candidates.select(keep),
    "eval": dataset.select(np.sort(held_out[:EVAL_DOCS])),
}
for name, split in splits.items():
    packed = split.map(
        tokenize_and_pack,
        batched=True,
        batch_size=PACK_DOCS,
        remove_columns=split.column_names,
        features=FEATURES,
        # Every worker first maps the whole 845 GB dataset (~20 min), so the 10k eval
        # docs are much faster in this process.
        num_proc=NUM_PROC if name == "train" else None,
        desc=f"tokenize and pack {name}",
    )
    packed.save_to_disk(f"{OUT_DIR}/{name}")
    tokens = sum(packed["n_tokens"])
    truncated = sum(lens.count(SEQ_LEN) for lens in packed["doc_lens"])
    print(
        f"{name}: {len(packed):,} rows, {tokens:,} tokens "
        f"({tokens / (len(packed) * SEQ_LEN):.2%} filled), {truncated:,} docs cut at {SEQ_LEN}"
    )
