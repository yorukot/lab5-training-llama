"""Reuse prepared Arrow datasets; never tokenize or download training data."""

import bisect
import fcntl
import hashlib
from itertools import chain
from pathlib import Path

import numpy as np
import torch

from exp.common import atomic_json, fingerprint


class PackedCollator:
    def __init__(self, pad_id, seq_len=8192, varlen=True):
        self.pad_id, self.seq_len, self.varlen = pad_id, seq_len, varlen

    def __call__(self, examples):
        ids = torch.full((len(examples), self.seq_len), self.pad_id, dtype=torch.long)
        positions = torch.zeros_like(ids)
        segments = torch.zeros_like(ids)
        real = torch.zeros_like(ids, dtype=torch.bool)
        all_lengths = []
        for row, example in enumerate(examples):
            tokens = torch.as_tensor(example["input_ids"], dtype=torch.long)
            lengths = list(map(int, example["doc_lens"]))
            n = len(tokens)
            if sum(lengths) != n or n > self.seq_len or not lengths or min(lengths) < 2:
                raise ValueError("Invalid packed document lengths")
            ids[row, :n], real[row, :n] = tokens, True
            if n < self.seq_len:
                lengths.append(self.seq_len - n)
            positions[row] = torch.cat([torch.arange(k) for k in lengths])
            segments[row] = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
            all_lengths.extend(lengths)
        batch = dict(input_ids=ids, labels=ids.masked_fill((positions == 0) | ~real, -100),
                     position_ids=positions)
        if self.varlen:
            cu = torch.tensor([0] + all_lengths).cumsum(0).int()
            batch.update(cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                         max_length_q=max(all_lengths), max_length_k=max(all_lengths))
        else:
            causal = torch.ones(self.seq_len, self.seq_len, dtype=torch.bool).tril()
            allowed = (segments[:, :, None] == segments[:, None, :]) & causal
            batch["attention_mask"] = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, float("-inf"))[:, None]
        return batch


def unpack_and_split(rows, split_seed=20261016):
    groups = {}
    for row in rows:
        start = 0
        for length in row["doc_lens"]:
            tokens = list(map(int, row["input_ids"][start:start + length]))
            start += length
            digest = hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()
            groups.setdefault(digest, []).append(tokens)
        if start != len(row["input_ids"]):
            raise ValueError("Corrupt evaluation packing")
    hashes = sorted(groups)
    order = np.random.default_rng(split_seed).permutation(len(hashes))
    ndev = max(1, min(len(hashes) - 1, round(len(hashes) * .2)))
    dev_hashes = {hashes[i] for i in order[:ndev]}
    dev, confirm = [], []
    for key in hashes:
        (dev if key in dev_hashes else confirm).extend(groups[key])
    return dev, confirm, fingerprint({"dev": sorted(dev_hashes), "confirm": sorted(set(hashes) - dev_hashes)})


def pack_documents(docs, seq_len=8192):
    rows, free = [], []
    for doc in sorted(docs, key=len, reverse=True):
        i = bisect.bisect_left(free, (len(doc), -1))
        if i == len(free):
            row, space = len(rows), seq_len
            rows.append([])
        else:
            space, row = free.pop(i)
        rows[row].append(doc)
        bisect.insort(free, (space - len(doc), row))
    return [dict(input_ids=list(chain.from_iterable(row)), doc_lens=[len(d) for d in row],
                 n_tokens=sum(map(len, row))) for row in rows]


def budget_order(counts, seed, max_tokens, global_batch):
    order = np.random.default_rng(seed).permutation(len(counts))
    order = order[:len(order) // global_batch * global_batch]
    batches = np.asarray(counts)[order].reshape(-1, global_batch).sum(axis=1)
    cumulative = batches.cumsum()
    steps = int(np.searchsorted(cumulative, max_tokens, side="right"))
    if steps < 1:
        raise ValueError("Token budget cannot fit one global batch")
    if steps == len(batches) and cumulative[-1] < max_tokens - global_batch * 8192:
        raise ValueError("Training dataset is too small for the requested budget")
    return order[:steps * global_batch], np.concatenate(([0], cumulative[:steps]))


def load_assets(data_dir, cache_dir, config):
    from datasets import Dataset
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train = Dataset.load_from_disk(str(data_dir / "train"))
    evaluation = Dataset.load_from_disk(str(data_dir / "eval"))
    source_id = fingerprint({"train": train._fingerprint, "eval": evaluation._fingerprint,
                             "path": str(data_dir.resolve()), "schema": 1})
    root = cache_dir / source_id[:16]
    root.mkdir(exist_ok=True)
    with (root / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        split_file = root / "split.json"
        if not split_file.exists():
            import shutil
            dev, confirm, split_id = unpack_and_split(evaluation)
            for name, docs in (("dev", dev), ("confirm", confirm)):
                target = root / name
                if target.exists():
                    shutil.rmtree(target)  # incomplete cache from an interrupted preparation
                Dataset.from_list(pack_documents(docs)).save_to_disk(str(target))
            atomic_json(split_file, dict(fingerprint=split_id, source=source_id,
                                         dev_documents=len(dev), confirm_documents=len(confirm),
                                         historically_used=True))
    import json
    split = json.loads(split_file.read_text())
    counts = np.asarray(train["n_tokens"], dtype=np.int64)
    order, prefix = budget_order(counts, config["seed"], config["max_tokens"], config["global_batch"])
    train = train.select(order.tolist())
    split["order_fingerprint"] = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
    return train, Dataset.load_from_disk(str(root / "dev")), Dataset.load_from_disk(str(root / "confirm")), prefix, split
