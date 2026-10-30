"""讓學生學老師的 top-128 分布；老師未保留的機率補到正確答案。

對位置 t，老師和學生都在預測 token[t + 1]，temperature = 1。
target[v] = 0.8 * teacher_prob[v]（只保留 top-128）
target[正確答案] += 1 - 0.8 * sum(teacher_top128_probs)

例如 top-128 的總機率是 0.965：老師拿 0.772，正確答案另外拿 0.228。
正確答案若也在 top-128，兩份權重相加。不能把 top-128 重新 softmax 成總和 1。
下面直接讓 PyTorch 微分；CP 原版用自訂 backward 節省記憶體和時間。
"""

import mmap

import settings as s
import torch
import torch.nn.functional as F
from data import make_batch
from torch.utils.checkpoint import checkpoint


class TeacherCache:
    def __init__(self):
        self.shards = []
        self.where = {}
        # mmap 只在讀取時載入需要的資料，避免把數 TB 快取全部讀進 RAM。
        with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
            for path in sorted(s.CACHE_DIR.glob("shard-*.pt")):
                shard = torch.load(
                    path, mmap=True, weights_only=True, map_location="cpu"
                )
                assert shard["topk_ids"].shape[1] == s.TOP_K
                for i, row in enumerate(shard["rows"].tolist()):
                    self.where[row] = (len(self.shards), i)
                self.shards.append(shard)
        if not self.shards:
            raise FileNotFoundError(f"找不到 CP 的 top-128 快取：{s.CACHE_DIR}")

    def targets(self, row_ids, device):
        ids = torch.zeros(len(row_ids), s.SEQ_LEN, s.TOP_K, dtype=torch.long)
        probs = torch.zeros(len(row_ids), s.SEQ_LEN, s.TOP_K)
        for b, row_id in enumerate(row_ids):
            shard_id, row = self.where[row_id]
            shard = self.shards[shard_id]
            start, end = shard["row_offsets"][row : row + 2].tolist()
            ids[b, : end - start] = shard["topk_ids"][start:end]
            # 分母是完整詞表的 logsumexp，不是只有 top-128。
            logits = shard["topk_logits"][start:end].float()
            probs[b, : end - start] = (
                logits - shard["logsumexp"][start:end, None]
            ).exp()
        return ids.to(device), probs.to(device)


def chunk_loss(hidden, weight, labels, teacher_ids, teacher_probs):
    logits = (hidden @ weight.T).float()
    if teacher_ids is None:  # eval 只看真實答案，不看老師
        return F.cross_entropy(logits, labels, reduction="sum")
    log_probs = logits.log_softmax(dim=-1)
    teacher_weight = s.KD_ALPHA * teacher_probs
    answer_weight = 1 - teacher_weight.sum(dim=-1)
    teacher_loss = -(teacher_weight * log_probs.gather(1, teacher_ids)).sum()
    answer_loss = -(
        answer_weight * log_probs.gather(1, labels[:, None]).squeeze(1)
    ).sum()
    return teacher_loss + answer_loss


def model_loss(model, inputs, labels, teacher_targets=None):
    hidden = model.model(**inputs, use_cache=False).last_hidden_state[:, :-1]
    labels = labels[:, 1:]  # hidden[t] 預測 labels[t + 1]
    keep = labels != -100
    hidden, labels = hidden[keep], labels[keep]
    ids = probs = None
    if teacher_targets is not None:
        ids, probs = [x[:, :-1][keep] for x in teacher_targets]

    loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(labels), 2048):
        end = start + 2048
        args = (
            hidden[start:end],
            model.lm_head.weight,
            labels[start:end],
            ids[start:end] if ids is not None else None,
            probs[start:end] if probs is not None else None,
        )
        # 詞表很大：一次只算 2048 tokens，backward 時重算，避免保存整批 logits。
        if torch.is_grad_enabled():
            loss = loss + checkpoint(chunk_loss, *args, use_reentrant=False)
        else:
            loss = loss + chunk_loss(*args)
    return loss / len(labels)


@torch.inference_mode()
def cache_teacher():
    """已經有 CP 快取就不用執行。單卡逐 row 產生與原版相同欄位的快取。"""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if any(s.CACHE_DIR.glob("shard-*.pt")):
        raise FileExistsError(
            f"請用空的 CACHE_DIR，避免混到不同資料的快取：{s.CACHE_DIR}"
        )
    rows = Dataset.load_from_disk(str(s.DATA_DIR / "train"))
    student_tokenizer = AutoTokenizer.from_pretrained(s.STUDENT_ID)
    teacher_tokenizer = AutoTokenizer.from_pretrained(s.TEACHER_ID)
    assert student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
    assert student_tokenizer.bos_token_id == teacher_tokenizer.bos_token_id
    teacher = (
        AutoModelForCausalLM.from_pretrained(
            s.TEACHER_ID, dtype=torch.bfloat16, attn_implementation="flash_attention_3"
        )
        .cuda()
        .eval()
    )
    s.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(s.SEED))
    for start in range(0, len(order), 128):
        path = s.CACHE_DIR / f"shard-{start:07d}.pt"
        row_ids = order[start : start + 128].tolist()
        all_ids, all_logits, all_lse, offsets = [], [], [], [0]
        for row_id in row_ids:
            row = rows[row_id]
            inputs, _ = make_batch([row], "cuda")
            hidden = teacher.model(**inputs, use_cache=False).last_hidden_state[
                0, : row["n_tokens"]
            ]
            for part in hidden.split(512):
                logits = teacher.lm_head(part).float()
                values, ids = logits.topk(s.TOP_K, dim=-1)
                all_ids.append(ids.cpu().int())
                all_logits.append(values.cpu().half())
                all_lse.append(logits.logsumexp(-1).cpu())
            offsets.append(offsets[-1] + row["n_tokens"])
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "rows": torch.tensor(row_ids),
                "row_offsets": torch.tensor(offsets),
                "topk_ids": torch.cat(all_ids),
                "topk_logits": torch.cat(all_logits),
                "logsumexp": torch.cat(all_lse),
            },
            temporary,
        )
        temporary.replace(path)
        print(f"老師快取：{start + len(row_ids):,} / {len(rows):,} rows")


if __name__ == "__main__":
    cache_teacher()
