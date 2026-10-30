"""讀取 CP 已準備好的資料，讓同一 row 裡的文件彼此看不到。

CP 的資料來自 prepare job 552807（詳見 note/prepare-AB.md）：
1. dolma3_mix-150B-1025 的文件用 seed 42 排列，最後 50,000 篇留給驗證。
2. 訓練端排除驗證文件的複本，做全文去重、品質過濾、MinHash 近似去重。
3. 每篇是 BOS + 本文，不加 EOS，最多 8192 tokens。
4. 每批 10,000 篇由長到短做 best-fit packing；文件不跨 row。
5. packed rows 用 seed 42 打亂，取完整 rows 到約 6B tokens。
結果是 732,756 個 train rows / 6,000,008,014 tokens；eval 使用保留集中的 10k 篇。

目前 src/prepare.py 已還原舊版，不能拿來重新產生這份 CP 資料。
這裡直接讀 CP 成品，不另寫一套近似的資料清洗來冒充原始流程。
每個 row 有 input_ids、doc_lens（各篇長度）、n_tokens 三個欄位。
"""

import torch
from settings import PAD_ID, SEQ_LEN


def make_batch(rows, device):
    input_ids = torch.full((len(rows), SEQ_LEN), PAD_ID, dtype=torch.long)
    position_ids = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    segment_lengths = []

    for i, row in enumerate(rows):
        n = row["n_tokens"]
        assert n == len(row["input_ids"]) == sum(row["doc_lens"])
        assert 0 < n <= SEQ_LEN
        input_ids[i, :n] = torch.tensor(row["input_ids"])
        labels[i, :n] = input_ids[i, :n]

        lengths = list(row["doc_lens"])
        if n < SEQ_LEN:
            lengths.append(SEQ_LEN - n)  # padding 自己也是一個獨立區段
        position_ids[i] = torch.cat([torch.arange(length) for length in lengths])
        segment_lengths.extend(lengths)

    # 例如 [BOS, A, B, BOS, C, PAD]：不計 BOS、PAD 的 loss。
    # labels 下一步會右移一格，於是也不會要求 B 去預測下一篇的 BOS。
    labels[position_ids == 0] = -100

    # FA3 將攤平後的 batch 在這些位置切開，各區段內才做 causal attention。
    # 只重設 position_ids 不夠，還要把文件邊界傳給 attention。
    offsets = torch.tensor([0] + segment_lengths).cumsum(0).int().to(device)
    inputs = {
        "input_ids": input_ids.to(device),
        "position_ids": position_ids.to(device),
        "cu_seq_lens_q": offsets,
        "cu_seq_lens_k": offsets,
        "max_length_q": max(segment_lengths),
        "max_length_k": max(segment_lengths),
    }
    return inputs, labels.to(device)
