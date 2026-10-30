"""先讀 settings.py 的說明，再從 main() 看整個 SOTA 訓練流程。

模型從隨機權重開始；每一步用正確答案 + 8B 老師的 top-128 分布教它。
矩陣交給 NorMuon，embedding / norm 交給 AdamW，再加上 EMA-Nesterov。
這些方法加上 CP 的資料、batch、排程和約 5.995B tokens，構成 11.4 的配方。
11.4 是舊 CP 實驗的實測成績，這份閱讀版沒有重新跑完整訓練。
"""

import math

import settings as s
import torch
from data import make_batch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

from distillation import TeacherCache, model_loss
from optimizer import SOTAOptimizer


@torch.no_grad()
def evaluate(model, eval_rows):
    model.eval()
    losses = []
    # CP 的 8 卡 eval 每輪取 16 rows，drop_last 使最後 8 / 1544 rows 沒被評到。
    # 保留這個細節，並像 Trainer 一樣平均各個 2-row batch 的 loss。
    # 所以這裡的 PPL 和 OJ 全部 50k 文件、按 token 加權的 PPL 要分開看。
    stop = len(eval_rows) // s.GLOBAL_BATCH * s.GLOBAL_BATCH
    for start in range(0, stop, s.MICRO_BATCH):
        rows = [eval_rows[i] for i in range(start, start + s.MICRO_BATCH)]
        inputs, labels = make_batch(rows, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model_loss(model, inputs, labels).item())
    model.train()
    return math.exp(sum(losses) / len(losses))


def main():
    train_rows = Dataset.load_from_disk(str(s.DATA_DIR / "train"))
    eval_rows = Dataset.load_from_disk(str(s.DATA_DIR / "eval"))
    assert len(train_rows) == 732_756, "請使用 CP 的 prepare-66ea6ae 資料"
    total_tokens = sum(train_rows["n_tokens"])
    assert total_tokens == 6_000_008_014, "資料和 CP 不同，老師的 row 也可能對不上"
    teacher = TeacherCache()

    # 只下載模型設定和 tokenizer，學生不載入預訓練權重。
    set_seed(s.SEED)
    config = AutoConfig.from_pretrained(s.STUDENT_ID)
    model = (
        AutoModelForCausalLM.from_config(
            config, dtype=torch.float32, attn_implementation="flash_attention_3"
        )
        .cuda()
        .train()
    )
    tokenizer = AutoTokenizer.from_pretrained(s.STUDENT_ID)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    model.config.pad_token_id = tokenizer.pad_token_id
    optimizer = SOTAOptimizer(model)

    # 不把 padding 算進 token 預算；使用資料的平均填滿程度估算總步數。
    average_tokens = total_tokens / len(train_rows)
    total_steps = int(s.MAX_TOKENS / (s.GLOBAL_BATCH * average_tokens))  # 45,759
    eval_every = math.ceil(total_steps * 0.05)
    accumulation = s.GLOBAL_BATCH // s.MICRO_BATCH  # 單卡累積 8 次，再更新一次
    order = torch.randperm(
        len(train_rows), generator=torch.Generator().manual_seed(s.SEED)
    ).tolist()
    tokens_seen = 0

    for step in range(1, total_steps + 1):
        # 保守預留完整一步的空間，與 CP 的 token budget callback 一致。
        if tokens_seen + s.GLOBAL_BATCH * s.SEQ_LEN > s.MAX_TOKENS:
            break
        row_ids = order[(step - 1) * s.GLOBAL_BATCH : step * s.GLOBAL_BATCH]
        model.zero_grad(set_to_none=True)
        step_loss = 0.0

        for start in range(0, s.GLOBAL_BATCH, s.MICRO_BATCH):
            ids = row_ids[start : start + s.MICRO_BATCH]
            rows = [train_rows[i] for i in ids]
            inputs, labels = make_batch(rows, "cuda")
            targets = teacher.targets(ids, "cuda")

            # 主權重維持 float32，矩陣乘法用 bfloat16。
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model_loss(model, inputs, labels, targets)
            (loss / accumulation).backward()
            step_loss += loss.item() / accumulation
            tokens_seen += sum(row["n_tokens"] for row in rows)

        torch.nn.utils.clip_grad_norm_(model.parameters(), s.MAX_GRAD_NORM)
        optimizer.step(step, total_steps)
        if step % 10 == 0:
            print(
                f"step {step}/{total_steps}, loss {step_loss:.4f}, tokens {tokens_seen:,}"
            )
        if step % eval_every == 0:
            print(f"eval PPL = {evaluate(model, eval_rows):.4f}")

    print(f"final PPL = {evaluate(model, eval_rows):.4f}, tokens = {tokens_seen:,}")
    model.save_pretrained(s.OUTPUT_DIR)
    tokenizer.save_pretrained(s.OUTPUT_DIR)


if __name__ == "__main__":
    main()
