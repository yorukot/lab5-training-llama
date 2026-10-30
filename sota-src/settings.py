"""只保留 CP（lab5-yoru-CP-566429）這一份配方。

原始結果：in-training PPL 11.3955618，OJ PPL 11.5252。
原始紀錄：note/CP.md、training-history/lab5-yoru-CP-566429.csv。

閱讀順序：train.py → data.py → distillation.py → optimizer.py。
這裡是方便閱讀的實作：拿掉 Trainer、分散式訓練和 Liger，直接寫訓練迴圈。
CP 用 8 張 H200，每張每次 2 rows；這裡用 1 張 GPU，2 rows 累積 8 次。
每步仍然是 16 rows，但運算順序、kernel 不同，不保證逐位元重現 11.3956，
也不會有原本 6 小時 43 分的速度。需要能容納 1B 模型訓練的 CUDA GPU 和 FA3。

先改下面兩個資料路徑，再直接執行 python sota-src/train.py，沒有命令列選項。
資料與老師快取不在這份 repo 裡；必須使用 CP 當時那一份，兩者的 row 必須對齊。
若只有 CP 資料、沒有快取，可先執行 python sota-src/distillation.py。
"""

from pathlib import Path

DATA_DIR = Path("/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit")
CACHE_DIR = DATA_DIR.parent / "distillation/llama31-8b-top128"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs/sota-readable/final"

STUDENT_ID = "meta-llama/Llama-3.2-1B"  # 只取架構，權重隨機初始化
TEACHER_ID = "meta-llama/Llama-3.1-8B"  # 老師才載入預訓練權重
SEED = 42
SEQ_LEN = 8192
PAD_ID = 128004  # <|finetune_right_pad_id|>
GLOBAL_BATCH = 16
MICRO_BATCH = 2
MAX_TOKENS = 5_995_000_000  # 包含 BOS，不包含 padding

MUON_LR = 1.2e-3
MUON_WD = 0.0729
ADAM_LR = 6.8e-3
EMBED_WD = 0.0129  # norm 不做 weight decay
ADAM_BETAS = (0.9, 0.9873)
MUON_MOMENTUM = 0.95
NORMUON_BETA2 = 0.95
WARMUP_FRACTION = 0.01
DECAY_FRACTION = 0.8
MAX_GRAD_NORM = 1.0

KD_ALPHA = 0.8
TOP_K = 128
EMA_BETA = 0.6
EMA_GAMMA = 0.99
EMA_START = 0.15
EMA_END = 0.74
