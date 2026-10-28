lab5-yoru-BZ-557109: BU with NorMuon's row EMA beta2 0.95 -> 0.9873 (NORMUON_BETA2, a new env var read in src/muon.py, default 0.95); everything else as BU (Muon lr 1.5e-3 wd 0.224, AdamW lr 8.5e-3 embedding wd 0.04, every group's tau 0.39, 1% warmup, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with BU (17.027) is clean; why: AU cut the batch from 524k to 131k tokens a step and rescaled AdamW's beta2 0.95 -> 0.9873 = 0.95 ** (1/4) to keep its half-life in tokens (Marek et al. 2025), but NorMuon's row EMA, also set to 0.95 at 524k (AM), was never rescaled: at 131k it averages a quarter of the tokens, so each row's scale follows the noisier small-batch updates; expected: -0.05 to -0.2% vs BU; worse means the row scale should track the current update closely; how it transfers to 6B: the 6B run keeps the 131k batch, so a 1B winner carries over as is; results: eval ppl 17.05 (17.047; +0.12% vs BU's 17.027), 1:07:37 on 8 GPUs (25a-hgpn041); slightly worse: level with BU through 40% (-0.25% to +0.22%), then +0.1-0.24% behind from 50% on (+0.24% at 75%) and +0.12% at the end; the longer row average does not help: the row scales of the orthogonalized update are not noisy enough at 131k tokens a step for the 4x longer average to pay for its lag; NorMuon beta2 stays 0.95; for 6B: 0.95
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.5e-3  # Muon
ADAM_LEARNING_RATE = 8.5e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.224  # Muon matrices
EMBED_WEIGHT_DECAY = 0.04  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.8  # last 80% of the steps: decay to 0
DECAY_TYPE = "linear"
WARMUP_FRACTION = 0.01
NORMUON_BETA2 = 0.9873
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01,NORMUON_BETA2=0.9873 script/train.sbatch BZ

warmup_steps 0.01, adam beta1 0.9, NorMuon beta2 0.9873, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
