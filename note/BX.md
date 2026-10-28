lab5-yoru-BX-555828: BU with Muon's Nesterov momentum 0.95 -> 0.9 (MUON_MOMENTUM=0.9); 1% warmup; everything else as BU (Muon lr 1.5e-3 wd 0.224, AdamW lr 8.5e-3 embedding wd 0.04, every group's tau 0.39, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with BU (17.027) is clean; why: BD (0.98) was +0.88% vs AZ, worse all the way and most early: at 131k tokens a step the longer average lags more than it gains; the curve is steep around 0.95, so 0.9 may be better (faster reaction, more noise per step); modded-nanogpt and Moonlight use 0.95 at larger batches; expected: within +-0.3% of BU, direction unknown; how it transfers to 6B: the momentum is set by the noise per step, i.e. by the batch, which stays 131k at 6B, so a 1B winner should hold; results: eval ppl 17.04 (17.043; +0.09% vs BU's 17.027), 1:07:11 on 8 GPUs (25a-hgpn061); a tie, slightly worse: far ahead early (-6.8% at 5%, -1.6% at 15%, -0.53% at 30%), level by 45%, then slightly behind during the decay (+0.18% at 75%, +0.09% at the end); a shorter average reacts faster while the lr is high, but while the lr falls the longer average of 0.95 filters more noise; with BD (0.98, +0.88%) the best momentum is about 0.93-0.95, so it is settled at 0.95; a schedule (0.9 early, 0.95 later) might keep both, but the early lead was gone by the decay anyway (like BL), so the gain would be small; for 6B: momentum 0.95
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
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01,MUON_MOMENTUM=0.9 script/train.sbatch BX

warmup_steps 0.01, adam beta1 0.9, Muon momentum 0.9, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
