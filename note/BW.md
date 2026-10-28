lab5-yoru-BW-555827: BU with the warmup 1% -> 0.25% of the steps (~19 steps, 2.5M tokens; WARMUP_FRACTION=0.0025), Muon momentum 0.95; everything else as BU (Muon lr 1.5e-3 wd 0.224, AdamW lr 8.5e-3 embedding wd 0.04, every group's tau 0.39, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with BU (17.027) is clean; why: 5% -> 1% gave -0.52% (BR), the largest single gain since the batch change; the curve may continue; Muon's fixed-size update needs little warmup (modded-nanogpt uses none for its matrices), and the first steps are clipped by max_grad_norm 1.0 anyway (grad norm 5-8 there); risk: loss spikes in the first ~100 steps from the AdamW embedding at lr 8.5e-3; expected: -0.05 to -0.15% vs BU; a spike or a loss means 1% is about the best; how it transfers to 6B: the warmup needed is set by the first, noisy steps, not by the run length, so the 6B run would use the same number of warmup steps (or the same fraction if that is longer); results: eval ppl 17.05 (17.050; +0.14% vs BU's 17.027), 1:07:39 on 8 GPUs (25a-hgpn058); worse: an early spike (grad norm 27 at step 20, loss back up from 9.40 there to 9.86 at step 30, while BU was at 8.07) put it +1.7% behind at 5%, and it never caught up (+0.20% at 15%, +0.38% at 30%, +0.17% at 75%, +0.14% at the end); with BN (5%, +0.52% vs BR) and BW (0.25%, +0.14%), 1% is about the best warmup, so the warmup is settled; for 6B: 1% warmup (~460 steps at 6B, longer than the ~76 steps at 1B, which is the safer side)
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
WARMUP_FRACTION = 0.0025
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.0025 script/train.sbatch BW

warmup_steps 0.0025, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
