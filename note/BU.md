lab5-yoru-BU-554701 (first try 554674 failed within 1 s on 25a-hgpn173 with no log written, signal 53; node now excluded): BR with BQ's lrs, both x1.25: Muon 1.2e-3 -> 1.5e-3, AdamW (embedding, norms) 6.8e-3 -> 8.5e-3; wd so every group's tau stays 0.39: Muon 0.224, embedding 0.04; everything else as BR (1% warmup, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with BR (17.061) is clean; why: BR (1% warmup, -0.52% vs BN) and BQ (lrs x1.25, -0.31% vs BN) both won; they act on different parts of the schedule (BR the first 5%, BQ the whole peak), so they should mostly add up; a shorter warmup with a higher peak lr raises the early spike risk, which this run also checks; expected: -0.2 to -0.3% vs BR (about 17.01); how it transfers to 6B: the 6B recipe would take both, with the lrs x0.8 from 1.5e-3 / 8.5e-3; results: eval ppl 17.03 (17.027; -0.20% vs BR's 17.061, -0.72% vs BN's 17.151), 1:07:19 on 8 GPUs (25a-hgpn174); best 1B run so far; no loss spikes (grad norm up to 8.5 in the first 30 steps, then falling); a small, steady lead from 20% on (-0.19% at 25%, -0.09% at 50%, -0.22% at 75%, -0.20% at the end); BR's and BQ's gains mostly add up (-0.72% vs -0.83% if fully additive); with BR (x1) and BV (x1.5), a parabola in log lr puts the best at x1.23, so the lrs are settled: Muon 1.5e-3, AdamW 8.5e-3; for 6B: this recipe (1% warmup, Muon 1.5e-3, AdamW 8.5e-3) before the scaling to 6B
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
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01 script/train.sbatch BU

warmup_steps 0.01, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
