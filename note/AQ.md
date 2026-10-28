lab5-yoru-AQ-540616: AM with the decay shape linear -> 1-sqrt (DECAY_TYPE=1-sqrt, lr factor 1 - sqrt(progress) over the last 50% of the steps), everything else as AM (normuon, lr 1.2e-3, wd 0.8, best-fit data from job 538113, 1B non-padding tokens); why: Hagele et al. 2024 (Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations) found the 1-sqrt cooldown beats linear: it drops fast at first (half the lr after 25% of the decay) and then spends longer at a low lr; expected: -0.3% to -1% vs AM's 18.37; caveat: their cooldowns were mostly ~20% of training, and over our 50% the mean lr factor of the decay falls from 0.5 (linear) to 0.33, so less total update, which could cost more than the shape gains; how it transfers to 6B: the schedule shape does not depend on the run length, so the winner carries over as is; results: eval ppl 18.48 (+0.60% vs AM's 18.37), 0:47:58 on 8 GPUs; worse: 1-sqrt led by up to 3% during the decay (19.41 vs 20.00 at step 1632) but lost it by the end, the caveat in the expectation: over a 50% decay its mean lr factor is 0.33 vs linear's 0.5, so less total update; keep linear (1-sqrt with a shorter decay, ~20% as in Hagele et al., is untested)
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 0.8
EMBED_WEIGHT_DECAY = 0.8  # the tied embedding; the other matrices use WEIGHT_DECAY
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "1-sqrt"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.2e-3,WEIGHT_DECAY=0.8,MUON_VARIANT=normuon,DECAY_TYPE=1-sqrt script/train.sbatch AQ

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
