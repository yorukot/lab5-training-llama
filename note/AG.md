lab5-yoru-AG-536084: AD with WEIGHT_DECAY 0.1 -> 0.4 (the Muon group and the embedding AdamW group; norms stay at 0), lr 1.2e-3, at 1B tokens (1,907 steps); why: with wd fixed at 0.1, the weight-decay timescale tau = 1/(lr*wd*steps) is ~4.4 run lengths in AD, far longer than the ~1.0 that Power Lines (arXiv 2505.13738, AdamW) extrapolates for ~1 token/param, and Semenov et al. (arXiv 2509.01440) find wd >= 0.5 helps short runs; AG's tau is ~1.09; this tests whether a shorter tau helps, i.e. whether part of AD/AE's gain over AC came from stronger decay rather than the lr itself; expected: lower eval ppl than AD's 20.45, small to medium (no number at our scale, unverified for Muon); how it transfers to 6B: not the wd value itself; the predicted best tau shrinks as tokens/param grow (tau_opt ~ TPP^-0.5: ~1.0 at 1B, ~0.43 at 6B), so if AG wins, the 6B wd = 1/(lr * 11444 * 0.43) (~0.2 at lr 1e-3), to be checked before the 6B run; results: eval ppl 19.90 (-2.7% vs AD's 20.45), 0:47:05 on 8 GPUs; best 1B run so far; a shorter tau helps at 1B, as the timescale view predicts, so part of AD/AE's gain over AC may also have come from stronger decay
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "muon"
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 0.4  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.2e-3,WEIGHT_DECAY=0.4 script/train.sbatch AG
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
