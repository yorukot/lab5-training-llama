lab5-yoru-AH-536085: AD with MUON_VARIANT muon -> normuon, lr 1.2e-3, wd 0.1, at 1B tokens (1,907 steps); normuon = NorMuon (arXiv 2510.05491: each output row of the orthogonalized update is divided by the root of an EMA (beta2 0.95) of its mean square, then the matrix norm is restored, as in github.com/zichongli5/NorMuon) plus Polar Express Newton-Schulz (arXiv 2505.16932: 5 per-step quintics from github.com/NoahAmsel/PolarExpress, Frobenius margin 1.01); why: the best optimizer change with evidence at >= 1B params: NorMuon at 1.1B params / 50B tokens reached AdamW's final loss in 21.7% fewer steps vs 10.4% for Muon, and Polar Express at 774M / 1B tokens took val loss 3.398 -> 3.340; modded-nanogpt adopted both; expected: lower eval ppl than AD's 20.45, a few percent at most (the papers' Muon baselines may not have been fully tuned); caveat: Polar Express pushes singular values closer to 1, so the update RMS is larger at the same lr (CPU test on a toy down_proj: 0.200 vs 0.175, ~+14%); on the AC/AD/AE lr curve a 14% higher lr is worth only ~0.3% ppl, so a gain beyond that comes from the method; how it transfers to 6B: an optimizer change, not a horizon-dependent hyperparameter, and NorMuon's gain held at 50B tokens, so a 1B win should carry over (unverified on our setup); results: eval ppl 20.11 (-1.7% vs AD's 20.45), 0:48:14 on 8 GPUs (train_runtime 2827 s vs AD's 2767 s, ~2% slower); the gain is well beyond the ~0.3% a 14% larger update would give, so most of it comes from the method
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.2e-3,MUON_VARIANT=normuon script/train.sbatch AH
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 0.1
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
