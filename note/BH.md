lab5-yoru-BH-552770: AZ with the block matrices on Sophia (OPTIMIZER=sophia+adamw), Sophia lr 1e-3 -> 2.5e-4 (BF's lr / 4), wd 1.34; AdamW for the embedding and norms as AZ (lr 6.8e-3, embedding wd 0.05); Sophia as BF/BG (official SophiaG, betas 0.965/0.99, rho 0.04, Gauss-Newton-Bartlett curvature every 10 steps, bias-corrected); warmup 5% -> 25% of the steps (WARMUP_FRACTION=0.25, a new env var; peak at step 1,908, then 25% stable and the last 50% linear decay to 0); the matrices' wd keeps tau = 1/(lr * wd * steps) at 0.39; everything else as AZ (131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, best-fit data from job 538113, 1B non-padding tokens); why: BF (lr 1e-3) and BG (lr 5e-4) both diverged once the lr reached its peak at step 382 (loss up, grad norm 0.2 -> 17-67); Semenov et al. 2025 saw Sophia diverge at this batch size (131k tokens) and needed about 25% warmup and a lower lr, so this round follows their small-batch recipe; the user asked for two more Sophia runs; expected: if it trains stably, 18-19.5 (4-12% worse than AZ's 17.26: Sophia is about even with AdamW in careful comparisons, and the long warmup costs some progress); it may still diverge (Semenov et al. saw divergence at small batches even at low lr); cancel early if the loss rises after the peak; how it transfers to 6B: only if it beats AZ, which is unlikely; results: cancelled at step ~2,650 of 7,632 at the user's request (21:44 on 25a-hgpn121), to start the next round with the research's changes; stable this time: no rise after the peak lr (step 1,908), loss 3.68 at step 2,600, grad norm 0.13-0.24, sophia_win_rate 0.11-0.14 after step 1,000 (inside the README's 0.1-0.5); eval ppl 304, 84.4, 60.7, 51.5, 46.3, 40.8 at steps 382-2,292 vs AZ's 139.7, 52.6, 40.8, 35.5, 32.4, 30.4 (+34% at 30%, partly the 25% warmup's lower lr early); so BF/BG diverged from too high a Sophia lr, as the follow-up research found (sources put SophiaG at 1e-4 to 3e-4 near 1B)
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "sophia+adamw"  # Sophia for the block matrices, AdamW for the embedding and norms
LEARNING_RATE = 2.5e-4  # Sophia
ADAM_LEARNING_RATE = 6.8e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 1.34  # Sophia matrices
EMBED_WEIGHT_DECAY = 0.05  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
WARMUP_FRACTION = 0.25  # first 25% of the steps: warm up
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,OPTIMIZER=sophia+adamw,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=2.5e-4,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=1.34,EMBED_WEIGHT_DECAY=0.05,WARMUP_FRACTION=0.25 script/train.sbatch BH

adam beta1 0.9, Sophia betas 0.965/0.99, rho 0.04, curvature every 10 steps, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
