lab5-yoru-BI-552771: AZ with the block matrices on Sophia (OPTIMIZER=sophia+adamw), Sophia lr 1e-3 -> 2.5e-4 (BF's lr / 4), wd 1.34; AdamW for the embedding and norms at a lower lr, 6.8e-3 -> 1.7e-3 (AU's, before the AdamW lr was raised for Muon; embedding wd 0.05 -> 0.2 keeps its tau 0.39); Sophia as BF/BG (official SophiaG, betas 0.965/0.99, rho 0.04, Gauss-Newton-Bartlett curvature every 10 steps, bias-corrected); warmup 5% -> 25% of the steps (WARMUP_FRACTION=0.25, a new env var; peak at step 1,908, then 25% stable and the last 50% linear decay to 0); the matrices' wd keeps tau = 1/(lr * wd * steps) at 0.39; everything else as AZ (131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, best-fit data from job 538113, 1B non-padding tokens); why: BF (lr 1e-3) and BG (lr 5e-4) both diverged once the lr reached its peak at step 382 (loss up, grad norm 0.2 -> 17-67); Semenov et al. 2025 saw Sophia diverge at this batch size (131k tokens) and needed about 25% warmup and a lower lr, so this round follows their small-batch recipe; the user asked for two more Sophia runs; expected: separates the two suspects: 6.8e-3 was tuned next to Muon, and with Sophia on the matrices it may be the embedding that blows up; if BH diverges and BI does not, the embedding lr was the trigger; if both train, their gap shows what the embedding lr is worth under Sophia; if both diverge, Sophia itself is unstable at this batch; if stable, 18-19.5; how it transfers to 6B: only if it beats AZ, which is unlikely; results: cancelled at step ~2,650 with BH; stable, but worse than BH all the way: eval ppl 402, 106, 67.3, 58.8, 56.5, 54.2 at steps 382-2,292 (BH: 40.8 at 2,292), loss 4.00 vs BH's 3.68 at step 2,600, sophia_win_rate 0.15-0.20; so the embedding's high AdamW lr (6.8e-3) was not what made BF/BG diverge, and it still helps under Sophia
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "sophia+adamw"  # Sophia for the block matrices, AdamW for the embedding and norms
LEARNING_RATE = 2.5e-4  # Sophia
ADAM_LEARNING_RATE = 1.7e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 1.34  # Sophia matrices
EMBED_WEIGHT_DECAY = 0.2  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
WARMUP_FRACTION = 0.25  # first 25% of the steps: warm up
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,OPTIMIZER=sophia+adamw,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=2.5e-4,ADAM_LEARNING_RATE=1.7e-3,WEIGHT_DECAY=1.34,EMBED_WEIGHT_DECAY=0.2,WARMUP_FRACTION=0.25 script/train.sbatch BI

adam beta1 0.9, Sophia betas 0.965/0.99, rho 0.04, curvature every 10 steps, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
