lab5-yoru-AO-540297: AL's recipe (Muon, lr 1.2e-3, best-fit data from job 538113) at 0.25B non-padding tokens (477 steps) instead of 1B, with wd 0.8 -> 1.6 so tau = 1/(lr * wd * steps) follows the rule we plan for 6B, tau ~ TPP^-0.5 (4x fewer tokens -> 2x AL's tau: 1.09 vs 0.55; keeping wd 0.8 would give tau 2.2, too weak, and make this point too high); why: one of three points (AO 0.25B, AP 0.5B, AL 1B) for fitting L(D) = E + K * D^-beta on our own recipe, to predict the 6B score and how low 1B has to go for 6B to reach Llama-3.2-1B's official 9.79 (AC 1B vs AB 2.5B alone could not pin the floor E: the 1B needed ranged from 15.3 with no floor to 12.1 with E = 2.0); expected: eval ppl around 30-32 (about 29.8 if there is no floor, about 32.4 if E = 2.0 nats, taking AL at ~19.2); caveats: with only 477 steps (24 warmup) this point may sit in an early-training regime that bends the fit, and the tau rule is unverified for Muon (AK beat its prediction at 1B); how it transfers to 6B: not a recipe change, it is the measurement behind the 6B prediction and the 1B target; results: eval ppl 33.57 at step 476 (249,483,605 tokens), 0:16:35 on 8 GPUs; the TokenBudget callback stops one step early here (it assumes a full 524,288-token step), so the end-of-training eval inside train() did not run and the final value is from trainer.evaluate(), logged to wandb only (run p9mpdy94); fit through AO, AP (23.50) and AL (18.51) as L = E + K * D^-beta: floor ppl 11.33, beta 0.57, 6B 13.51 (13.1-13.9 with +-0.3% noise on each point); this floor is above the official Llama-3.2-1B's 9.79 on the same eval, which the architecture already reaches, so the 0.25B point (only 476 steps) bends the curve and this fit is too pessimistic; from AP and AL alone with the floor capped (floor ppl 1 / 4.5 / 7.4 / 9.0): 6B 10.8 / 11.6 / 12.2 / 12.7, the 1B needed for 6B < 9.79 is 16.4 / 14.4 / 12.3 / 10.7, and this recipe would need ~9 / 14 / 35 / 180B tokens to reach 9.79; so for 6B < 9.79 the recipe needs 1.5-6x the data efficiency of AL's
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "muon"
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 1.6  # sbatch --export=ALL,MAX_TOKENS=250000000,LEARNING_RATE=1.2e-3,WEIGHT_DECAY=1.6 script/train.sbatch AO
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 250_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0
DATA_DIR = "data/bestfit"

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
