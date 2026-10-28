lab5-yoru-AP-540298: AL's recipe (Muon, lr 1.2e-3, best-fit data from job 538113) at 0.5B non-padding tokens (954 steps) instead of 1B, with wd 0.8 -> 1.1 so tau follows tau ~ TPP^-0.5 (2x fewer tokens -> ~1.41x AL's tau: 0.79 vs 0.55), as in AO; why: the middle of the three points (AO 0.25B, AP 0.5B, AL 1B) for fitting L(D) = E + K * D^-beta, see AO; expected: eval ppl around 24 (23.7 with no floor, 24.2 with E = 2.0 nats, taking AL at ~19.2); how it transfers to 6B: the measurement behind the 6B prediction and the 1B target, see AO; results: eval ppl 23.50 (499,990,093 tokens, 954 steps; final value from wandb run ab8l4a21, as for AO), 0:26:40 on 8 GPUs; drops 0.357 nats from AO and 0.239 to AL per doubling, so the curve flattens quickly; for the fit and what it says about 6B see AO
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "muon"
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 1.1  # sbatch --export=ALL,MAX_TOKENS=500000000,LEARNING_RATE=1.2e-3,WEIGHT_DECAY=1.1 script/train.sbatch AP
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 500_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0
DATA_DIR = "data/bestfit"

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
