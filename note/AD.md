lab5-yoru-AD-534948: AC with LEARNING_RATE 6e-4 -> 1.2e-3 (Muon and the AdamW groups both), the first rung of the lr ladder at 1B tokens (1,907 steps); why: 6e-4 was carried over from the AdamW baseline and was never tuned for Muon + WSD, and lr is usually the most sensitive hyperparameter; expected: lower eval ppl than AC's 21.97 (DeepSeek LLM's fit eta_opt = 0.3118 * C^-0.125 gives ~1.4e-3 at this run's ~7e18 FLOPs and ~1.1e-3 at 6B; Muon's update RMS is matched to AdamW's, so AdamW's lr range applies); if worse, 6e-4 is near the optimum and the next rung is 3e-4; results: eval ppl 20.45 (-6.9% vs AC's 21.97), 0:47:21 on 8 GPUs; best 1B run so far, as expected
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
LEARNING_RATE = 1.2e-3  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.2e-3 script/train.sbatch AD
WEIGHT_DECAY = 0.1
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
