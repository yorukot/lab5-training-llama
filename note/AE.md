lab5-yoru-AE-534949: AC with LEARNING_RATE 6e-4 -> 2.4e-3 (Muon and the AdamW groups both), the second rung of the lr ladder at 1B tokens (1,907 steps), run alongside AD; why: with only 6e-4 and 1.2e-3, a win at 1.2e-3 cannot tell whether the optimum is higher still, so this brackets it from above; expected: uncertain, which is the point of the run: worse than AD (or loss / grad_norm spikes) puts the optimum between 6e-4 and 2.4e-3, better than AD makes 4.8e-3 the next rung; results: eval ppl 20.85 (-5.1% vs AC's 21.97, but worse than AD's 20.45), 0:48:15 on 8 GPUs; so the 1B optimum is between 6e-4 and 2.4e-3: a parabola in log2(lr) through AC/AD/AE eval losses puts it at ~1.46e-3 (predicted ppl ~20.38, only ~0.4% better than AD), close to DeepSeek's 1.37e-3; still confounded with weight decay (wd fixed at 0.1, so higher lr also decays harder)
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
LEARNING_RATE = 2.4e-3  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=2.4e-3 script/train.sbatch AE
WEIGHT_DECAY = 0.1
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
