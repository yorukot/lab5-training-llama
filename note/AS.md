lab5-yoru-AS-540618: AM with lr 1.2e-3 -> 1.7e-3 (all groups; wd stays 0.8, so tau = 1/(lr * wd * steps) goes 0.55 -> 0.39, inside the 0.27-0.55 range where AL/AN were flat), everything else as AM (normuon, linear decay, best-fit data from job 538113, 1B non-padding tokens); why: lr 1.2e-3 was picked at wd 0.1 with plain Muon (AC 6e-4 21.97, AD 1.2e-3 20.45, AE 2.4e-3 20.85, fitted best ~1.46e-3); since then normuon (~14% larger updates at the same lr) and wd 0.8 may have moved the best lr, and the 6B run needs a checked value; 1.7e-3 (~sqrt(2) x) tells which side of 1.2e-3 the best lies; expected: within about +-0.5% of AM's 18.37; better means the best lr is above 1.2e-3, worse means 1.2e-3 is at or past it; how it transfers to 6B: the best lr falls slowly with compute (DeepSeek LLM: lr ~ C^-0.125, 6x the tokens -> ~0.8x), so 6B uses ~0.8x the best 1B lr; results: eval ppl 18.25 (-0.65% vs AM's 18.37), 0:47:52 on 8 GPUs; best 1B run so far: with normuon and wd 0.8 the best lr is above 1.2e-3; ahead early, behind through the stable phase (+0.9% at step 1632), ahead after the decay; caveat: tau also fell 0.55 -> 0.39, and a lower tau alone gave -0.32% in AN (plain Muon, tau 0.27), so part of the gain may come from tau rather than lr; scaled to 6B: ~0.8x the best 1B lr, ~1.4e-3 if 1.7e-3 stays best
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.7e-3
WEIGHT_DECAY = 0.8
EMBED_WEIGHT_DECAY = 0.8  # the tied embedding; the other matrices use WEIGHT_DECAY
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.7e-3,WEIGHT_DECAY=0.8,MUON_VARIANT=normuon script/train.sbatch AS

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
