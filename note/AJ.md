lab5-yoru-AJ-536870: AI with MUON_VARIANT muon -> normuon (NorMuon + Polar Express, as in AH), lr 1.2e-3, wd 0.4, on the new train data (job 536429), at 1B tokens (1,907 steps); why: AG (wd 0.4, -2.7% vs AD) and AH (normuon, -1.7% vs AD) were each the best single change at 1B; this checks whether the two gains stack; expected: lower eval ppl than AI, most likely by less than AH's 1.7% because both changes act on the size and conditioning of the weight updates, so part of their gains may overlap (unverified); caveat as in AH: normuon's update RMS is ~14% larger at the same lr (CPU toy test), worth only ~0.3% ppl on the lr curve; how it transfers to 6B: the normuon part is an optimizer change and should carry over (NorMuon's gain held at 50B tokens); the wd part does not carry as a value: use the tau rule (6B wd = 1/(lr * 11444 * 0.43), ~0.2 at lr 1e-3); results: eval ppl 19.74 (-1.35% vs AI's 20.01), 0:48:15 on 8 GPUs; the gains partly stack, as expected: normuon adds -1.35% on top of wd 0.4, vs -1.7% on top of wd 0.1 (AH vs AD)
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.2e-3,WEIGHT_DECAY=0.4,MUON_VARIANT=normuon script/train.sbatch AJ
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 0.4
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3; train data without copies of held-out docs (job 536429)
