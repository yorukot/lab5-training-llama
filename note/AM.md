lab5-yoru-AM-539950 (first try 539940 crashed on the doc_lens column, see AL): AL with MUON_VARIANT muon -> normuon (NorMuon + Polar Express, as in AH/AJ), lr 1.2e-3, wd 0.8, on the best-fit data (data/bestfit, job 538113), at 1B non-padding tokens; why: wd 0.8 (AK) and normuon (AJ) are the two best changes so far, so this is the candidate recipe; expected: lower eval ppl than AL by about 1% (normuon gave -1.7% at wd 0.1 and -1.35% at wd 0.4, so the gain may shrink a bit more at wd 0.8); caveat as in AH: normuon's update RMS is ~14% larger at the same lr, worth only ~0.3% ppl; how it transfers to 6B: the normuon part is an optimizer change and should carry over (NorMuon's gain held at 50B tokens); wd carries through tau, not as a value; results: eval ppl 18.37 (-0.76% vs AL's 18.51), 0:47:36 on 8 GPUs; best 1B run so far; normuon's gain keeps shrinking as wd grows: -1.7% at wd 0.1 (AH), -1.35% at wd 0.4 (AJ), -0.76% at wd 0.8, so part of what it fixes overlaps with what stronger wd fixes
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.2e-3,WEIGHT_DECAY=0.8,MUON_VARIANT=normuon script/train.sbatch AM
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 0.8
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0
DATA_DIR = "data/bestfit"

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
