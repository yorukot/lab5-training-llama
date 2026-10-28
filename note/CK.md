lab5-yoru-CK-561710: BY (3B tokens) with distillation, KD_ALPHA=0.8; everything else as BY (Muon lr 1.3e-3 wd 0.086, AdamW lr 7.4e-3 for the embedding and norms, embedding wd 0.0151, every group's tau 0.39, 1% warmup, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~22,898 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 3B non-padding tokens, seed 42), so the comparison with BY (13.734) is clean; distillation as in CE/CH (Llama 3.1 8B, its top-32 cached for all rows (job 560351); target = alpha x the teacher's top-32 probabilities, the rest on the actual next token), code as CH, unchanged; why: the 6B projection hangs on how much of distillation's 1B gain (CH, -9.04% vs BU) is left with more tokens; within a 1B run the lead shrinks from -13% at 5% to -9% at the end, and BY kept only a third of BU's 1B tricks, so the 6B gain could be anywhere from ~2% to ~8% (6B in-training ~11.0-11.7); a 3B point pins the trend down, and it runs the 6B recipe for 3x as many steps as any KD run so far; alpha 0.8, not CH's 0.9: it is the 6B candidate (the 1B best is ~0.86, and CI showed the best share falls as training goes on), and at 1B 0.7-0.9 differ by only 0.1-0.4%, small next to the gain being measured; expected: 12.6-13.3 (-3 to -8% vs BY), most likely ~12.9 (-6%); ahead of BY at every eval, by less toward the end; how it transfers to 6B: with g1 = 9.0% at 1B and g3 at 3B, extrapolating the gain in log tokens gives g6 ~ g3 x (g3/g1)^0.63, so g3 = 8% -> 6B ~11.07, 6% -> ~11.40, 3% -> ~11.78 in-training (from 11.96 without KD); a gain much below 3% would mean the teacher matters little at 6B; results: cancelled after 3:46 (step 313 of 22,898) at the user's request, to run distillation together with EMA-Nesterov at 3B instead (CL); its first steps were healthy (2.02 it/s, first grad norm 6.352 vs BY's 6.179, so distillation was on)
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.3e-3  # Muon
ADAM_LEARNING_RATE = 7.4e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.086  # Muon matrices
EMBED_WEIGHT_DECAY = 0.0151  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 3_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.8  # last 80% of the steps: decay to 0
DECAY_TYPE = "linear"
WARMUP_FRACTION = 0.01
KD_ALPHA = 0.8  # the teacher's share of the target
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=3000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.3e-3,ADAM_LEARNING_RATE=7.4e-3,WEIGHT_DECAY=0.086,EMBED_WEIGHT_DECAY=0.0151,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01,KD_ALPHA=0.8 script/train.sbatch CK

warmup_steps 0.01, adam beta1 0.9, KD_ALPHA 0.8, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~3.3 h (BY 3:10 + ~3% for distillation), ~26 H200-h
