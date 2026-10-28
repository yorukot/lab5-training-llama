lab5-yoru-CR-566262: CQ with tau 0.25 instead of 0.39 in every group (tau = 1 / (lr x wd x steps)): Muon wd 0.224 -> 0.349 (1 / (0.25 x 1.5e-3 x 7,632)), embedding wd 0.04 -> 0.0617 (1 / (0.25 x 8.5e-3 x 7,632)), the same lr; everything else as CQ (top-128 cache, KD_ALPHA 0.9, KD_UNBIASED_TAIL off, EMA-Nesterov 0.6 with gamma 0.99 over 15-74% of the steps, Muon lr 1.5e-3, AdamW lr 8.5e-3, 1% warmup, new data from job 552807, normuon, 131k tokens a step, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42) and the same code (sha256 as CQ), so the comparison with CQ (15.122) is clean (the ID CR was set aside for top-128 + K1, which was dropped when CO lost); why: the user can run only one 6B now (the last slot needs the partner's OK), so its tau has to be chosen without a 6B pair; the literature review (note/literature-review/99-summary.md) and its red team both put the 6B tau at 0.25-0.27 (in single-run papers the best tau shrinks as the tokens grow; EMA-Nesterov already scales the decay up x1.3-1.5 at 1B), and the old 1B tau curve was flat on the short side and steep on the long side; but that curve came from the old recipe (before distillation and EMA-Nesterov), and the red team argues the new parts push the best nominal tau longer; CR measures the short side of the curve with the current recipe; expected: +0.1 to +0.6% vs CQ (15.14-15.21), most likely +0.25% (~15.16), since 0.39 was tuned at 1B and the shift toward shorter tau needs more tokens; ~75 min, ~10 H200-h (a 1B screening run, not one of the 6B slots); how it transfers to 6B, the rule set beforehand (noise ~0.05-0.15% per pair): CR <= -0.3% vs CQ -> 6B tau 0.2 (wd 0.0911 / 0.0161); -0.3% to +0.25% -> 6B tau 0.25 (wd 0.0729 / 0.0129); > +0.25% -> 6B tau 0.32 (wd 0.0569 / 0.0100); wd for 6B = 1 / (tau x lr x 45,758 steps) with lr 1.2e-3 / 6.8e-3 at 5.995B tokens; results: eval ppl 15.13 (15.125; +0.02% vs CQ's 15.122), 1:13:40 on 8 GPUs (25a-hgpn145), the same speed as CQ; a tie, at the good end of the expected +0.1 to +0.6%; the path was nothing like a tie: +0.5-0.7% at 5-10%, then the gap opened once EMA-Nesterov came on at 15% (+1.9% at 15%, +4.0% at 25%, +5.9% at 40-50%, +5.5% at 60%) and closed through the decay (+4.4% at 70%, +2.8% at 80%, +1.1% at 90%, +0.5% at 95%, +0.02% at the end); the stronger decay costs a lot at a high lr and pays it back as the lr falls, and it was still gaining ~0.5% per 5% of the steps at the end, so with more steps (or a longer training, where the best tau is shorter) tau 0.25 should come out ahead; for 6B: by the rule set beforehand (-0.3% to +0.25%), the 6B run uses tau 0.25 (Muon wd 0.0729, embedding wd 0.0129); its curve will run well behind CM's (tau 0.39) through the middle and catch up only in the decay, so mid-run checkpoints say little about the final
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.5e-3  # Muon
ADAM_LEARNING_RATE = 8.5e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.349  # Muon matrices (tau 0.25)
EMBED_WEIGHT_DECAY = 0.0617  # the tied embedding (tau 0.25)
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.8  # last 80% of the steps: decay to 0
DECAY_TYPE = "linear"
WARMUP_FRACTION = 0.01
KD_ALPHA = 0.9  # the teacher's share of the target
TOP_K = 128  # src/distill.py; CACHE_DIR .../distillation/llama31-8b-top128 (job 564939)
EMA_NESTEROV = 0.6  # largest lookahead beta; window 15-74% of the steps
EMA_NESTEROV_GAMMA = 0.99  # default
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.349,EMBED_WEIGHT_DECAY=0.0617,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01,KD_ALPHA=0.9,EMA_NESTEROV=0.6 script/train.sbatch CR

warmup_steps 0.01, adam beta1 0.9, KD_ALPHA 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~75 min
