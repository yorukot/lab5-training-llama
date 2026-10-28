lab5-yoru-AN-539951 (first try 539941 crashed on the doc_lens column, see AL): AL with WEIGHT_DECAY 0.8 -> 1.6 (Muon group and embedding AdamW group; norms stay at 0), lr 1.2e-3, Muon, on the best-fit data (data/bestfit, job 538113), at 1B non-padding tokens; why: AK (wd 0.8, tau ~0.55) beat AI (wd 0.4, tau ~1.09) by 2.0%, against the Power Lines extrapolation, so the best wd at 1B may be higher still; AN halves tau again (1/(1.2e-3 * 1.6 * ~1909) ~ 0.27) to bracket it from above; expected: uncertain, which is the point: worse than AL puts the best tau between ~0.27 and ~1.09 (near 0.55), better than AL means wd should go up again; how it transfers to 6B: through tau, not the wd value: tau ~ TPP^-0.5 gives the 6B tau as ~0.41x the best 1B tau (6x the tokens), e.g. best 1B tau 0.55 -> 6B tau ~0.22 -> wd ~0.4 at lr 1e-3 (unverified for Muon); results: eval ppl 18.45 (-0.32% vs AL's 18.51), 0:47:38 on 8 GPUs; worse than AL through the stable phase (e.g. 20.66 vs 20.16 at 85%) but better after the decay, so the best tau at 1B is at or below ~0.55 and the curve is flat between 0.27 and 0.55 (the 0.32% gap may be within seed noise); scaled to 6B with tau ~ TPP^-0.5: tau ~0.11-0.22, i.e. wd ~0.33-0.66 at lr 1.2e-3 over ~11,450 steps (unverified for Muon)
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "muon"
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 1.6  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.2e-3,WEIGHT_DECAY=1.6 script/train.sbatch AN
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0
DATA_DIR = "data/bestfit"

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
