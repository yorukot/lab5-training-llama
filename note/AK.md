lab5-yoru-AK-536871: AI with WEIGHT_DECAY 0.4 -> 0.8 (Muon group and embedding AdamW group; norms stay at 0), lr 1.2e-3, Muon, on the new train data (job 536429), at 1B tokens (1,907 steps); why: AG showed a shorter weight-decay timescale helps (tau 4.4 -> 1.09 gave -2.7% vs AD); AK halves it again (tau = 1/(1.2e-3 * 0.8 * 1907) ~ 0.55) to see whether 0.4 is still too conservative; expected: about equal to AI or slightly worse, since Power Lines (arXiv 2505.13738, AdamW, extrapolated) puts the best tau near 1.0 at ~1 token/param, which AI already matches; if AK wins, the best tau is shorter than that fit predicts; how it transfers to 6B: through tau, not the wd value: the 6B wd = 1/(lr * 11444 * tau_6B), with tau_6B ~ 0.43 if AI wins (~0.2 at lr 1e-3), or about half that tau (~0.4 at lr 1e-3) if AK wins (unverified for Muon); results: eval ppl 19.61 (-2.0% vs AI's 20.01), 0:47:31 on 8 GPUs; best 1B run so far and against the expectation: the best tau at 1B is ~0.55 or shorter, not ~1.0, so wd may still go higher (1.6 next); scaled to 6B (tau ~ TPP^-0.5, 6x the tokens): tau ~0.22, i.e. wd ~0.4 at lr 1e-3 (unverified)
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "muon"
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 0.8  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.2e-3,WEIGHT_DECAY=0.8 script/train.sbatch AK
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3; train data without copies of held-out docs (job 536429)
