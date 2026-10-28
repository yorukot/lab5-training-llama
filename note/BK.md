lab5-yoru-BK-552809: all-Sophia (OPTIMIZER=sophia): every parameter on Sophia at lr 2.5e-4, the block matrices and the tied embedding with wd 1.34 (tau 0.39), the norms with wd 0; as the official repo runs it (Sophia on the tied embedding too); Sophia changed from BF-BI per the follow-up research: the curvature EMA's beta2 0.99 -> 0.95 (Wen et al. 2025 found 0.9-0.95 best, 0.99 divergent at 520M) and the curvature pass's gradient no longer clipped (Levanter and Semenov et al. don't clip it; clipping shrinks the curvature once the gradient grows, so steps grow); kept: lr 2.5e-4 (the top of the 1e-4 to 3e-4 the sources give near 1B; BH was stable with it), 25% warmup (Wen: 2,000-step warmups diverged at 300M/520M; Semenov: 25% best), rho 0.04 (win rate already 0.11-0.14 in BH; rho, eps and wd barely mattered in Wen's sweeps), wd 1.34 (tau 0.39), bias-corrected curvature, k 10; everything else as AZ (131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, linear decay over the last 50%, best-fit data from job 538113, 1B non-padding tokens); why: the user asked to cancel BH/BI (both stable, BH 40.8 vs AZ's 30.4 at step 2,292) and apply the research's changes; expected: 3-10% worse than BJ: the embedding was the part that wanted a much larger step (AX/AZ, and BI vs BH), and Sophia moves each entry by at most lr = 2.5e-4 per step, while AdamW at 6.8e-3 moves it by about 1.4e-3 (RMS); better than expected would mean Sophia's per-entry curvature suits the embedding; how it transfers to 6B: only if it beats AZ, which is unlikely (Wen et al. and Semenov et al. find Sophia at best even with AdamW); results: diverged, cancelled at step ~2,800 of 7,632 (scancel after ~16 min on 25a-hgpn087); behind BJ from the start (loss 8.04 vs 7.21 at step 150, eval ppl 482.7 vs 304.3 at step 382), then after the peak lr (step 1,908) the loss rose 4.05 -> 4.25 (steps 1,800-2,700) and grad norm 0.3 -> 1.23; eval ppl 482.7, 130.7, 72.6, 62.7, 61.4, 63.2 at steps 382-2,292 (BJ: 304.3, 85.5, 60.4, 51.1, 45.7, 40.4); sophia_win_rate 0.09-0.17; so all-Sophia at lr 2.5e-4 is both slower (the embedding moves at most 2.5e-4 an entry a step) and unstable (the embedding on Sophia is what BJ does not have)
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "sophia"  # Sophia for every parameter
LEARNING_RATE = 2.5e-4  # Sophia
WEIGHT_DECAY = 1.34  # Sophia matrices
EMBED_WEIGHT_DECAY = 1.34  # the tied embedding (the default, WEIGHT_DECAY)
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
WARMUP_FRACTION = 0.25  # first 25% of the steps: warm up
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,OPTIMIZER=sophia,GLOBAL_BATCH=16,MICRO_BATCH=2,LEARNING_RATE=2.5e-4,WEIGHT_DECAY=1.34,WARMUP_FRACTION=0.25 script/train.sbatch BK

Sophia betas 0.965/0.95, rho 0.04, curvature every 10 steps (its gradient not clipped), max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
