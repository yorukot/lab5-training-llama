lab5-yoru-BJ-552808: BH with the research's Sophia changes: Sophia+AdamW (OPTIMIZER=sophia+adamw), Sophia lr 2.5e-4, wd 1.34; the embedding and norms on AdamW as AZ (lr 6.8e-3, embedding wd 0.05, beta2 0.9873; BI showed the lower 1.7e-3 is worse); Sophia changed from BF-BI per the follow-up research: the curvature EMA's beta2 0.99 -> 0.95 (Wen et al. 2025 found 0.9-0.95 best, 0.99 divergent at 520M) and the curvature pass's gradient no longer clipped (Levanter and Semenov et al. don't clip it; clipping shrinks the curvature once the gradient grows, so steps grow); kept: lr 2.5e-4 (the top of the 1e-4 to 3e-4 the sources give near 1B; BH was stable with it), 25% warmup (Wen: 2,000-step warmups diverged at 300M/520M; Semenov: 25% best), rho 0.04 (win rate already 0.11-0.14 in BH; rho, eps and wd barely mattered in Wen's sweeps), wd 1.34 (tau 0.39), bias-corrected curvature, k 10; everything else as AZ (131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, linear decay over the last 50%, best-fit data from job 538113, 1B non-padding tokens); why: the user asked to cancel BH/BI (both stable, BH 40.8 vs AZ's 30.4 at step 2,292) and apply the research's changes; expected: at or below BH's curve at the same steps (BH: eval 46.3 at step 1,910, 40.8 at 2,292), final 18-19.5 (4-12% worse than AZ's 17.26); if it is clearly below BH, the two changes help Sophia's stability or tracking; how it transfers to 6B: only if it beats AZ, which is unlikely (Wen et al. and Semenov et al. find Sophia at best even with AdamW); results: eval ppl 19.19 (19.187; +11.2% vs AZ's 17.260, +11.9% vs BE's 17.140), 1:02:22 on 8 GPUs (25a-hgpn076); stable all the way (no rise after the peak lr); vs BH (same but beta2 0.99 and the curvature gradient clipped, cancelled at step ~2,650) only -1% at step 2,292 (40.4 vs 40.8); behind AZ +44% at 20%, +18% at 50%, +12.8% at 75%, +11.2% at the end; sophia_win_rate fell to 0.025-0.03 after step 2,000 (97% of entries clipped, i.e. close to sign-momentum, Lion-like): the faster curvature EMA (beta2 0.95) sits below its mean most of the time; matches the research: Sophia is at best even with AdamW, and Muon is far ahead here; 2.4 it/s (no Newton-Schulz, curvature pass included); for 6B: not used
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "sophia+adamw"  # Sophia for the block matrices, AdamW for the embedding and norms
LEARNING_RATE = 2.5e-4  # Sophia
ADAM_LEARNING_RATE = 6.8e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 1.34  # Sophia matrices
EMBED_WEIGHT_DECAY = 0.05  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
WARMUP_FRACTION = 0.25  # first 25% of the steps: warm up
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,OPTIMIZER=sophia+adamw,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=2.5e-4,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=1.34,EMBED_WEIGHT_DECAY=0.05,WARMUP_FRACTION=0.25 script/train.sbatch BJ

adam beta1 0.9, Sophia betas 0.965/0.95, rho 0.04, curvature every 10 steps (its gradient not clipped), max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
