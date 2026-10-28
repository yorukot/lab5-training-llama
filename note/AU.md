lab5-yoru-AU-541884: AS with the batch 64 -> 16 sequences (131k tokens a step, 7,632 steps for 1B tokens, MICRO_BATCH 2, no gradient accumulation), wd 0.8 -> 0.2 so tau = 1/(lr * wd * steps) stays 0.39 (lr 1.7e-3 unchanged), AdamW beta2 0.95 -> 0.9873 (= 0.95 ** (16/64), the same half-life in tokens, ~7.1M), Muon momentum 0.95 unchanged, everything else as AS (normuon, linear decay over the last 50%, best-fit data from job 538113, 1B non-padding tokens); why: Marek et al. 2025 (arXiv 2507.07101, NeurIPS 2025) find that at a fixed token count small batches (down to 1 sequence) match or beat large ones, with beta2 scaled to keep its half-life in tokens, beta1 unchanged and lr nearly flat in the batch; at 1.3B params / 10B tokens their untuned batch-1 Adam beat AdamW at ~1M tokens a step; the lab caps tokens, not compute (the 6B run needs ~38 of the 64 H200-h), so steps are cheap for us; caveats: their baselines are AdamW, and Muon's edge is at large batches, so our gain is likely smaller; 4x the steps of AS; expected: another -0.5 to -1.5% vs AT, or flat if 262k is already below the batch where smaller stops helping; how it transfers to 6B: the paper's 1.3B / 10B-token case is close to our 6B run, so a batch that wins at 1B should not lose at 6B (the batch where smaller stops helping grows with the token count); the 6B run uses the smallest winning batch whose wall time (~6x this run's) fits in 64 H200-h, keeping tau via the TPP rule and beta2 via the half-life rule; results: eval ppl 17.54 (-3.89% vs AS's 18.25, -0.96% vs AT's 17.71), 999,678,772 tokens, 1:07:11 on 8 GPUs (train 65.6 min, 0.48 s a step, ~275k tokens/s); best 1B run so far; ahead of AS all the way: -7.7% at 20%, -4.6% at 50%, -3.5% at 75%, and the gap held through the decay; each halving of the batch helped less (524k -> 262k -3.0%, 262k -> 131k -1.0%), and 65k (AV) at the same lr fell far behind, so the best batch at lr 1.7e-3 is near 131k; scaled to 6B: ~6x this run's wall time, ~6.7 h, ~54 H200-h, inside 64 with ~10 H200-h to spare; the batch where smaller stops helping grows with the token count, so at 6B 131k vs 262k likely matters less than here
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.7e-3
ADAM_LEARNING_RATE = 1.7e-3  # the AdamW params (embedding, norms); default LEARNING_RATE
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.2
EMBED_WEIGHT_DECAY = 0.2  # the tied embedding; the other matrices use WEIGHT_DECAY
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.7e-3,WEIGHT_DECAY=0.2,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873 script/train.sbatch AU

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0 (AS's grad norm sits at 0.15-0.3, ~sqrt(64/16)x that here), bf16, liger kernel, FA3; expected wall time ~62 min
