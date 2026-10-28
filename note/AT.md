lab5-yoru-AT-541927 (first try 541883 failed at DDP init on node 25a-hgpn150, NCCL "No route to host"; node now excluded): AS with the batch 64 -> 32 sequences (262k tokens a step, 3,816 steps for 1B tokens, MICRO_BATCH 4, no gradient accumulation), wd 0.8 -> 0.4 so tau = 1/(lr * wd * steps) stays 0.39 (lr 1.7e-3 unchanged), AdamW beta2 0.95 -> 0.9747 (= 0.95 ** (32/64), the same half-life in tokens, ~7.1M), Muon momentum 0.95 unchanged, everything else as AS (normuon, linear decay over the last 50%, best-fit data from job 538113, 1B non-padding tokens); why: Marek et al. 2025 (arXiv 2507.07101, NeurIPS 2025) find that at a fixed token count small batches (down to 1 sequence) match or beat large ones, with beta2 scaled to keep its half-life in tokens, beta1 unchanged and lr nearly flat in the batch; at 1.3B params / 10B tokens their untuned batch-1 Adam beat AdamW at ~1M tokens a step; the lab caps tokens, not compute (the 6B run needs ~38 of the 64 H200-h), so steps are cheap for us; caveats: their baselines are AdamW, and Muon's edge is at large batches, so our gain is likely smaller; the batch with no gradient accumulation (8 GPUs x 4 sequences): same tokens per second, twice the steps; expected: -1 to -3% vs AS's 18.25 (about 17.7-18.1); how it transfers to 6B: the paper's 1.3B / 10B-token case is close to our 6B run, so a batch that wins at 1B should not lose at 6B (the batch where smaller stops helping grows with the token count); the 6B run uses the smallest winning batch whose wall time (~6x this run's) fits in 64 H200-h, keeping tau via the TPP rule and beta2 via the half-life rule; results: eval ppl 17.71 (-2.96% vs AS's 18.25), 998,368,242 tokens, 0:53:33 on 8 GPUs (train 51.9 min, 0.75 s a step, ~350k tokens/s vs AS's ~400k); ahead of AS all the way: -26% at 15%, -3.3% at 50%, -2.6% at 70%, and the gap held through the decay; smaller batches help at 1B even with Muon, and halving the batch with tau and beta2 kept is a free win (same lr); same batch run AU (131k) got 17.54, AV (65k) fell far behind; scaled to 6B: ~6x this run's wall time, ~5.3 h, ~43 H200-h, well inside 64
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.7e-3
ADAM_LEARNING_RATE = 1.7e-3  # the AdamW params (embedding, norms); default LEARNING_RATE
ADAM_BETA2 = 0.9747
WEIGHT_DECAY = 0.4
EMBED_WEIGHT_DECAY = 0.4  # the tied embedding; the other matrices use WEIGHT_DECAY
SEQ_LEN = 8192
GLOBAL_BATCH = 32  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.7e-3,WEIGHT_DECAY=0.4,MUON_VARIANT=normuon,GLOBAL_BATCH=32,MICRO_BATCH=4,ADAM_BETA2=0.9747 script/train.sbatch AT

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0 (AS's grad norm sits at 0.15-0.3, ~sqrt(64/32)x that here), bf16, liger kernel, FA3; expected wall time ~53 min
