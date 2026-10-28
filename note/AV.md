lab5-yoru-AV-541885: AS with the batch 64 -> 8 sequences (65k tokens a step, 15,265 steps for 1B tokens, MICRO_BATCH 1, no gradient accumulation), wd 0.8 -> 0.1 so tau = 1/(lr * wd * steps) stays 0.39 (lr 1.7e-3 unchanged), AdamW beta2 0.95 -> 0.9936 (= 0.95 ** (8/64), the same half-life in tokens, ~7.1M), Muon momentum 0.95 unchanged, everything else as AS (normuon, linear decay over the last 50%, best-fit data from job 538113, 1B non-padding tokens); why: Marek et al. 2025 (arXiv 2507.07101, NeurIPS 2025) find that at a fixed token count small batches (down to 1 sequence) match or beat large ones, with beta2 scaled to keep its half-life in tokens, beta1 unchanged and lr nearly flat in the batch; at 1.3B params / 10B tokens their untuned batch-1 Adam beat AdamW at ~1M tokens a step; the lab caps tokens, not compute (the 6B run needs ~38 of the 64 H200-h), so steps are cheap for us; caveats: their baselines are AdamW, and Muon's edge is at large batches, so our gain is likely smaller; one sequence per GPU, the smallest batch without changing the model or SEQ_LEN; expected: flat to slightly better than AU; it also shows the per-step cost of Muon (Newton-Schulz runs on every GPU, ~0.1-0.15 s a step, not yet sharded), which decides whether this batch fits the 6B time budget; how it transfers to 6B: the paper's 1.3B / 10B-token case is close to our 6B run, so a batch that wins at 1B should not lose at 6B (the batch where smaller stops helping grows with the token count); the 6B run uses the smallest winning batch whose wall time (~6x this run's) fits in 64 H200-h, keeping tau via the TPP rule and beta2 via the half-life rule; results: eval ppl 21.48 (+17.7% vs AS's 18.25, +22.5% vs AU's 17.54; final eval at step 15264 from wandb, the log stops at step 14516's 22.02), 999,678,772 tokens, 1:36:29 on 8 GPUs (train 95.1 min, 0.37 s a step, ~175k tokens/s); much worse: ahead only at 5% (91.6 vs AS's 373.7), behind from 15% on and the gap grew through the stable phase (+12% at 20%, +35% at 50%); the decay won back only part of it (+29% at 65%, +18% at the end); grad norm stayed at 0.25-0.3 with no spikes or clipping, so not instability; most likely lr 1.7e-3 is too high at 8 sequences a step (Muon's step size does not shrink with gradient noise), and maybe Muon momentum 0.95 averages too few tokens (~0.9M vs ~7M in AS); per-step cost ~0.19 s on top of the forward/backward, so 6B at this batch would take ~9.5 h (~76 H200-h), over the 64 cap unless Newton-Schulz is sharded; for 6B: do not use 65k at this lr; AW (Muon lr 1.2e-3 at 131k) tests the lr side
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.7e-3
ADAM_LEARNING_RATE = 1.7e-3  # the AdamW params (embedding, norms); default LEARNING_RATE
ADAM_BETA2 = 0.9936
WEIGHT_DECAY = 0.1
EMBED_WEIGHT_DECAY = 0.1  # the tied embedding; the other matrices use WEIGHT_DECAY
SEQ_LEN = 8192
GLOBAL_BATCH = 8  # sequences per optimizer step
MICRO_BATCH = 1  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.7e-3,WEIGHT_DECAY=0.1,MUON_VARIANT=normuon,GLOBAL_BATCH=8,MICRO_BATCH=1,ADAM_BETA2=0.9936 script/train.sbatch AV

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0 (AS's grad norm sits at 0.15-0.3, ~sqrt(64/8)x that here), bf16, liger kernel, FA3; expected wall time ~80 min
