lab5-yoru-AZ-543673: AU with Muon lr 1.7e-3 -> 1.2e-3 (wd 0.2 -> 0.28) and AdamW lr (tied embedding, norms) 1.7e-3 -> 6.8e-3 (embedding wd 0.2 -> 0.05); every group's tau = 1/(lr * wd * steps) stays 0.39; everything else as AU (normuon, 131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, linear decay over the last 50%, best-fit data from job 538113, 1B non-padding tokens); why: at 131k tokens a step both sides of the AdamW-vs-Muon step-size test helped (AU 17.54; AW, Muon lr 1.2e-3: 17.42; AX, AdamW lr 3.4e-3: 17.32), and neither lr is bracketed yet; AY and AZ both use Muon lr 1.2e-3, so AY vs AX gives the Muon lr effect at AdamW lr 3.4e-3, AY vs AW the AdamW lr effect at Muon lr 1.2e-3, and AZ vs AY whether the AdamW lr's best lies above 3.4e-3; expected: within +-0.5% of AY; better means the AdamW lr's best lies above 3.4e-3, worse means it is near 3.4e-3; risk: 4x AU's AdamW lr may bring loss spikes early in training (watch the grad norm); how it transfers to 6B: this round sets the 6B lrs: 131k tokens a step, beta2 0.9873, both lrs ~0.8x the best 1B values (DeepSeek lr ~ C^-0.125; the AdamW/Muon ratio carries over since the width is the same), wd from tau via the TPP rule; results: eval ppl 17.26 (17.260; -0.33% vs AY's 17.317, -0.36% vs AX's 17.322, -1.57% vs AU's 17.536), 999,678,772 tokens, 1:07:52 on 8 GPUs (25a-hgpn080); best 1B run so far; ahead of AY all the way: -0.88% at 20%, -0.30% at 50%, -0.33% at the end; no loss spikes (warmup grad norm peaks like AY's), grad norm lower than AY's late (~0.12-0.13 vs ~0.14-0.15); the AdamW lr's gain shrinks with each doubling at Muon lr 1.2e-3 (1.7e-3 -> 3.4e-3: -0.61%, 3.4e-3 -> 6.8e-3: -0.33%); a parabola in log lr through AW, AY and AZ puts the best near 1.1e-2, only ~0.06% below AZ, so 6.8e-3 is close enough and the lr search can stop here; scaled to 6B: Muon ~0.96e-3, AdamW ~5.4e-3 (the same 5.67 ratio), wd from tau via the TPP rule
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.2e-3  # Muon
ADAM_LEARNING_RATE = 6.8e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.28  # Muon matrices
EMBED_WEIGHT_DECAY = 0.05  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=0.28,EMBED_WEIGHT_DECAY=0.05 script/train.sbatch AZ

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
