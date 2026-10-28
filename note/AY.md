lab5-yoru-AY-543672: AU with both of AW's and AX's changes: Muon lr 1.7e-3 -> 1.2e-3 (wd 0.2 -> 0.28) and AdamW lr (tied embedding, norms) 1.7e-3 -> 3.4e-3 (embedding wd 0.2 -> 0.1); every group's tau = 1/(lr * wd * steps) stays 0.39; everything else as AU (normuon, 131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, linear decay over the last 50%, best-fit data from job 538113, 1B non-padding tokens); why: at 131k tokens a step both sides of the AdamW-vs-Muon step-size test helped (AU 17.54; AW, Muon lr 1.2e-3: 17.42; AX, AdamW lr 3.4e-3: 17.32), and neither lr is bracketed yet; AY and AZ both use Muon lr 1.2e-3, so AY vs AX gives the Muon lr effect at AdamW lr 3.4e-3, AY vs AW the AdamW lr effect at Muon lr 1.2e-3, and AZ vs AY whether the AdamW lr's best lies above 3.4e-3; expected: about 17.15-17.25 (-1.7 to -2.2% vs AU) if the two gains add; close to AX's 17.32 means the gains overlap (both come from the AdamW/Muon lr ratio, here 2.83); how it transfers to 6B: this round sets the 6B lrs: 131k tokens a step, beta2 0.9873, both lrs ~0.8x the best 1B values (DeepSeek lr ~ C^-0.125; the AdamW/Muon ratio carries over since the width is the same), wd from tau via the TPP rule; results: eval ppl 17.32 (17.317; -0.03% vs AX's 17.322, -0.61% vs AW's 17.423, -1.25% vs AU's 17.536), 999,678,772 tokens, 1:07:35 on 8 GPUs (25a-hgpn051); the two gains overlap: with the AdamW lr at 3.4e-3, lowering Muon's lr 1.7e-3 -> 1.2e-3 helps early (-0.70% vs AX at 20%, -0.11% at 50%) and not at the end; so AW's gain came from the AdamW/Muon lr ratio, and Muon's lr is flat between 1.2e-3 and 1.7e-3; the AdamW lr effect at Muon lr 1.2e-3 (AY vs AW) is -0.61%, half the -1.25% it gave at Muon lr 1.7e-3 (AX vs AU), since the lower Muon lr had already taken part of the same ratio gain; grad norm ~0.14-0.15 late, no spikes; same batch run AZ (AdamW lr 6.8e-3) got 17.26
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.2e-3  # Muon
ADAM_LEARNING_RATE = 3.4e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.28  # Muon matrices
EMBED_WEIGHT_DECAY = 0.1  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=3.4e-3,WEIGHT_DECAY=0.28,EMBED_WEIGHT_DECAY=0.1 script/train.sbatch AY

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
