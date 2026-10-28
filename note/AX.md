lab5-yoru-AX-542893 (first try 542579 failed at start on node 25a-hgpn071, its GPU7 unreadable by NVML, node now excluded; second try 542624 failed at DDP init on 25a-hgpn163, NCCL "No route to host", fixed by NCCL_SOCKET_IFNAME=lo in train.sbatch): AU with the AdamW lr (tied embedding, norms) 1.7e-3 -> 3.4e-3 (ADAM_LEARNING_RATE; the embedding's wd 0.2 -> 0.1 so its tau stays 0.39); Muon keeps lr 1.7e-3 and wd 0.2, so only the AdamW params change; everything else as AU (normuon, 131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, linear decay over the last 50%, best-fit data from job 538113, 1B non-padding tokens); why: Muon's step has a fixed size (orthogonalized, then scaled to RMS 0.2) however noisy the gradient is, while AdamW's m/sqrt(v) shrinks when noise dominates (m averages ~10 steps, v sees the full variance); the two were matched at 524k tokens a step, so at 131k Muon likely steps too far relative to the AdamW params (embedding, norms), and AV (65k, same lr) falling far behind fits Muon stepping too far; AW and AX test this from the two sides (AW lowers Muon's lr, AX raises AdamW's); expected: 0 to -1% vs AU's 17.54; the embedding is 21% of the params and is both the input lookup (sparse gradient, rare tokens learn slowly) and the output layer (dense gradient); AR showed its wd does not matter, so lr is the knob left; better means the AdamW params want a larger lr than Muon's at this batch, worse means the shared lr is at or above their best; how it transfers to 6B: the model width is the same, so the AdamW/Muon lr ratio carries over as is; both lrs then scale by ~0.8; results: eval ppl 17.32 (-1.25% vs AU's 17.54), 999,678,772 tokens, 1:07:34 on 8 GPUs (train 66.3 min); best 1B run so far; ahead of AU all the way: -6.9% at 20%, -2.4% at 50%, -1.25% at the end; grad norm lower than AU's through training (~0.11-0.15 vs ~0.15-0.2), as expected when the embedding moves faster; the AdamW params (tied embedding, norms) want a larger lr than Muon's at 131k; same batch run AW (Muon lr 1.7e-3 -> 1.2e-3) got 17.42 (-0.68%): both raise the AdamW/Muon lr ratio, and both help, AX more; not yet bracketed above 3.4e-3; scaled to 6B: the same AdamW/Muon lr ratio (the width does not change), both lrs ~0.8x
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.7e-3  # Muon
ADAM_LEARNING_RATE = 3.4e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.2  # Muon matrices
EMBED_WEIGHT_DECAY = 0.1  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.7e-3,ADAM_LEARNING_RATE=3.4e-3,WEIGHT_DECAY=0.2,EMBED_WEIGHT_DECAY=0.1 script/train.sbatch AX

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3; expected wall time ~67 min
