lab5-yoru-BQ-554086: BN with both lrs x1.25: Muon 1.2e-3 -> 1.5e-3, AdamW (embedding, norms) 6.8e-3 -> 8.5e-3; wd scaled down so every group's tau stays 0.39: Muon 0.28 -> 0.224, embedding 0.05 -> 0.04; 5% warmup; everything else as BN (new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with BN (17.151) is clean; why: the lrs were tuned with the decay over the last 50% (AX, AY, AZ); with 80% the peak lasts only 15% of the steps and the lr averaged over the run is lower, so the best peak lr may have moved up; AZ's fit put AdamW's best near 1.1e-2, and Muon was flat between 1.2e-3 and 1.7e-3; expected: -0.1 to -0.3% vs BN; worse means the lrs are already at their best; how it transfers to 6B: if higher wins at 1B with the long decay, the 6B lrs should start from the higher values before the x0.8 scaling (Muon 1.2e-3, AdamW 6.8e-3 for 6B); results: eval ppl 17.10 (17.098; -0.31% vs BN's 17.151), 1:08:05 on 8 GPUs (25a-hgpn054); better, as expected: a steady lead from the end of the warmup on (-0.36% at 20%, -0.25% at 50%, -0.24% at 75%, -0.31% at 90% and at the end), so with the 80% decay the best lrs are above 1.2e-3 / 6.8e-3; the edge did not shrink during the decay, so x1.5 may still help; for 6B: start from Muon 1.5e-3 and AdamW 8.5e-3 (or the next winner) before the scaling to 6B
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.5e-3  # Muon
ADAM_LEARNING_RATE = 8.5e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.224  # Muon matrices
EMBED_WEIGHT_DECAY = 0.04  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.8  # last 80% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8 script/train.sbatch BQ

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
