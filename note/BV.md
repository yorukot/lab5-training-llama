lab5-yoru-BV-554676: BR with both lrs x1.5: Muon 1.2e-3 -> 1.8e-3, AdamW (embedding, norms) 6.8e-3 -> 1.02e-2; wd so every group's tau stays 0.39: Muon 0.187, embedding 0.033; everything else as BR (1% warmup, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with BR (17.061) is clean; why: BQ's lead (x1.25) did not shrink during the decay, and AZ's fit put AdamW's best near 1.1e-2, so the best lrs may be higher still; BU (x1.25) and BV (x1.5) next to BR (x1) bracket the best lr with the 1% warmup; expected: about BU's result (-0.2 to -0.4% vs BR); worse than BU means the best is near x1.25; the main risk is early spikes with the short warmup; how it transfers to 6B: the best lr multiplier here sets the 6B base before the x0.8 scaling; results: eval ppl 17.06 (17.058; -0.02% vs BR's 17.061, +0.18% vs BU's 17.027), 1:07:31 on 8 GPUs (25a-hgpn175); no loss spikes (grad norm up to 6.2 in the first 30 steps); level with BR from 10% on, behind BU all the way; so x1.5 is past the best lr (about x1.23, between BU and BV); for 6B: use BU's lrs
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.8e-3  # Muon
ADAM_LEARNING_RATE = 1.02e-2  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.187  # Muon matrices
EMBED_WEIGHT_DECAY = 0.033  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.8  # last 80% of the steps: decay to 0
DECAY_TYPE = "linear"
WARMUP_FRACTION = 0.01
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.8e-3,ADAM_LEARNING_RATE=1.02e-2,WEIGHT_DECAY=0.187,EMBED_WEIGHT_DECAY=0.033,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01 script/train.sbatch BV

warmup_steps 0.01, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
