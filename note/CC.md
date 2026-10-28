lab5-yoru-CC-557870: BU with the norms' lr x4: 8.5e-3 -> 3.4e-2 (NORM_LEARNING_RATE=3.4e-2); the tied embedding keeps 8.5e-3; everything else as BU (Muon lr 1.5e-3 wd 0.224, AdamW lr 8.5e-3 for the tied embedding with wd 0.04, every group's tau 0.39, 1% warmup, new data from job 552807, normuon, NorMuon beta2 0.95, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with BU (17.027) and CA (17.142, norms x0.5) is clean; why: CA (norm lr halved, 4.25e-3) was worse all the way and +0.68% at the end, so the norms want a high lr and may want a higher one than the embedding's; CB (x2) and CC (x4) are run together so that with CA (x0.5) and BU (x1) four points bracket the best norm lr; the norms have no weight decay, so no tau to keep; the risk is noise: a gain multiplies a whole block's input, so at a too high lr its per-step jitter reaches every activation (loss spikes, a worse end); expected: -0.3% to +0.5% vs BU, likely past the best; better than CB means the norms want even more; how it transfers to 6B: the 6B run would keep the best norm / embedding lr ratio and scale both by x0.8 with the other lrs; results: eval ppl 17.19 (17.189; +0.95% vs BU's 17.027), 1:07:35 on 8 GPUs (25a-hgpn056); worse: ahead early (-3.2% at 5%), behind from 20% on (+1.5-1.7% through the middle, +0.95% at the end); too high, as expected; with CA and CB the norm lr is settled at BU's 8.5e-3; for 6B: keep the norms at the embedding's lr
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
WARMUP_FRACTION = 0.01
NORM_LEARNING_RATE = 3.4e-2
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01,NORM_LEARNING_RATE=3.4e-2 script/train.sbatch CC

warmup_steps 0.01, adam beta1 0.9, norm lr 3.4e-2, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
