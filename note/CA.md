lab5-yoru-CA-557110: BU with the norms' lr halved: 8.5e-3 -> 4.25e-3 (NORM_LEARNING_RATE, a new env var read in src/training.py, default ADAM_LEARNING_RATE); the tied embedding keeps 8.5e-3; the norms have no weight decay, so no tau to keep; everything else as BU (Muon lr 1.5e-3 wd 0.224, AdamW lr 8.5e-3 embedding wd 0.04, every group's tau 0.39, 1% warmup, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with BU (17.027) is clean; why: the norms share the AdamW lr with the embedding, and AX/AZ raised it 1.7e-3 -> 6.8e-3 (then BU 8.5e-3) for both together, so we never learned which of the two wanted the high lr; modded-nanogpt gives its scalars a far lower lr than its embedding; a norm gain multiplies a whole block's input, so at a high lr its per-step noise reaches every activation; expected: -0.2 to +0.2% vs BU, direction unknown; worse means the norms want the high lr too and the shared lr stays; how it transfers to 6B: the 6B run would keep the norm / embedding lr ratio and scale both by x0.8; results: eval ppl 17.14 (17.142; +0.68% vs BU's 17.027), 1:07:11 on 8 GPUs (25a-hgpn141); worse all the way, most early: +7.1% at 5%, +3.1% at 15%, +1.3% at 30%, +1.0% at 45%, +0.8% at 75%, +0.68% at the end; the norms want the high lr (a gain has to move far from its init of 1, and Adam's step is at most about lr), so part of AX/AZ's gain from the higher AdamW lr was the norms'; next: the other side, norm lr x2 (1.7e-2) with the embedding at 8.5e-3; for 6B: keep the norms at least at the embedding's lr
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
NORM_LEARNING_RATE = 4.25e-3
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01,NORM_LEARNING_RATE=4.25e-3 script/train.sbatch CA

warmup_steps 0.01, adam beta1 0.9, norm lr 4.25e-3, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
