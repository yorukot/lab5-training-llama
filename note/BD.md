lab5-yoru-BD-552756: AZ with Muon's momentum 0.95 -> 0.98 (MUON_MOMENTUM=0.98, a new env var read in src/muon.py); linear decay over the last 50% as AZ; everything else as AZ (normuon, Muon lr 1.2e-3 and wd 0.28, AdamW lr 6.8e-3 and embedding wd 0.05, every group's tau 0.39, 131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, best-fit data from job 538113, 1B non-padding tokens); code: the optimizer split (src/optimizer.py with adamw.py, muon.py, sophia.py), checked against AZ by AZ-check (552717); why: the momentum averages about 1/(1-0.95) = 20 steps; at 131k tokens a step that is 2.6M tokens, a quarter of what it averaged at the 524k batch it was set for (Moonlight and modded-nanogpt use 0.95 at 0.5M+ token batches); AdamW's beta2 was rescaled for the smaller batch (AU), Muon's momentum never was; 0.98 averages 50 steps = 6.5M tokens; expected: within +-0.5% of AZ's 17.26: more averaging lowers the noise in the direction Muon orthogonalizes, but the momentum lags more; better means the small batch wants a longer momentum; how it transfers to 6B: the 6B run uses the same batch, so the momentum carries over as is; results: eval ppl 17.41 (17.412; +0.88% vs AZ's 17.260), 1:07:32 on 8 GPUs (25a-hgpn012); worse: behind AZ all the way, most early (+27% at 5%, +4.0% at 20%), then +0.5% from 50% on and +0.88% at the end; the longer momentum lags more than it gains from averaging at 131k tokens a step; Muon momentum stays 0.95; caveat: the init was not seeded (set_seed(42) came later), so AZ and BD also differ in their random init; for 6B: keep 0.95
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.2e-3  # Muon
ADAM_LEARNING_RATE = 6.8e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.28  # Muon matrices
EMBED_WEIGHT_DECAY = 0.05  # the tied embedding
MUON_MOMENTUM = 0.98
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=0.28,EMBED_WEIGHT_DECAY=0.05,MUON_MOMENTUM=0.98 script/train.sbatch BD

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.98, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
