lab5-yoru-BE-552757: AZ with the lr decay over the last 80% of the steps instead of 50% (DECAY_FRACTION=0.8: 5% warmup, 15% at the peak, then linear to 0); everything else as AZ (normuon, Muon lr 1.2e-3 and wd 0.28, AdamW lr 6.8e-3 and embedding wd 0.05, every group's tau 0.39, 131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, best-fit data from job 538113, 1B non-padding tokens); code: the optimizer split (src/optimizer.py with adamw.py, muon.py, sophia.py), checked against AZ by AZ-check (552717); why: BA and BB both led AZ by 2-5% mid-training and lost it all in the decay, so the decay phase sets most of the final loss; Bergsma et al. 2025 (Straight to Zero) find a linear decay to 0 over the whole run beats shorter cooldowns, more so the more tokens per parameter; AQ changed the decay's shape (1-sqrt, worse), not its length; expected: -0.2 to -0.5% vs AZ; worse means the 50% decay is long enough at 1B; how it transfers to 6B: the 6B run has 6x the tokens per parameter, where a long decay should help at least as much; results: eval ppl 17.14 (17.140; -0.70% vs AZ's 17.260), 1:07:39 on 8 GPUs (25a-hgpn177); best 1B run so far; level with AZ until its decay starts (20%), then ahead: -8.9% at 50%, -5.5% at 75%, -1.4% at 95% (mid-run leads come from its lower lr there, as with BA), -0.70% at the end; so the longer decay helps, unlike BA/BB whose mid-run leads vanished; caveat: the init was not seeded (set_seed(42) came later), and the run-to-run noise is not measured yet; a 10% decay is not needed (longer helps); next: decay over all steps after warmup (0.95, Straight to Zero's D2Z); for 6B: decay over the last 80% (or longer if 0.95 wins)
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
DECAY_FRACTION = 0.8  # last 80% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=0.28,EMBED_WEIGHT_DECAY=0.05,DECAY_FRACTION=0.8 script/train.sbatch BE

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
