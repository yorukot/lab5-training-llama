lab5-yoru-BL-552939: BE with the lr decay over all the steps after warmup (DECAY_FRACTION=0.95: 5% warmup, then linear to 0, no stable phase), i.e. Straight to Zero's linear decay to zero (Bergsma et al. 2025); everything else as BE (normuon, Muon lr 1.2e-3 and wd 0.28, AdamW lr 6.8e-3 and embedding wd 0.05, every group's tau 0.39, 131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, 5% warmup, best-fit data from job 538113, 1B non-padding tokens); the init is now seeded (set_seed(42) in training.py, added after BE ran), so BL and BM start from the same weights and see the same data order: BL vs BM is a clean comparison; why: BE showed a longer decay helps (50% -> 80%: -0.70%); Straight to Zero finds the full linear decay best, more so the more tokens per parameter; expected: -0.2 to -0.5% vs BM (BE seeded); worse means 80% is about the best length; how it transfers to 6B: 6x the tokens per parameter, where a long decay should help at least as much; the 6B run would use the better of 0.8 and 0.95; results: eval ppl 17.21 (17.206; +0.37% vs BM's 17.143), 1:07:33 on 8 GPUs (25a-hgpn012); worse: ahead of BM while its lr is lower (-2.8% at 20%, -1.4% at 50%, -0.7% at 75%), even at 90% (-0.06%), then behind (+0.21% at 95%, +0.37% at the end); the 15% of steps at the peak lr pay off at the end; a parabola through 0.5 (AZ 17.260), 0.8 (BM 17.143) and 0.95 (BL 17.206) puts the best decay at ~0.76, only ~0.02% better than 0.8, so the decay length is settled at 1B; for 6B: decay over the last 80% (the Straight to Zero argument that longer decays win at more tokens per parameter is untested here, and 0.8 vs 0.95 is only 0.37% at 1B)
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
DECAY_FRACTION = 0.95  # all steps after the 5% warmup: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=0.28,EMBED_WEIGHT_DECAY=0.05,DECAY_FRACTION=0.95 script/train.sbatch BL

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
