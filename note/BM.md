lab5-yoru-BM-552943 (first try 552940 failed at start on 25a-hgpn069: "No space left on device" creating the dataloader's semaphore, i.e. that node's /dev/shm was full; node now excluded): BE rerun unchanged except the now-seeded init (DECAY_FRACTION=0.8); everything else as BE (normuon, Muon lr 1.2e-3 and wd 0.28, AdamW lr 6.8e-3 and embedding wd 0.05, every group's tau 0.39, 131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, 5% warmup, best-fit data from job 538113, 1B non-padding tokens); the init is now seeded (set_seed(42) in training.py, added after BE ran), so BL and BM start from the same weights and see the same data order: BL vs BM is a clean comparison; why: two uses: (1) BE vs BM differ only in the random init (BE's was not seeded), which measures the run-to-run noise we have never measured while deciding on 0.3-0.7% gaps; (2) a seeded baseline of the best recipe for later seeded runs (BL first); expected: within about +-0.3% of BE's 17.14; a larger gap means earlier small differences (e.g. AW/AX, AY/AZ) were within noise; how it transfers to 6B: the noise level tells how far 1B gaps can be trusted when picking the 6B settings; results: eval ppl 17.14 (17.143; +0.02% vs BE's 17.140), 1:07:33 on 8 GPUs (25a-hgpn076); the different random init (plus GPU nondeterminism) moves the curve by up to ~0.8% early (step 1146: 40.66 vs 40.98), under 0.4% from 30% on, and 0.02% at the end; so the final-ppl noise from the init is about 0.02% (one pair; the data order was the same, seed 42, in both), and earlier final gaps of 0.3% and more were real; BM is the seeded baseline for later runs (BL: +0.37%); for 6B: 1B gaps of ~0.1% and up can be trusted when picking settings
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
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=0.28,EMBED_WEIGHT_DECAY=0.05,DECAY_FRACTION=0.8 script/train.sbatch BM

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
