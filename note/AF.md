lab5-yoru-AF-535007: AC with OPTIMIZER muon -> adamw (lr 6e-4, same WSD schedule and document-masked attention), the AdamW control at 1B tokens (1,907 steps); why: AB vs the AdamW baseline (16.79 vs 18.31) mixes three changes (optimizer, schedule, document masking) and a different eval, so this isolates what Muon itself gives; expected: higher eval ppl than AC's 21.97 (Moonlight reports ~2x compute efficiency for Muon over AdamW); caveat: 6e-4 is tuned for neither optimizer, so this is a rough comparison, not AdamW at its best lr; wandb logs optim adamw_torch_fused for every run (Muon is passed in separately), so this note is the record that AF ran AdamW; the first attempt (534950) failed before training: NCCL "No route to host" during DDP init on node 25a-hgpn146, now in train.sbatch's --exclude; results: eval ppl 23.97 (+9.1% vs AC's 21.97, i.e. Muon is 8.3% lower ppl than AdamW at lr 6e-4), 0:44:12 on 8 GPUs (train_runtime ~7% shorter than AD: no Newton-Schulz)
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "adamw"  # sbatch --export=ALL,MAX_TOKENS=1000000000,OPTIMIZER=adamw script/train.sbatch AF
LEARNING_RATE = 6e-4  # shared by Muon (update RMS scaled to 0.2) and AdamW
WEIGHT_DECAY = 0.1
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
