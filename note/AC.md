lab5-yoru-AC-523180: AB with MAX_TOKENS 2.5B -> 1B (1,907 steps, the budget AA planned; AA was never run), to see how close a 1B run gets to AB, i.e. whether 1B runs are a good enough proxy to pick the lr in the ladder before spending 2.5B+ tokens per run; results: eval ppl 21.97 vs AB's 16.79, 0:47:44 on 8 GPUs
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
LEARNING_RATE = 6e-4  # shared by Muon (update RMS scaled to 0.2) and AdamW
WEIGHT_DECAY = 0.1
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # sbatch --export=ALL,MAX_TOKENS=1000000000 script/train.sbatch AC
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
