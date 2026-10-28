lab5-yoru-AB-523179: AA's recipe (Muon lr 6e-4, WSD: warmup 5%, flat, linear decay to 0 over the last 50%, document-masked attention) with MAX_TOKENS 1B -> 2.5B (4,768 steps), the same token count as the AdamW baseline adamw-lr0.0006-2.5B-519334 (cosine to 10%, packed attention, in-training eval ppl 18.31), to measure what the new recipe gives at equal tokens; compare on the OJ score: the in-training eval is now per document (no cross-document attention, BOS not scored), so its ppl is not directly comparable to the baseline's packed eval; results: eval ppl 16.79 (in-training, per document), 1:49:46 on 8 GPUs; best so far, uploaded to yorukot/lab5-llama (HF commit 1c2e2eb); OJ job 65bb772c44a44578b9533306792fefe0: OJ ppl 16.48 (16.48114), ~2% below the in-training 16.79, so the in-training eval is a usable proxy for the OJ (it scores split document tails without their beginning, which makes it slightly pessimistic)
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
LEARNING_RATE = 6e-4  # shared by Muon (update RMS scaled to 0.2) and AdamW
WEIGHT_DECAY = 0.1
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 2_500_000_000  # sbatch --export=ALL,MAX_TOKENS=2500000000 script/train.sbatch AB
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
