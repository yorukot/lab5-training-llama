lab5-yoru-AA-<job id>: first run after the AdamW baseline: OPTIMIZER adamw -> muon (src/muon.py: Muon on the 2D weights in the blocks, Nesterov 0.95, Newton-Schulz 5 in bf16, update RMS scaled to 0.2 (Moonlight) so it shares AdamW's lr and weight decay; AdamW on the tied embedding and the norms); LR schedule cosine to 10% -> warmup 5%, flat, linear decay to 0 over the last 50% (gpt2 AD's trapezoid); document-masked attention (collator finds documents by EOS, FA3 varlen via cu_seq_lens, first token of each document not scored), matching the OJ scoring every document alone; MAX_TOKENS 1B (1,907 steps) as the first rung of the lr ladder (AB AdamW control, AC/AD Muon lr 1.2e-3/2.4e-3); results: not run; the same recipe runs as AB (2.5B tokens) and AC (1B tokens) instead
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
LEARNING_RATE = 6e-4  # shared by Muon (update RMS scaled to 0.2) and AdamW
WEIGHT_DECAY = 0.1
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # lab cap is 6B
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
