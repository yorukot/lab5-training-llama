lab5-yoru-BR-554087: BN with the warmup 5% -> 1% of the steps (~76 steps, 10M tokens; WARMUP_FRACTION=0.01), the decay unchanged (last 80%), so the peak lr lasts 19% of the steps instead of 15%; lrs and wd as BN (Muon 1.2e-3 wd 0.28, AdamW 6.8e-3 embedding wd 0.05); everything else as BN (new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with BN (17.151) is clean; why: Muon's update has a fixed size and needs little warmup (modded-nanogpt uses none for its Muon matrices); a 5% warmup spends 380 steps below the peak lr, and at 6B it would be 2,300 steps and 300M tokens; risk: the AdamW embedding with lr 6.8e-3 could spike early with a short warmup; expected: -0.1 to -0.2% vs BN; how it transfers to 6B: the warmup needed is set by the early, noisy steps, not by the run length, so at 6B a 1% warmup (460 steps, still longer than here) is safe and keeps more steps at the peak; results: eval ppl 17.06 (17.061; -0.52% vs BN's 17.151, -0.46% vs BE's 17.140), 1:07:20 on 8 GPUs (25a-hgpn055); best 1B run so far; no loss spikes (grad norm up to 5.5 in the first 30 steps, then falling); ahead all the way: -4.5% at 15%, -2.9% at 20%, -0.91% at 50%, -0.55% at 75% and 90%, -0.52% at the end; the lead held through the decay (unlike BA's), so the 4% more steps at the peak lr pay off; better than expected (-0.1 to -0.2%); for 6B: warm up over the first 1% (~460 steps) or less; next: BR with BQ's higher lrs, and a still shorter warmup
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
WARMUP_FRACTION = 0.01
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=0.28,EMBED_WEIGHT_DECAY=0.05,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01 script/train.sbatch BR

warmup_steps 0.01, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
