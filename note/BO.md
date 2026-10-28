lab5-yoru-BO-553157: BM with twice the batch, 16 -> 32 sequences (262k tokens a step, ~3,816 steps for 1B tokens, MICRO_BATCH 4, no gradient accumulation, as AT); lrs unchanged (Muon 1.2e-3, AdamW 6.8e-3; Marek et al. 2025: lr nearly flat in the batch at a fixed token count); Muon wd 0.28 -> 0.56 and embedding wd 0.05 -> 0.1 so tau = 1/(lr * wd * steps) stays 0.39; AdamW beta2 0.9873 -> 0.9747 (the same half-life in tokens); old data (data/bestfit, job 538113), so BO vs BM is a clean comparison; everything else as BM (normuon, Muon lr 1.2e-3, AdamW lr 6.8e-3, every group's tau 0.39, Muon momentum 0.95, 5% warmup, linear decay over the last 80%, 1B non-padding tokens, seed 42); why: the batch size is the biggest open question for 6B: the best batch grows with the token count (DeepSeek LLM's fit B_opt ~ C^0.33: 6x the tokens -> ~1.8x the batch), so 131k, best at 1B, may be too small at 6B; AT (262k) was +1.0% vs AU (131k), but with the old recipe (before the AdamW lr, tau and decay changes); expected: +0.3 to +1% vs BM's 17.143; how it transfers to 6B: a gap of 0.5% or less at 1B means 262k likely ties or wins at 6x the tokens (and halves the optimizer steps, so a bit faster); 1% or more keeps 131k; in between, a 2B-token pair would decide; results: eval ppl 17.41 (17.405; +1.53% vs BM's 17.143), 53:13 on 8 GPUs (25a-hgpn070); 1.35 steps/s = 354k tokens/s vs BM's 272k (+30%: 4 sequences per GPU use it better, and half the optimizer steps), so ~21% less wall time; worse at the same tokens all the way: +5.2% at 20%, +2.3% at 50%, +1.6% at 75%, +1.53% at the end; the gap is larger than AT vs AU's 1.0% because the recipe (AdamW lr, tau, decay) is now tuned at 131k; the lrs were not retuned at 262k, so part of the gap may be the lr, but it is above the 1% threshold; for 6B: keep 131k (6.7 h, 54 H200-h, fits); the ~1.8x shift in the best batch at 6x the tokens is unlikely to close a 1.5% gap
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.2e-3  # Muon
ADAM_LEARNING_RATE = 6.8e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9747
WEIGHT_DECAY = 0.56  # Muon matrices
EMBED_WEIGHT_DECAY = 0.1  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 32  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.8  # last 80% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=32,MICRO_BATCH=4,ADAM_BETA2=0.9747,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=0.56,EMBED_WEIGHT_DECAY=0.1,DECAY_FRACTION=0.8 script/train.sbatch BO

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
