lab5-yoru-BP-554075: the BM/BN recipe scaled to 3B tokens (3x 1B), the midpoint toward the 6B run; new data (job 552807, the same as BN; 3B is a random half of its 6B); ~22,898 steps at 131k tokens (16 sequences, MICRO_BATCH 2), linear decay over the last 80%, 5% warmup, AdamW beta2 0.9873, normuon, Muon momentum 0.95, seed 42; lrs x0.87 (= 3 ** -0.125, DeepSeek LLM's fit eta_opt ~ C^-0.125, compute 3x): Muon 1.2e-3 -> 1.05e-3, AdamW (embedding, norms) 6.8e-3 -> 5.9e-3; wd set so every group's tau = 1/(lr * wd * steps) stays 0.39 (Wang & Aitchison 2024: the best tau as a fraction of training is about constant across dataset sizes): Muon 0.107, embedding 0.019; why: the user asked for a 3B try before the 6B run; it checks that the recipe and the scaling rule hold over 3x the steps (no instability, the decay still pays off), and with BN (1B) gives a second point on the tokens-vs-ppl curve to project 6B; one run cannot tell whether x0.87 is the best lr (that needs a pair, e.g. a second 3B at x1.0); expected: eval ppl 12-13 (AC 1B -> AB 2.5B fell as tokens^-0.29: 17.15 * 3^-0.29 = 12.4; diminishing returns put it at or above that), ~3.3 h, ~27 H200-h; how it transfers to 6B: the 6B run uses the same rule (lrs x0.8, tau 0.39); if the 3B curve and the 1B-3B fit look right, the 6B run follows; if 3B falls short of the projection, the lr rule is the first suspect; results: eval ppl 13.77 (13.771; -19.7% vs BN's 17.151 at 1B), 3:09:23 on 8 GPUs (25a-hgpn159), ~25 H200-h, 2.08 steps/s as at 1B; worse than expected (12-13): 3x the tokens gave ppl x0.803, i.e. ppl ~ tokens^-0.20, not the -0.29 of AC -> AB (AC's untuned lr inflated that drop); stable, no spikes; the decay (last 80%) took it from 17.86 (50%) to 13.77; projection for 6B with this recipe: ppl x0.87 (tokens^-0.20) to x0.90 (a Chinchilla-shaped curve, data exponent 0.28, fitted to BN and BP): about 12.0-12.4; caveats: BP predates BR/BU (1% warmup, lrs x1.25: -0.72% at 1B), and it kept tau 0.39, which Power Lines (Bergsma et al. 2025: best tau ~ (tokens per parameter)^-0.5) puts at ~0.225 for 3B; for 6B: the 6B recipe should take BU's changes, and the tau (and lr) rule needs a 3B check before the 6B run; OJ: 13.935 (job 9e22b54d7aac471f95ff5bbe3a2d2046, HF commit 786e8b9; +1.2% vs the in-training 13.771): the OJ scores all 50k held-out docs (avg 1,245 tokens vs 1,265 in our 10k) with the model in bf16 (weights too), while our eval keeps fp32 weights under bf16 autocast
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.05e-3  # Muon
ADAM_LEARNING_RATE = 5.9e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.107  # Muon matrices
EMBED_WEIGHT_DECAY = 0.019  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 3_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.8  # last 80% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=3000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.05e-3,ADAM_LEARNING_RATE=5.9e-3,WEIGHT_DECAY=0.107,EMBED_WEIGHT_DECAY=0.019,DECAY_FRACTION=0.8 script/train.sbatch BP

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~3.3 h
