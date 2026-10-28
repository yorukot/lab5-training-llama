lab5-yoru-BB-544151: AZ with each block's output projections zero-initialized (ZERO_INIT=1: o_proj and down_proj start at 0, the rest keep the default std 0.02), so every block starts as the identity; everything else as AZ (normuon, Muon lr 1.2e-3 and wd 0.28, AdamW lr 6.8e-3 and embedding wd 0.05, every group's tau 0.39, 131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, linear decay over the last 50%, best-fit data from job 538113, 1B non-padding tokens); why: the lrs and the batch are now tuned (AZ 17.26), and 17 needs another ~1.5%; with std 0.02 everywhere, each of the 16 blocks adds a random function of its input to the residual stream at the start; modded-nanogpt zero-inits these projections; the gradient into o_proj and down_proj is not zero, and Muon gives a zero matrix a full-size update on step 1, so the blocks switch on at once (a local test on a tiny model confirmed every weight gets a gradient by step 2 and the loss falls); expected: -0.2 to -0.5% vs AZ, most of it early in training; how it transfers to 6B: it mainly speeds up the start, so the gain at 6B is likely smaller than at 1B, but it costs nothing; results: eval ppl 17.34 (17.341; +0.47% vs AZ's 17.260), 999,678,772 tokens, 1:07:21 on 8 GPUs (25a-hgpn086); worse: the faster start does not last: vs AZ -4.4% at 10%, -2.35% at 20%, -0.80% at 50%, level at 75%, +0.42% at 95%, +0.47% at the end; grad norm above AZ's through the first half (~0.16-0.19 vs ~0.13-0.15), level late; so the default init (std 0.02) ends better here; a guess, not checked: with Muon and wd, the blocks that start at 0 settle into a different balance between the embedding and the block outputs; for 6B: a gain that shrinks and then reverses with training would only be worse over 6x the tokens, so zero init is dropped; same batch run BA (cautious wd) was far ahead of AZ mid-training
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.2e-3  # Muon
ADAM_LEARNING_RATE = 6.8e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.28  # Muon matrices
EMBED_WEIGHT_DECAY = 0.05  # the tied embedding
CAUTIOUS_WD = False
ZERO_INIT = True  # o_proj and down_proj start at 0
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=0.28,EMBED_WEIGHT_DECAY=0.05,ZERO_INIT=1 script/train.sbatch BB

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
