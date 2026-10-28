lab5-yoru-BA-544150: AZ with cautious weight decay (CAUTIOUS_WD=1, Chen et al., arXiv 2510.12402) on every decayed group (Muon matrices, tied embedding): decay only the entries whose update has their sign; the wd values stay AZ's, as the paper keeps the baseline's, so with about half the entries decayed the effective tau goes from 0.39 to ~0.8; everything else as AZ (normuon, Muon lr 1.2e-3 and wd 0.28, AdamW lr 6.8e-3 and embedding wd 0.05, every group's tau 0.39, 131k tokens a step = 16 sequences, 7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, linear decay over the last 50%, best-fit data from job 538113, 1B non-padding tokens); why: the lrs and the batch are now tuned (AZ 17.26), and 17 needs another ~1.5%; plain decay pulls every weight toward 0, also where the update pushes it away from 0, so it works against the loss there; cautious decay only acts where the two agree, keeping the shrinking without biasing the loss's minimum (the paper's bilevel view); the paper reports gains with Muon (OLMo-1B, 100B tokens: loss 2.51 -> 2.42) and AdamW (338M-2B at 20 tokens a param, about -0.8% ppl); a local test on a tiny model confirmed the rule (p -= lr * (u + wd * [u * p >= 0] * p)) and that about 50% of the entries get decayed; expected: -0.3 to -1% vs AZ; the effective tau doubling may cost a little, but AN showed tau 0.27-0.55 is flat; how it transfers to 6B: the paper's largest gain is at the longest training (100 tokens a param), which suggests 6B (4.8 tokens a param vs 0.8 here) keeps it; the wd values still come from tau via the TPP rule; results: eval ppl 17.27 (17.267; +0.04% vs AZ's 17.260, the same), 999,678,772 tokens, 1:09:24 on 8 GPUs (25a-hgpn085); far ahead mid-training, level at the end: vs AZ -1.8% at 20%, -5.0% at 50%, -4.0% at 75%, -0.61% at 95%, +0.04% at the end; grad norm below AZ's late (~0.10 vs ~0.12-0.13); this is what a weaker decay looks like: with about half the entries decayed the effective tau is ~0.8, a longer tau gives a lower loss in the stable phase, and the lr decay evens it out (AN: tau 0.27-0.55 about flat at the end); so at 1B cautious wd acts like plain wd with a longer tau, no gain; for 6B: the paper's gain was at 100 tokens a param, and at 4.8 it is unknown; with no 1B gain it is not used for 6B; same batch run BB (zero init) got 17.34
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.2e-3  # Muon
ADAM_LEARNING_RATE = 6.8e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.28  # Muon matrices
EMBED_WEIGHT_DECAY = 0.05  # the tied embedding
CAUTIOUS_WD = True  # decay only the entries whose update has their sign
ZERO_INIT = False
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=0.28,EMBED_WEIGHT_DECAY=0.05,CAUTIOUS_WD=1 script/train.sbatch BA

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
