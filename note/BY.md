lab5-yoru-BY-557059: BP (3B tokens) with BU's two improvements: 1% warmup (~229 steps; WARMUP_FRACTION=0.01) and BU's lrs before the same x0.87 scaling: Muon 1.5e-3 x 0.87 = 1.3e-3, AdamW (embedding, norms) 8.5e-3 x 0.87 = 7.4e-3; wd so every group's tau = 1/(lr * wd * steps) stays 0.39 as in BP: Muon 0.086, embedding 0.0151; everything else as BP (new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~22,898 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 3B non-padding tokens, seed 42), so the comparison with BP (13.771) is clean; why: at 1B the short warmup (BR, -0.52%) and the higher lrs (BQ, -0.31%) gave BU -0.72% vs BN together; whether they hold at 3x the steps decides the 6B recipe: the warmup's share of the run shrinks with more steps, and the best lr may move with the token count (BP's x0.87 rule is untested); expected: -0.3 to -0.7% vs BP (13.68-13.73); about 0 means the gains are 1B-only and the 6B run should keep BP's settings; worse means the x0.87 rule is too weak and the lr is now too high at 3B; how it transfers to 6B: a win means the 6B run takes BU's recipe with the lrs x0.8 (1.2e-3 / 6.8e-3); the size of the gain vs 1B (-0.72%) tells how much is left of it at 6B; results: eval ppl 13.73 (13.734; -0.27% vs BP's 13.771), 3:10:04 on 8 GPUs (25a-hgpn042), ~25 H200-h, W&B run dyk87mek; best 3B run; ahead all the way, by less and less: -20.5% at 5%, -2.7% at 20%, -1.3% at 30%, -0.56% at 50%, -0.34% at 75%, -0.27% at the end; the gain is about a third of BU's at 1B (-0.72% vs BN): the early lead from the short warmup fades with 3x the steps, and the higher lr helps less (its best may fall with more tokens, faster than the x0.87 rule); no spikes (grad norm at most 6.2 in the warmup); for 6B: BU's recipe still wins, by maybe -0.1 to -0.2% at 6B; Muon 1.2e-3 / AdamW 6.8e-3 (BU's x0.8) would follow the same rule
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.3e-3  # Muon
ADAM_LEARNING_RATE = 7.4e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.086  # Muon matrices
EMBED_WEIGHT_DECAY = 0.0151  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 3_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.8  # last 80% of the steps: decay to 0
DECAY_TYPE = "linear"
WARMUP_FRACTION = 0.01
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=3000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.3e-3,ADAM_LEARNING_RATE=7.4e-3,WEIGHT_DECAY=0.086,EMBED_WEIGHT_DECAY=0.0151,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01 script/train.sbatch BY

warmup_steps 0.01, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~3.2 h
