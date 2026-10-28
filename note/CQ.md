lab5-yoru-CQ-565042: CN with the teacher's top-128 instead of its top-64 (a new cache from job 564939: the same teacher, rows, order and format, only TOP_K = 128 in src/distillation/generate_logits.py and src/distill.py; KD_UNBIASED_TAIL off); everything else as CN (KD_ALPHA 0.9, EMA-Nesterov 0.6 with gamma 0.99 over 15-74% of the steps, Muon lr 1.5e-3 wd 0.224, AdamW lr 8.5e-3 for the embedding and norms, embedding wd 0.04, every group's tau 0.39, 1% warmup, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with CN (15.204) is clean; it starts once the cache holds the first 1,000 shards while job 564939 keeps writing the rest (the full cache is due ~17:00); why: the user approved a top-128 cache after CN (top-64) beat CJ (top-32) by 0.78%; measured on the first 8 shards (8.4M positions): the top 128 hold 96.5% of the teacher's mass (top-64 94.7%, top-32 92.2%), so another 1.75% of the mass moves from the label to the teacher's ranks 65-128, and the label is outside the top 128 3.65% of the time (top-64 5.5%); cost: ~7.2 h of 8 GPUs (not in the budget) and ~4.65 TB (the top-32 cache was deleted at the user's request to make room); expected: 15.10-15.21 (0 to -0.7% vs CN), most likely ~15.16 (-0.3%): CN gave -0.31% per 1% of the mass moved, which would give -0.55% here, but ranks 65-128 are rarer tokens whose probabilities matter at fewer positions, so likely less; a lead that holds through the decay like CN's; ~75 min; how it transfers to 6B: if CQ beats CN by >= 0.15%, the 6B run uses top-128 (it needs the full cache, ~17:00, or a Teacher that re-reads the cache directory as it grows); otherwise top-64; results: eval ppl 15.12 (15.122; -0.54% vs CN's 15.204), 1:12:26 on 8 GPUs (25a-hgpn175), the same wall time as CN (1:12:40) with a cache twice the size; best 1B run; inside the expected 0 to -0.7%, past the most likely -0.3%; ahead at every eval: -1.3% at 5%, -0.4% at 10-15%, -1.5% at 20% (EMA-Nesterov's switch-on), then steady at -0.57 to -0.66% from 25% to 85% and -0.47 to -0.54% at the end; like CN's, the lead barely shrinks in the decay; per 1% of the teacher's mass moved from the label: -0.31% for top-32 -> 64 (2.5% of the mass), -0.31% here (1.75%), so ranks 65-128 are worth as much per unit of mass as ranks 33-64; the cache's tail is now 3.5% of the mass, so top-256 would move about another 1.2% (~-0.35% at 1B) for another ~9 TB, more than the disk holds; for 6B: by the rule set beforehand (>= 0.15%), the 6B run uses top-128; the full cache is due ~16:55 (job 564939 at round 247 of 716, 36 s a round at 12:11); if the gain shrinks with tokens like CN's assumed x0.8 per 3x tokens, ~-0.4% at 6B: 6B ~11.45 in-training (11.49 x 0.996), OJ ~11.58
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.5e-3  # Muon
ADAM_LEARNING_RATE = 8.5e-3  # the AdamW params (embedding, norms)
ADAM_BETA2 = 0.9873
WEIGHT_DECAY = 0.224  # Muon matrices
EMBED_WEIGHT_DECAY = 0.04  # the tied embedding
SEQ_LEN = 8192
GLOBAL_BATCH = 16  # sequences per optimizer step
MICRO_BATCH = 2  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.8  # last 80% of the steps: decay to 0
DECAY_TYPE = "linear"
WARMUP_FRACTION = 0.01
KD_ALPHA = 0.9  # the teacher's share of the target
TOP_K = 128  # src/distill.py; CACHE_DIR .../distillation/llama31-8b-top128 (job 564939)
EMA_NESTEROV = 0.6  # largest lookahead beta; window 15-74% of the steps
EMA_NESTEROV_GAMMA = 0.99  # default
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01,KD_ALPHA=0.9,EMA_NESTEROV=0.6 script/train.sbatch CQ

warmup_steps 0.01, adam beta1 0.9, KD_ALPHA 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~75 min
