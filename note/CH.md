lab5-yoru-CH-560491: BU with KD_ALPHA=0.9 (CE with the teacher's share 0.7 -> 0.9; on average 82% of the target on the teacher, CE 64%, CD 45%); everything else as CE/BU (distillation from Llama 3.1 8B, its top-32 cached for the rows in training order; target = alpha x the teacher's top-32 probabilities, the rest (1 - alpha plus the teacher's mass beyond its top 32) on the actual next token; Muon lr 1.5e-3 wd 0.224, AdamW lr 8.5e-3 for the embedding and norms, embedding wd 0.04, every group's tau 0.39, 1% warmup, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with CE (15.549) and CD (15.846) is clean; code as CE (src/distill.py, hooked in LabTrainer._inner_training_loop after Liger; cache to 1.01B tokens and growing to 6B, job 560351, which only adds shards beyond the rows a 1B run reads); why: CE (alpha 0.7) beat CD (0.5) at every eval by ~2 points (-8.68% vs -6.94% vs BU), so the share is not yet past its best; unlike the GPT-2 lab (best 0.4-0.6, 0.8 worse), our target keeps the actual token where the teacher is unsure (its mass beyond the top 32 goes to the label, all of it at the 7% of positions where the label is outside them), so even alpha 1 is not pure distillation; expected: 15.40-15.50 (-0.3 to -1% vs CE), the gain smaller than 0.5 -> 0.7's; worse than CE means the best share is between 0.7 and 0.9; how it transfers to 6B: picks the 6B share; the best may fall with more tokens per parameter (the student gets nearer the teacher, and the actual token matters more), so if 0.9/1.0 beat 0.7 only slightly, 6B takes ~0.8; a 3B check of the best share against BY (13.734) can start when the cache's first 3B tokens exist (~04:10); results: eval ppl 15.49 (15.487; -0.40% vs CE's 15.549, -9.04% vs BU's 17.027), 1:09:21 on 8 GPUs (25a-hgpn024), same speed as CE; best 1B run so far; ahead of CE at every eval, most at 20%, then by less: -1.1% at 5%, -1.6% at 20%, -1.1% at 50%, -0.8% at 70%, -0.5% at 90%, -0.4% at the end; CI (alpha 1.0, 15.527) beat CH over the first 10% (-0.56% at 5%) and fell behind from 20% on, by more toward the end (+0.26% at the end): the teacher's distribution helps most early, and the actual token matters more as the student nears the teacher, so the best share falls during training; a quadratic fit of log ppl over alpha 0.5/0.7/0.9/1.0 puts the 1B best at ~0.86 (15.48, 0.05% better than 0.9); for 6B: 6x the steps, so the best share should be lower still, take ~0.8 unless a 3B check says otherwise
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
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01,KD_ALPHA=0.9 script/train.sbatch CH

warmup_steps 0.01, adam beta1 0.9, KD_ALPHA 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
