lab5-yoru-CI-560492: BU with KD_ALPHA=1.0 (CE with the teacher's share 0.7 -> 1.0: the target is the teacher's top 32 plus its remaining mass on the actual token; on average 91% on the teacher); everything else as CE/BU (distillation from Llama 3.1 8B, its top-32 cached for the rows in training order; target = alpha x the teacher's top-32 probabilities, the rest (1 - alpha plus the teacher's mass beyond its top 32) on the actual next token; Muon lr 1.5e-3 wd 0.224, AdamW lr 8.5e-3 for the embedding and norms, embedding wd 0.04, every group's tau 0.39, 1% warmup, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with CE (15.549) and CD (15.846) is clean; code as CE (src/distill.py, hooked in LabTrainer._inner_training_loop after Liger; cache to 1.01B tokens and growing to 6B, job 560351, which only adds shards beyond the rows a 1B run reads); why: CE (alpha 0.7) beat CD (0.5) at every eval by ~2 points (-8.68% vs -6.94% vs BU), so the share is not yet past its best; unlike the GPT-2 lab (best 0.4-0.6, 0.8 worse), our target keeps the actual token where the teacher is unsure (its mass beyond the top 32 goes to the label, all of it at the 7% of positions where the label is outside them), so even alpha 1 is not pure distillation; expected: about CH's (15.35-15.60); better than CH means no data share is needed beyond the teacher's tail, worse means the best share is around 0.8-0.9; how it transfers to 6B: picks the 6B share; the best may fall with more tokens per parameter (the student gets nearer the teacher, and the actual token matters more), so if 0.9/1.0 beat 0.7 only slightly, 6B takes ~0.8; a 3B check of the best share against BY (13.734) can start when the cache's first 3B tokens exist (~04:10); results: eval ppl 15.53 (15.527; -0.14% vs CE's 15.549, +0.26% vs CH's 15.487, -8.81% vs BU's 17.027), 1:09:33 on 8 GPUs (25a-hgpn026), same speed as CE; ahead of CH over the first 10% (-0.56% at 5%, -0.33% at 10%), behind from 20% on and by more toward the end (+0.20% at 20%, 0.00% at 50%, +0.14% at 70%, +0.21% at 90%, +0.26% at the end): a near-pure teacher target is best early, the actual token matters more later; the best share is ~0.86 at 1B (fit over CD/CE/CH/CI) and likely lower at 6B (~0.8)
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
KD_ALPHA = 1.0  # the teacher's share of the target
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01,KD_ALPHA=1.0 script/train.sbatch CI

warmup_steps 0.01, adam beta1 0.9, KD_ALPHA 1.0, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
