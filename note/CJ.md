lab5-yoru-CJ-561615: CH with EMA-Nesterov (EMA_NESTEROV=0.6, CF's setting: the lookahead beta = 0.6 * lr_t/lr_peak over 15-74% of the steps, 0 outside it, EMA rate gamma 0.99 per step); everything else as CH (distillation from Llama 3.1 8B with the teacher's share KD_ALPHA 0.9, its top-32 cached for the rows in training order, now for all 6B tokens (job 560351); Muon lr 1.5e-3 wd 0.224, AdamW lr 8.5e-3 for the embedding and norms, embedding wd 0.04, every group's tau 0.39, 1% warmup, new data from job 552807, normuon, 131k tokens a step = 16 sequences, ~7,632 steps, MICRO_BATCH 2, AdamW beta2 0.9873, Muon momentum 0.95, linear decay over the last 80%, 1B non-padding tokens, seed 42), so the comparison with CH (15.487) is clean; code as CH and CF, unchanged (EMA-Nesterov only changes how the optimizer moves the weights, distillation only the loss, so they do not touch); why: the two best 1B results come from different places, distillation (CH, -9.04% vs BU) and EMA-Nesterov (CF, -0.62% vs BU), and both are 6B candidates; whether they add up decides the 6B recipe; they may not: the teacher's soft targets give less noisy gradients, and EMA-Nesterov's gain may come partly from averaging out the noise of hard labels; and their gains peak at different times (distillation's early and shrinking, EMA-Nesterov's late, -1.9% at 80% in CF); expected: 15.41-15.47 (-0.1 to -0.5% vs CH, less than CF's -0.62% on BU), with CF's jump when the lookahead switches on (+several % vs CH at 20%), level again by ~55%; within +-0.1% of CH or worse means they do not add up; how it transfers to 6B: at least -0.2% vs CH puts EMA-Nesterov (0.6, gamma 0.99 per step) into the 6B recipe with distillation (share ~0.8); less, and 6B takes distillation alone, the simpler recipe; results: eval ppl 15.32 (15.324; -1.05% vs CH's 15.487, -10.0% vs BU's 17.027, -9.4% vs CF's 16.921), 1:13:07 on 8 GPUs (25a-hgpn058), ~5% more wall time than CH; best 1B run so far; past the expected -0.1 to -0.5%: the two more than add up (CH's gain times CF's -0.62% would give 15.39); the curve follows CF's on BU, a little better at every eval from 15% on: the switch-on jump +5.4% at 20% (CF +6.0%), level by 40% (CF by 60%), ahead from 50% on, most at 80% (-2.21%, CF -1.90%), -1.59% at 90%, -1.27% at 95%, -1.05% at the end (CF -0.62%); so the guess that soft targets leave less noise for EMA-Nesterov to average out was wrong: it gains more with distillation, perhaps because the teacher's targets make the step direction more consistent from step to step, the motion the lookahead amplifies; for 6B: EMA-Nesterov (0.6, gamma 0.99 per step) goes into the recipe with distillation; CL (3B, both on) checks it over 3x the steps
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
EMA_NESTEROV = 0.6  # largest lookahead beta; window 15-74% of the steps
EMA_NESTEROV_GAMMA = 0.99  # default
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.5e-3,ADAM_LEARNING_RATE=8.5e-3,WEIGHT_DECAY=0.224,EMBED_WEIGHT_DECAY=0.04,DECAY_FRACTION=0.8,WARMUP_FRACTION=0.01,KD_ALPHA=0.9,EMA_NESTEROV=0.6 script/train.sbatch CJ

warmup_steps 0.01, adam beta1 0.9, KD_ALPHA 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~72 min
