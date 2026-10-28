lab5-yoru-BN-553156: BM with the new data from the user's preparation job 552807 (commit 66ea6ae, src/prepare.py; DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit): on top of the old data's exact dedup and removal of the 50k held-out docs' copies, it drops ~125k more docs (quality filters: short_text 37,767, few_words 8,219, invalid_characters 9,023, repeated_lines 4,421, character_runs 2,859, repeated_5grams 268; MinHash near-duplicates: train_near 30,615, held_out_near 1,131), only ~0.9% of the tokens (7.749B vs 7.817B accepted), then shuffles and cuts the train set at exactly 6.0B tokens (732,756 rows); the eval set is identical to the old one (same sha256 of input_ids and doc_lens), so BN vs BM is a clean comparison; everything else as BM (normuon, Muon lr 1.2e-3, AdamW lr 6.8e-3, every group's tau 0.39, Muon momentum 0.95, 5% warmup, linear decay over the last 80%, 1B non-padding tokens, seed 42); why: decides which data the 6B run uses; expected: within +-0.3% of BM's 17.143: the data differ by under 1% of the tokens, and the eval (and OJ) docs are not filtered, so dropping short or noisy docs can also cost a little; how it transfers to 6B: if BN ties or wins, the 6B run uses the new data (exactly 6B tokens, every token seen once; with the old data it would see a random 6B of 7.8B); the near-dedup matters more at 6B, where repeats would be seen several times; results: eval ppl 17.15 (17.151; +0.05% vs BM's 17.143), 1:07:32 on 8 GPUs (25a-hgpn068); a tie: -5.3% at step 382 (the first 5%, different data order), within +-0.7% after that, +0.05% at the end (BE vs BM, the init alone, was 0.02%); the filters and the near-dedup neither help nor hurt at 1B; for 6B: use the new data: it ties at 1B, is exactly 6B tokens (every token seen once), and has no near-duplicates of the held-out docs, so the in-training eval stays honest
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
DATA_DIR = "/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit"  # sbatch --export=ALL,DATA_DIR=/work/u9603854/lab5-training-llama/prepare-66ea6ae/data/bestfit,MAX_TOKENS=1000000000,MUON_VARIANT=normuon,GLOBAL_BATCH=16,MICRO_BATCH=2,ADAM_BETA2=0.9873,LEARNING_RATE=1.2e-3,ADAM_LEARNING_RATE=6.8e-3,WEIGHT_DECAY=0.28,EMBED_WEIGHT_DECAY=0.05,DECAY_FRACTION=0.8 script/train.sbatch BN

warmup_steps 0.05, adam beta1 0.9, Muon momentum 0.95, max_grad_norm 1.0, bf16, liger kernel, FA3, NCCL_SOCKET_IFNAME=lo; expected wall time ~68 min
