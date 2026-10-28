lab5-yoru-AR-540617: AM with the tied embedding's weight decay 0.8 -> 0 (EMBED_WEIGHT_DECAY=0; Muon matrices keep 0.8, norms stay at 0), everything else as AM (normuon, lr 1.2e-3, wd 0.8, linear decay, best-fit data from job 538113, 1B non-padding tokens); why: the embedding, tied with the output layer, is 128256 x 2048 = 263M parameters, 21% of the model, and its wd was never tuned on its own, it just followed the Muon matrices; over AM's schedule wd 0.8 shrinks a row that gets no gradient to ~27%; OLMo 2 removed weight decay from embeddings; expected: uncertain, within about +-1% of AM's 18.37: the tied output layer gets a dense gradient (every vocab row, through the softmax), so few rows are left to wd alone, and AN (wd 1.6 everywhere, embedding included) was not worse than AL; how it transfers to 6B: if wd 0 wins, the embedding drops out of the tau-based wd transfer and stays at 0 at 6B; results: eval ppl 18.39 (+0.11% vs AM's 18.37), 0:47:58 on 8 GPUs; no measurable effect (within seed noise): ~0.7% behind AM early, ~0.5% ahead mid-decay, even at the end; keep the embedding's wd equal to WEIGHT_DECAY (one setting fewer), so at 6B it follows the same tau transfer as the matrices
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "normuon"
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 0.8
EMBED_WEIGHT_DECAY = 0  # the tied embedding; the other matrices use WEIGHT_DECAY
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: decay to 0
DECAY_TYPE = "linear"
DATA_DIR = "data/bestfit"  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.2e-3,WEIGHT_DECAY=0.8,MUON_VARIANT=normuon,EMBED_WEIGHT_DECAY=0 script/train.sbatch AR

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
