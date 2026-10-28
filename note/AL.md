lab5-yoru-AL-539949 (first try 539939 cancelled: the Trainer dropped the doc_lens column, fixed with remove_unused_columns=False): AK's recipe (Muon, lr 1.2e-3, wd 0.8) on the new best-fit data, at 1B non-padding tokens (~1,909 steps); data: data/bestfit (job 538113): the first copy of each text kept (case and whitespace normalized hash), any copy of the OJ's 50k held-out docs dropped, tokenized exactly like the OJ (study-group-online-judge PerplexityEvaluator: BOS + text, truncated to 8192, no EOS), best-fit decreasing packing so no document is ever split across rows; rows padded with <|finetune_right_pad_id|>, the padding is its own attention segment with label -100; the token budget counts non-padding tokens only (Trainer include_num_input_tokens_seen="non_padding", max_steps from the rows' average fill, a TokenBudget callback stops before MAX_TOKENS); eval: the same 10k held-out docs, each now whole and truncated like the OJ, so the in-training eval is the OJ's metric on 20% of its docs and is not comparable with runs before AL; why: the baseline for the best-fit data (AM and AN compare against it); expected: lower eval ppl than AK's 19.61, mostly because the eval no longer scores the parts of long documents past 8192 or document tails cut from their beginning (AB's OJ 16.48 was ~2% below its old in-training eval of 16.79); whole-document training and dedup should help a little on top (unverified); how it transfers to 6B: the final 6B run will use this data and this eval; results: eval ppl 18.51, 0:47:26 on 8 GPUs, 999,940,886 non-padding tokens in 1,908 steps (padding 0.01%), ~400k tokens/s; the new baseline: 5.6% below AK's 19.61 on the old data and eval, as expected mostly from the eval no longer scoring cut document tails and text past 8192, so later runs compare against 18.51, not 19.61
---
MODEL_ID = "meta-llama/Llama-3.2-1B"
OPTIMIZER = "muon"  # "adamw" or "muon"
MUON_VARIANT = "muon"
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 0.8  # sbatch --export=ALL,MAX_TOKENS=1000000000,LEARNING_RATE=1.2e-3,WEIGHT_DECAY=0.8 script/train.sbatch AL
SEQ_LEN = 8192
GLOBAL_BATCH = 64  # sequences per optimizer step
MICRO_BATCH = 4  # sequences per GPU per forward pass
MAX_TOKENS = 1_000_000_000  # non-padding tokens
DECAY_FRACTION = 0.5  # last 50% of the steps: linear decay to 0
DATA_DIR = "data/bestfit"

warmup_steps 0.05, adam betas (0.9, 0.95), max_grad_norm 1.0, bf16, liger kernel, FA3
