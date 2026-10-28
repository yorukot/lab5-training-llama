import math
import os
import time

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import distill
import sophia
from optimizer import build_optimizer

os.environ.setdefault("WANDB_ENTITY", "cerulean-labs")
os.environ.setdefault("WANDB_PROJECT", "lab5-training-llama")

MODEL_ID = "meta-llama/Llama-3.2-1B"
SEQ_LEN = 8192
DATA_DIR = os.environ.get("DATA_DIR", "data/bestfit")  # written by src/prepare_data.py
# A queued job reads this file only when it starts, so pass a run's settings at submit
# time instead of editing it: sbatch --export=ALL,LEARNING_RATE=1.2e-3 ...
# The update rules for the block matrices / the embedding and norms: "muon" (Muon / AdamW),
# "adamw", "sophia+adamw" (Sophia / AdamW) or "sophia" (see src/optimizer.py).
OPTIMIZER = os.environ.get("OPTIMIZER", "muon")
# "muon", "polar_express" or "normuon" (NorMuon plus Polar Express, see src/muon.py).
MUON_VARIANT = os.environ.get("MUON_VARIANT", "muon")
# The block matrices' lr and weight decay (Muon's update RMS is scaled to AdamW's).
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "6e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.1"))
# The embedding and norms' lr (AdamW for every OPTIMIZER but "sophia"); default LEARNING_RATE.
ADAM_LEARNING_RATE = float(os.environ.get("ADAM_LEARNING_RATE", LEARNING_RATE))
# The norms' own lr; the default is ADAM_LEARNING_RATE.
NORM_LEARNING_RATE = float(os.environ.get("NORM_LEARNING_RATE", ADAM_LEARNING_RATE))
# AdamW's beta2. For a smaller batch keep its half-life in tokens: 0.95 ** (batch / 64).
ADAM_BETA2 = float(os.environ.get("ADAM_BETA2", "0.95"))
# The (tied) embedding's weight decay; the default is WEIGHT_DECAY.
EMBED_WEIGHT_DECAY = float(os.environ.get("EMBED_WEIGHT_DECAY", WEIGHT_DECAY))
# "1": decay only the weights whose update has their sign (see src/optimizer.py).
CAUTIOUS_WD = os.environ.get("CAUTIOUS_WD", "0") == "1"
# EMA-Nesterov's largest lookahead (src/ema_nesterov.py, e.g. 0.6); 0 = off.
EMA_NESTEROV = float(os.environ.get("EMA_NESTEROV", "0"))
# "1": zero each block's output projections (o_proj, down_proj), so every block starts as the
# identity (modded-nanogpt); the rest keep the default init (std 0.02).
ZERO_INIT = os.environ.get("ZERO_INIT", "0") == "1"
# Sequences per optimizer step, and per GPU per forward pass.
GLOBAL_BATCH = int(os.environ.get("GLOBAL_BATCH", "64"))
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "4"))
# Non-padding tokens, as the lab counts them; the cap is 6B.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1000000000"))
# First 5% of the steps: warm up to the peak lr.
WARMUP_FRACTION = float(os.environ.get("WARMUP_FRACTION", "0.05"))
# Last 50% of the steps: decay to 0, "linear", "cosine" or "1-sqrt" (fast first, then slow).
DECAY_FRACTION = float(os.environ.get("DECAY_FRACTION", "0.5"))
DECAY_TYPE = os.environ.get("DECAY_TYPE", "linear")
# Distillation (src/distill.py): the teacher's share of each position's target; 0 = off.
KD_ALPHA = float(os.environ.get("KD_ALPHA", "0"))
# 1 = the label stands in for the teacher's mass beyond its top-k (src/distill.py).
KD_UNBIASED_TAIL = os.environ.get("KD_UNBIASED_TAIL", "0") == "1"

world_size = int(os.environ.get("WORLD_SIZE", "1"))
USES_SOPHIA = "sophia" in OPTIMIZER
assert OPTIMIZER in ("muon", "adamw", "sophia+adamw", "sophia"), OPTIMIZER
assert MUON_VARIANT in ("muon", "polar_express", "normuon"), MUON_VARIANT
assert DECAY_TYPE in ("linear", "cosine", "1-sqrt"), DECAY_TYPE
assert GLOBAL_BATCH % (MICRO_BATCH * world_size) == 0, "GLOBAL_BATCH vs MICRO_BATCH"
# Sophia's curvature pass runs once per training_step: once per step needs no grad accumulation.
assert not USES_SOPHIA or GLOBAL_BATCH == MICRO_BATCH * world_size
run_name = os.environ.get("RUN_NAME", "lab5-local")  # Set by train.sbatch.

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)
# Rows are padded with a reserved token, so the Trainer can count the non-padding tokens.
tokenizer.pad_token = "<|finetune_right_pad_id|>"
PAD = tokenizer.pad_token_id
config = AutoConfig.from_pretrained(MODEL_ID)
# The Trainer seeds only later (the data order), so seed the init here too.
set_seed(42)
model = AutoModelForCausalLM.from_config(
    config, dtype=torch.float32, attn_implementation="flash_attention_3"
)
if ZERO_INIT:
    for layer in model.model.layers:
        torch.nn.init.zeros_(layer.self_attn.o_proj.weight)
        torch.nn.init.zeros_(layer.mlp.down_proj.weight)

train_data = Dataset.load_from_disk(f"{DATA_DIR}/train")
teacher = None
if KD_ALPHA:
    # Each row's index, to look up its teacher targets; eval rows have none.
    train_data = train_data.add_column("row", list(range(len(train_data))))
    teacher = distill.Teacher()
train_data = train_data.with_format("torch")
eval_data = Dataset.load_from_disk(f"{DATA_DIR}/eval").with_format("torch")
# Rows are not full, so the steps that fit MAX_TOKENS follow the rows' average fill.
fill = sum(train_data.with_format(None)["n_tokens"]) / (len(train_data) * SEQ_LEN)
max_steps = int(MAX_TOKENS / (GLOBAL_BATCH * SEQ_LEN * fill))


def collate(examples):
    """Attention stays inside each document, like the OJ scoring every document alone.

    Rows hold whole documents (best-fit by prepare_data.py) and are padded here to SEQ_LEN;
    the padding is a segment of its own. Each document's first token (the BOS, which the OJ
    never scores) and the padding get label -100.
    """
    input_ids = torch.full((len(examples), SEQ_LEN), PAD)
    position_ids = torch.zeros_like(input_ids)
    real = torch.zeros_like(input_ids, dtype=torch.bool)
    seg_lens = []
    for row, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[row, :n] = e["input_ids"]
        real[row, :n] = True
        lens = e["doc_lens"].tolist() + ([SEQ_LEN - n] if n < SEQ_LEN else [])
        position_ids[row] = torch.cat([torch.arange(length) for length in lens])
        seg_lens += lens
    cu_seqlens = torch.tensor([0] + seg_lens).cumsum(0).int()
    max_len = max(seg_lens)
    batch = {
        "input_ids": input_ids,
        "labels": input_ids.masked_fill((position_ids == 0) | ~real, -100),
        "position_ids": position_ids,
        # FA3's varlen kernel splits the flattened batch at these offsets.
        "cu_seq_lens_q": cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": max_len,
        "max_length_k": max_len,
    }
    if teacher is not None and "row" in examples[0]:
        batch |= teacher.targets([int(e["row"]) for e in examples], SEQ_LEN)
    return batch


class StepTimer(TrainerCallback):
    """The last step's time and non-padding tokens (all GPUs)."""

    seconds, tokens = float("nan"), 0

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.start, self.start_tokens = time.perf_counter(), state.num_input_tokens_seen

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start
        self.tokens = state.num_input_tokens_seen - self.start_tokens


class TokenBudget(TrainerCallback):
    """Stop before a step could take the non-padding tokens past MAX_TOKENS."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.num_input_tokens_seen + GLOBAL_BATCH * SEQ_LEN > MAX_TOKENS:
            control.should_training_stop = True


timer = StepTimer()


class LabTrainer(Trainer):
    def _inner_training_loop(self, *args, **kwargs):
        # train() has just let Liger replace model.forward; accelerate wraps it next, in here.
        if KD_ALPHA:
            distill.use_mixed_loss(self.model, KD_ALPHA, KD_UNBIASED_TAIL)
        return super()._inner_training_loop(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if USES_SOPHIA and self.state.global_step % sophia.HESSIAN_INTERVAL == 0:
            self.update_hessian(model, self._prepare_inputs(inputs))
        return super().training_step(model, inputs, num_items_in_batch)

    def update_hessian(self, model, inputs):
        """Sophia's curvature estimate, as in the official train_sophiag.py but without clipping
        (see src/sophia.py): backpropagate the loss on labels sampled from the model itself."""
        model.train()
        unwrapped = self.accelerator.unwrap_model(model)
        labels = inputs["labels"]
        inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            hidden = unwrapped.model(**inputs).last_hidden_state
        sampled = sophia.sample_labels(hidden, unwrapped.lm_head.weight, labels)
        del hidden
        self.accelerator.backward(model(**inputs, labels=sampled).loss)
        optimizer.update_hessian(GLOBAL_BATCH * SEQ_LEN)
        model.zero_grad(set_to_none=True)

    def log(self, logs, start_time=None):
        if "loss" in logs:
            logs["tokens_per_second"] = timer.tokens / timer.seconds
            logs["total_tokens_seen"] = self.state.num_input_tokens_seen
            if USES_SOPHIA:
                logs["sophia_win_rate"] = float(optimizer.win_rate)
        if "eval_loss" in logs:
            logs["eval_perplexity"] = math.exp(logs["eval_loss"])
        super().log(logs, start_time)


args = TrainingArguments(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    report_to="wandb",
    max_steps=max_steps,
    per_device_train_batch_size=MICRO_BATCH,
    per_device_eval_batch_size=MICRO_BATCH,
    gradient_accumulation_steps=GLOBAL_BATCH // (MICRO_BATCH * world_size),
    optim="adamw_torch_fused",
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    adam_beta1=0.9,
    adam_beta2=ADAM_BETA2,
    max_grad_norm=1.0,
    lr_scheduler_type="warmup_stable_decay",
    lr_scheduler_kwargs={
        "num_decay_steps": int(DECAY_FRACTION * max_steps),
        "decay_type": DECAY_TYPE,
        "min_lr_ratio": 0.0,
    },
    warmup_steps=WARMUP_FRACTION,
    bf16=True,
    use_liger_kernel=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=0.05,
    prediction_loss_only=True,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=1,
    dataloader_num_workers=4,
    dataloader_drop_last=True,
    remove_unused_columns=False,  # collate needs doc_lens, which forward() does not take
    ddp_find_unused_parameters=False,
    include_num_input_tokens_seen="non_padding",  # input_ids != PAD, summed over GPUs
)

optimizer = build_optimizer(
    model,
    OPTIMIZER,
    LEARNING_RATE,
    WEIGHT_DECAY,
    ADAM_LEARNING_RATE,
    EMBED_WEIGHT_DECAY,
    NORM_LEARNING_RATE,
    ADAM_BETA2,
    MUON_VARIANT,
    CAUTIOUS_WD,
    max_steps,
    EMA_NESTEROV,
)

trainer = LabTrainer(
    model=model,
    args=args,
    data_collator=collate,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer,
    callbacks=[timer, TokenBudget()],
    optimizers=(optimizer, None),
)
trainer.train()
trainer.evaluate()
trainer.save_model(f"outputs/{run_name}/final")
