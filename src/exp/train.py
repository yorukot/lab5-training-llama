"""150M experiment entrypoint: PYTHONPATH=src python -m exp.train --config ..."""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from exp.common import DEFAULT, atomic_json, calibrated_decay, model_config, schedule
from exp.data import PackedCollator, budget_order, load_assets
from exp.diagnose import EMA_WINDOWS, Probe
from optimizer import build_optimizer

MODEL_ID = "meta-llama/Llama-3.2-1B"
# config["method"]: "adamw" (AdamW everywhere) or a Muon variant for the block matrices (see
# src/muon.py). Other config keys for the candidates: "ema_nesterov" (its largest β), "attn_scale",
# "mlp_scale" (note/optimizer-research/03-candidates.md).
METHODS = (
    "adamw",
    "muon",
    "polar_express",
    "normuon",
    "normuon_long",
    "normuon_damped",
)


class Progress(TrainerCallback):
    def __init__(self, prefix, stop_after=None):
        self.prefix, self.stop_after = prefix, stop_after
        self.seconds = 0.0
        self.last_tokens = 0
        self.step_seconds = deque(maxlen=50)

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start
        self.step_seconds.append(self.seconds)
        self.last_tokens = int(
            self.prefix[state.global_step] - self.prefix[state.global_step - 1]
        )
        state.num_input_tokens_seen = int(self.prefix[state.global_step])
        if self.stop_after and state.global_step >= self.stop_after:
            control.should_training_stop = True
            control.should_save = True


class ExperimentTrainer(Trainer):
    def __init__(self, *args, progress, **kwargs):
        self.progress = progress
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if not isinstance(dataset, Dataset):
            raise TypeError("Experiments require a prepared Arrow Dataset")
        return SequentialSampler(dataset)

    def _clip_grad_norm(self, model):
        value = super()._clip_grad_norm(model)
        if value is not None and not torch.isfinite(torch.as_tensor(value)):
            raise FloatingPointError("Non-finite accumulated gradient norm")
        return value

    def log(self, logs, start_time=None):
        if "loss" in logs:
            logs.update(
                tokens_per_second=self.progress.last_tokens
                / max(self.progress.seconds, 1e-9),
                total_tokens_seen=self.state.num_input_tokens_seen,
            )
            if torch.cuda.is_available():
                logs["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        super().log(logs, start_time)

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Sum NLL and valid shifted-label counts, never average batch means.

        Rank-strided indices do not duplicate the last batch. Use the unwrapped
        model to avoid DDP forward collectives when ranks have unequal batches.
        """
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if not isinstance(dataset, Dataset):
            raise TypeError("Evaluation requires a prepared Arrow Dataset")
        world, rank = self.accelerator.num_processes, self.accelerator.process_index
        subset = dataset.select(list(range(rank, len(dataset), world)))
        loader = DataLoader(
            subset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=0,
        )
        model = self.accelerator.unwrap_model(self.model_wrapped)
        was_training = model.training
        model.eval()
        totals = torch.zeros(2, dtype=torch.float64, device=self.args.device)
        for batch in loader:
            batch = self._prepare_inputs(batch)
            count = (batch["labels"][:, 1:] != -100).sum()
            with torch.autocast(
                device_type=self.args.device.type,
                dtype=torch.bfloat16,
                enabled=self.args.bf16,
            ):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss")
            totals[0].add_(loss.double() * count)
            totals[1].add_(count)
        if world > 1:
            torch.distributed.all_reduce(totals)
        if totals[1] <= 0:
            raise ValueError("No evaluation target tokens")
        nll = (totals[0] / totals[1]).item()
        if not math.isfinite(nll) or nll > math.log(sys.float_info.max):
            raise FloatingPointError("Evaluation perplexity overflow")
        metrics = {
            f"{metric_key_prefix}_loss": nll,
            f"{metric_key_prefix}_perplexity": math.exp(nll),
            f"{metric_key_prefix}_target_tokens": int(totals[1].item()),
        }
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        model.train(was_training)
        return metrics


def source_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")) + sorted(root.parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(options):
    config = DEFAULT | (
        json.loads(Path(options.config).read_text()) if options.config else {}
    )
    if options.tiny_smoke:
        config.update(
            max_tokens=1536, global_batch=4, micro_batch=min(config["micro_batch"], 4)
        )
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])
    tokenizer = None
    if options.tiny_smoke:
        cfg = model_config(tiny=True)
        rows = [
            {
                "input_ids": [1]
                + [3 + (i + j) % 120 for j in range(15)]
                + [1]
                + [3 + (i + j) % 120 for j in range(7 if i % 2 else 11)],
                "doc_lens": [16, 8 if i % 2 else 12],
                "n_tokens": 24 if i % 2 else 28,
            }
            for i in range(80)
        ]
        order, prefix = budget_order(
            np.array([row["n_tokens"] for row in rows]),
            config["seed"],
            config["max_tokens"],
            config["global_batch"],
        )
        train_data = Dataset.from_list([rows[i] for i in order])
        dev_data = confirm_data = Dataset.from_list(rows[:4])
        split = {
            "fingerprint": "synthetic-smoke",
            "order_fingerprint": "synthetic-smoke",
        }
        collator = PackedCollator(0, 32, varlen=False)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "150M training requires a CUDA GPU; --tiny-smoke is CPU compatible"
            )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        cfg = model_config(AutoConfig.from_pretrained(MODEL_ID))
        cfg.pad_token_id = tokenizer.pad_token_id
        train_data, dev_data, confirm_data, prefix, split = load_assets(
            options.data_dir, options.cache_dir, config
        )
        collator = PackedCollator(tokenizer.pad_token_id)
    steps = len(prefix) - 1
    multipliers, _ = schedule(steps, config["warmup"], config["decay"])
    matrix_wd = calibrated_decay(
        config["matrix_lr"], multipliers, config["matrix_decay"]
    )
    embed_wd = calibrated_decay(config["embed_lr"], multipliers, config["embed_decay"])
    set_seed(config["seed"])
    model = AutoModelForCausalLM.from_config(
        cfg,
        dtype=torch.float32,
        attn_implementation="sdpa" if options.tiny_smoke else "flash_attention_3",
    )
    count = sum(p.numel() for p in model.parameters())
    if not options.tiny_smoke and count != 149_314_048:
        raise AssertionError(f"Incorrect model size: {count}")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise AssertionError("Embedding must be tied")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if config["global_batch"] % (config["micro_batch"] * world):
        raise ValueError(
            "Global batch must be divisible by micro batch times world size"
        )
    if config["method"] not in METHODS:
        raise ValueError(f"Unknown method: {config['method']}")
    optimizer = build_optimizer(
        model,
        "adamw" if config["method"] == "adamw" else "muon",
        config["matrix_lr"],
        matrix_wd,
        config["embed_lr"],
        embed_wd,
        config["embed_lr"],
        config["beta2"],
        config["method"],
        False,
        steps,
        config.get("ema_nesterov", 0.0),
    )
    # Per-type rotation (card S5-B): a group's lr and wd scaled together keep its weight norm
    # and scale its rotation per step; its decay timescale tau goes as 1/scale^2.
    for group in optimizer.param_groups:
        group["lr"] *= config.get(f"{group['name']}_scale", 1.0)
        group["weight_decay"] *= config.get(f"{group['name']}_scale", 1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: multipliers[i] if i < steps else 0.0
    )
    progress = Progress(prefix, options.stop_after)
    rank = int(os.environ.get("RANK", "0"))
    name = config.get("run_id", output.name)
    metadata = {
        "config": config,
        "parameters": count,
        "total_steps": steps,
        "total_tokens": int(prefix[-1]),
        "matrix_weight_decay": matrix_wd,
        "embed_weight_decay": embed_wd,
        "source_digest": source_digest(),
        "split": split,
        "gpu_count": world,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if rank == 0:
        prior = output / "metadata.json"
        if options.resume and prior.exists():
            old = json.loads(prior.read_text())
            for key in (
                "config",
                "source_digest",
                "split",
                "total_steps",
                "total_tokens",
            ):
                if old[key] != metadata[key]:
                    raise ValueError(f"Resume mismatch: {key}")
        atomic_json(prior, metadata)
    wandb_run = None
    if not options.no_wandb and rank == 0:
        import wandb

        wandb_run = wandb.init(
            project="lab-training-llama",
            entity=os.environ.get("WANDB_ENTITY", "cerulean-labs"),
            name=name,
            group=config.get("wandb_group", "screen150m"),
            job_type=config["phase"],
            config=metadata,
            id=hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16],
            resume="allow",
        )
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        run_name=name,
        report_to=[] if options.no_wandb else ["wandb"],
        max_steps=steps,
        per_device_train_batch_size=config["micro_batch"],
        per_device_eval_batch_size=config["micro_batch"],
        gradient_accumulation_steps=config["global_batch"]
        // (config["micro_batch"] * world),
        learning_rate=config["matrix_lr"],
        max_grad_norm=1.0,
        bf16=not options.tiny_smoke,
        use_cpu=options.tiny_smoke,
        use_liger_kernel=not options.tiny_smoke,
        logging_steps=1 if options.tiny_smoke else 10,
        eval_strategy="steps",
        eval_steps=max(1, steps // 10),
        save_strategy="steps",
        save_steps=max(1, steps // 10),
        save_total_limit=None if config.get("probe") else 1,
        dataloader_num_workers=0 if options.tiny_smoke else 2,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=config["seed"],
        data_seed=config["seed"],
        disable_tqdm=not options.tiny_smoke,
    )
    # A baseline run with "probe": true keeps what src/exp/diagnose.py needs afterwards.
    probe = Probe(model, steps, output / "probe.jsonl") if config.get("probe") else None
    trainer = ExperimentTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_data,
        eval_dataset=dev_data,
        processing_class=tokenizer,
        callbacks=[progress] + ([probe] if probe else []),
        optimizers=(optimizer, scheduler),
        progress=progress,
    )
    start = time.perf_counter()
    result = dict(status="running", **metadata)
    if rank == 0:
        atomic_json(output / "result.json", result)
    try:
        resume = options.resume
        if resume == "auto":
            from transformers.trainer_utils import get_last_checkpoint

            resume = get_last_checkpoint(str(output / "checkpoints"))
        trainer.train(resume_from_checkpoint=resume)
        metrics = trainer.evaluate()
        if probe and probe.emas:
            for window in EMA_WINDOWS:
                with probe.ema_weights(window):
                    key = f"ema{window * 100:g}"
                    metrics.update(trainer.evaluate(metric_key_prefix=key))
        if config["phase"] == "confirm":
            metrics.update(trainer.evaluate(confirm_data, metric_key_prefix="confirm"))
        trainer.save_model(str(output / "final"))
        status = (
            "completed"
            if trainer.state.global_step == steps or config["phase"] == "smoke"
            else "interrupted"
        )
        result.update(
            status=status,
            metrics=metrics,
            completed_steps=trainer.state.global_step,
            actual_tokens=int(prefix[trainer.state.global_step]),
            median_step_seconds=float(
                np.median(
                    list(progress.step_seconds)[10:] or list(progress.step_seconds)
                )
            ),
            peak_memory_bytes=torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
        )
    except Exception as exc:
        result.update(
            status="diverged" if isinstance(exc, FloatingPointError) else "error",
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        elapsed = time.perf_counter() - start
        result.update(
            elapsed_seconds=elapsed,
            gpu_hours=elapsed * world / 3600 if not options.tiny_smoke else 0,
            wandb_url=wandb_run.url if wandb_run else None,
        )
        if rank == 0:
            atomic_json(output / "result.json", result)
            if wandb_run:
                wandb_run.summary.update(
                    {
                        k: v
                        for k, v in result.items()
                        if k
                        in (
                            "status",
                            "metrics",
                            "gpu_hours",
                            "actual_tokens",
                            "peak_memory_bytes",
                        )
                    }
                )
                wandb_run.finish(exit_code=0 if result["status"] == "completed" else 1)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--data-dir", default=os.environ.get("DATA_DIR", "data/bestfit")
    )
    parser.add_argument("--cache-dir", default="outputs/exp150m-assets")
    parser.add_argument("--tiny-smoke", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument(
        "--stop-after", type=int, help="Smoke only; preserves the full schedule horizon"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
