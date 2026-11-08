#!/usr/bin/env python3
"""用 SSQR 的 CQC 目标继续预训练 T5，并提供可复现验证和完整断点恢复。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_constant_schedule

from training_checkpoint import (
    capture_rng_state, load_trainer_state, move_optimizer_to_device,
    resolve_resume_checkpoint, restore_rng_state, save_checkpoint,
)


class CQCDataset(Dataset):
    def __init__(self, path: Path, max_samples: int = 0):
        self.queries = [line.strip() for line in path.open(encoding="utf-8") if line.strip()]
        if max_samples:
            self.queries = self.queries[:max_samples]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return self.queries[index]


class FixedCQCDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if any(not row.get("source") or not row.get("target") for row in self.rows):
            raise ValueError("fixed validation rows require source and target")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return row["source"], row["target"]


def corrupt(query: str, rng: random.Random) -> tuple[str, str]:
    words = query.split()
    if not words:
        raise ValueError("CQC query cannot be empty")
    span_length = max(1, round(0.15 * len(words)))
    start = rng.randrange(len(words) - span_length + 1)
    span = words[start:start + span_length]
    source = " ".join(words[:start] + ["<extra_id_0>"] + words[start + span_length:])
    target = "<extra_id_0> " + " ".join(span) + " <extra_id_1>"
    return source, target


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_batch(tokenizer, sources, targets, max_length, device):
    inputs = tokenizer(list(sources), padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    labels = tokenizer(list(targets), padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt").input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    return ({key: value.to(device) for key, value in inputs.items()}, labels.to(device))


def validation_loss(model, tokenizer, dataset, device, batch_size, max_length, use_bf16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device == "cuda")
    model.eval(); weighted_loss = 0.0; tokens = 0
    with torch.inference_mode():
        for sources, targets in loader:
            inputs, labels = encode_batch(tokenizer, sources, targets, max_length, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                enabled=device == "cuda"):
                loss = model(**inputs, labels=labels).loss
            count = int((labels != -100).sum().item())
            weighted_loss += float(loss.item()) * count; tokens += count
    model.train()
    return weighted_loss / max(tokens, 1)


def save_best(output, model, tokenizer, step, value, configuration):
    best = output / "best"
    temporary = output / "best.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary); tokenizer.save_pretrained(temporary)
    (temporary / "best_checkpoint.json").write_text(json.dumps({
        "global_step": step, "validation_loss": value,
        "configuration": configuration,
    }, indent=2), encoding="utf-8")
    if best.exists():
        shutil.rmtree(best)
    temporary.replace(best)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-max-samples", type=int, default=0,
                        help="Only for smoke tests; formal training must use 0 (the full file).")
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-start-step", type=int, default=3000)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--reset-early-stopping-counter", action="store_true",
                        help="Resume after downstream MRR review found improvement.")
    parser.add_argument("--complete-all-epochs", action="store_true",
                        help="Keep validation/checkpoints but do not pause before all epochs finish.")
    parser.add_argument("--resume", default="auto", help="auto, none, or checkpoint directory")
    args = parser.parse_args()
    if min(args.batch_size, args.gradient_accumulation, args.save_steps) < 1:
        parser.error("batch, accumulation, and save steps must be positive")
    if args.batch_size * args.gradient_accumulation != 32:
        parser.error("effective batch size must be exactly 32")
    if not args.data_manifest.is_file():
        parser.error("data manifest does not exist")

    complete_marker = args.output / "training_complete.json"
    if args.resume.lower() == "auto" and complete_marker.is_file():
        print(f"Training already complete: {args.output}", flush=True); return
    if args.resume.lower() == "none" and args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Fresh output directory is not empty: {args.output}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = resolve_resume_checkpoint(args.output, args.resume)
    model_source = checkpoint or args.model
    tokenizer = AutoTokenizer.from_pretrained(str(model_source))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_source)).to(device)
    dataset = CQCDataset(args.corpus, args.max_samples)
    validation = FixedCQCDataset(args.validation)
    if args.validation_max_samples:
        validation.rows = validation.rows[:args.validation_max_samples]
    batches_per_epoch = math.ceil(len(dataset) / args.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    total_steps = args.max_steps or updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_constant_schedule(optimizer)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and not use_bf16)
    signature = {
        "model": str(args.model.resolve()), "corpus": str(args.corpus.resolve()),
        "validation": str(args.validation.resolve()), "data_manifest": str(args.data_manifest.resolve()),
        "validation_max_samples": args.validation_max_samples,
        "data_manifest_sha256": file_sha256(args.data_manifest), "epochs": args.epochs,
        "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "max_length": args.max_length, "learning_rate": args.learning_rate,
        "max_samples": args.max_samples, "max_steps": args.max_steps, "seed": args.seed,
        "save_steps": args.save_steps, "total_steps": total_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_start_step": args.early_stopping_start_step,
        "min_delta": args.min_delta, "scheduler": "constant",
        "precision": "bf16" if use_bf16 else "fp16" if device == "cuda" else "fp32",
        "masking": "dynamic per epoch; one contiguous span of round(15% words), minimum 1",
    }

    step = start_epoch = start_batch = 0
    best_value = float("inf"); best_step = 0
    patience_reference = float("inf"); bad_count = 0
    corruption_rng = random.Random(args.seed)
    recent_losses = deque(maxlen=100)
    interval_loss_sum = 0.0; interval_loss_count = 0
    if checkpoint:
        state = load_trainer_state(checkpoint)
        stored_signature = state.get("signature")
        if stored_signature != signature:
            raise ValueError("Checkpoint configuration does not match this training run")
        optimizer.load_state_dict(state["optimizer"]); move_optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        step, start_epoch = state["global_step"], state["epoch"]
        start_batch = state["batch_index"] + 1
        restore_rng_state(state["rng_state"]); corruption_rng.setstate(state["corruption_rng_state"])
        best_value = state.get("best_validation_loss", best_value)
        best_step = state.get("best_step", 0)
        patience_reference = state.get("patience_reference", best_value)
        bad_count = state.get("bad_count", 0)
        if args.reset_early_stopping_counter:
            bad_count = 0
        recent_losses.extend(state.get("recent_losses", []))
        interval_loss_sum = state.get("interval_loss_sum", 0.0)
        interval_loss_count = state.get("interval_loss_count", 0)
        if start_batch % args.gradient_accumulation and start_batch != batches_per_epoch:
            raise RuntimeError("Checkpoint is inside a gradient-accumulation group")
        print(f"Resumed from {checkpoint} epoch={start_epoch + 1} batch={start_batch} step={step}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_config.json").write_text(json.dumps(signature, indent=2), encoding="utf-8")
    model.train(); optimizer.zero_grad(set_to_none=True)
    if checkpoint is None:
        save_checkpoint(
            output=args.output, global_step=0, model=model, tokenizer=tokenizer,
            save_total_limit=args.save_total_limit,
            trainer_state={
                "global_step": 0, "epoch": 0, "batch_index": -1,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "rng_state": capture_rng_state(),
                "corruption_rng_state": corruption_rng.getstate(), "signature": signature,
                "validation_loss": None, "best_validation_loss": best_value,
                "best_step": best_step, "patience_reference": patience_reference,
                "bad_count": bad_count, "recent_losses": [],
                "interval_loss_sum": 0.0, "interval_loss_count": 0,
            })
        print(f"Saved initial healthy checkpoint: {args.output / 'checkpoint-0'}", flush=True)
    stop_reason = "completed"
    for epoch in range(start_epoch, args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            pin_memory=device == "cuda", generator=generator)
        for batch_index, queries in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            sources, targets = zip(*(corrupt(query, corruption_rng) for query in queries))
            inputs, labels = encode_batch(tokenizer, sources, targets, args.max_length, device)
            group_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            accumulation_size = min(args.gradient_accumulation, len(loader) - group_start)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                enabled=device == "cuda"):
                raw_loss = model(**inputs, labels=labels).loss
                loss = raw_loss / accumulation_size
            if not torch.isfinite(raw_loss):
                stop_reason = f"non_finite_loss_at_batch_{batch_index}"
                raise FloatingPointError(stop_reason)
            if use_bf16:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            should_update = ((batch_index + 1) % args.gradient_accumulation == 0
                             or batch_index + 1 == len(loader))
            if not should_update:
                continue
            if not use_bf16:
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                stop_reason = f"non_finite_gradient_at_batch_{batch_index}"
                raise FloatingPointError(stop_reason)
            if use_bf16:
                optimizer.step()
            else:
                scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True); scheduler.step(); step += 1
            current_loss = float(raw_loss.item())
            recent_losses.append(current_loss); interval_loss_sum += current_loss; interval_loss_count += 1
            learning_rate = float(scheduler.get_last_lr()[0])
            if step % 100 == 0:
                print(f"epoch={epoch+1} step={step} batch_loss={current_loss:.6f} "
                      f"train_loss_100={np.mean(recent_losses):.6f} lr={learning_rate:.8g} "
                      f"grad_norm={float(gradient_norm):.6f}", flush=True)
            if step % args.save_steps == 0 or step >= total_steps:
                value = validation_loss(model, tokenizer, validation, device, args.batch_size,
                                        args.max_length, use_bf16)
                interval_average = interval_loss_sum / max(interval_loss_count, 1)
                is_global_best = value < best_value
                significant_improvement = value < patience_reference - args.min_delta
                if is_global_best:
                    best_value, best_step = value, step
                    save_best(args.output, model, tokenizer, step, value, signature)
                if significant_improvement:
                    patience_reference, bad_count = value, 0
                elif step >= args.early_stopping_start_step:
                    bad_count += 1
                print(f"checkpoint step={step} train_loss_interval={interval_average:.6f} "
                      f"validation_loss={value:.6f} best={best_value:.6f}@{best_step} "
                      f"global_best={is_global_best} significant_improvement={significant_improvement} "
                      f"bad_count={bad_count}", flush=True)
                state = {
                    "global_step": step, "epoch": epoch, "batch_index": batch_index,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "rng_state": capture_rng_state(), "corruption_rng_state": corruption_rng.getstate(),
                    "signature": signature, "validation_loss": value,
                    "best_validation_loss": best_value, "best_step": best_step,
                    "patience_reference": patience_reference,
                    "bad_count": bad_count, "recent_losses": list(recent_losses),
                    "interval_loss_sum": 0.0, "interval_loss_count": 0,
                }
                saved = save_checkpoint(output=args.output, global_step=step, model=model,
                                        tokenizer=tokenizer, trainer_state=state,
                                        save_total_limit=args.save_total_limit)
                print(f"Saved checkpoint: {saved}", flush=True)
                interval_loss_sum = 0.0; interval_loss_count = 0
                if (not args.complete_all_epochs
                        and step >= args.early_stopping_start_step
                        and bad_count >= args.early_stopping_patience):
                    stop_reason = "paused_for_downstream_mrr_review"
                    break
            if step >= total_steps:
                break
        start_batch = 0
        if step >= total_steps or stop_reason != "completed":
            break

    result = {"steps": step, "epochs_requested": args.epochs, "stop_reason": stop_reason,
              "best_step": best_step, "best_validation_loss": best_value,
              "best_model": str((args.output / "best").resolve())}
    marker = (args.output / "training_complete.json" if stop_reason == "completed"
              else args.output / "training_paused.json")
    marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    stale_marker = (args.output / "training_paused.json" if stop_reason == "completed"
                    else args.output / "training_complete.json")
    if stale_marker.exists():
        stale_marker.unlink()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
