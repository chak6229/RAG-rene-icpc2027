#!/usr/bin/env python3
"""Leakage-safe CodeBERT training for CodeXGLUE AdvTest with resumable validation."""
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from training_checkpoint import (
    capture_rng_state, load_trainer_state, move_optimizer_to_device,
    resolve_resume_checkpoint, restore_rng_state, save_checkpoint,
)


class PairDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        code_tokens = row.get("code_tokens") or row.get("function_tokens")
        query_tokens = row.get("docstring_tokens")
        if not code_tokens or not query_tokens:
            raise ValueError(f"row {index} lacks tokenized code/query fields")
        return " ".join(map(str, code_tokens)), " ".join(map(str, query_tokens))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def encode(tokenizer, texts, max_length, device):
    values = tokenizer(list(texts), padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    return {name: tensor.to(device) for name, tensor in values.items()}


def vectors(model, tokenizer, texts, max_length, batch_size, device, use_bf16):
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = encode(tokenizer, texts[start:start + batch_size], max_length, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda" and use_bf16):
                hidden = model(**encoded).last_hidden_state[:, 0, :]
                hidden = torch.nn.functional.normalize(hidden.float(), dim=1)
            output.append(hidden.cpu().numpy().astype(np.float32))
    return np.vstack(output)


def validate(model, tokenizer, dataset, max_length, batch_size, device, use_bf16, k):
    model.eval()
    codes = [dataset[index][0] for index in range(len(dataset))]
    queries = [dataset[index][1] for index in range(len(dataset))]
    code_vectors = vectors(model, tokenizer, codes, max_length, batch_size, device, use_bf16)
    query_vectors = vectors(model, tokenizer, queries, max_length, batch_size, device, use_bf16)
    reciprocal_ranks = []
    for start in range(0, len(dataset), 256):
        scores = query_vectors[start:start + 256] @ code_vectors.T
        relevant = scores[np.arange(len(scores)), np.arange(start, start + len(scores))]
        ranks = 1 + np.count_nonzero(scores > relevant[:, None], axis=1)
        reciprocal_ranks.extend(np.where(ranks <= k, 1.0 / ranks, 0.0).tolist())
    model.train()
    return float(np.mean(reciprocal_ranks))


def save_best(output, model, tokenizer, step, mrr, signature):
    temporary, best = output / "best.tmp", output / "best"
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary); tokenizer.save_pretrained(temporary)
    (temporary / "best_checkpoint.json").write_text(json.dumps({
        "global_step": step, "validation_mrr@100": mrr, "configuration": signature,
    }, indent=2), encoding="utf-8")
    if best.exists(): shutil.rmtree(best)
    temporary.replace(best)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--validation-steps", type=int, default=500)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", default="auto")
    args = parser.parse_args()
    if args.batch_size != 32: parser.error("official train batch size must be 32")
    if not args.data_manifest.is_file(): parser.error("missing leakage audit manifest")
    if args.resume.lower() == "none" and args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"fresh output is not empty: {args.output}")
    complete = args.output / "training_complete.json"
    if args.resume.lower() == "auto" and complete.is_file():
        print(f"Training already complete: {args.output}", flush=True); return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = resolve_resume_checkpoint(args.output, args.resume)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint or args.model))
    model = AutoModel.from_pretrained(str(checkpoint or args.model)).to(device)
    train_data, valid_data = PairDataset(args.train), PairDataset(args.validation)
    batches_per_epoch = math.ceil(len(train_data) / args.batch_size)
    total_steps = args.max_steps or batches_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * .1), num_training_steps=total_steps)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and not use_bf16)
    signature = {
        "model": str(args.model.resolve()), "train": str(args.train.resolve()),
        "validation": str(args.validation.resolve()), "train_sha256": sha256(args.train),
        "validation_sha256": sha256(args.validation),
        "data_manifest_sha256": sha256(args.data_manifest), "epochs": args.epochs,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "save_steps": args.save_steps, "validation_steps": args.validation_steps,
        "total_steps": total_steps, "precision": "bf16" if use_bf16 else "fp16",
        "representation": "code_tokens/function_tokens only; query=docstring_tokens",
    }
    step = start_epoch = start_batch = 0
    best_mrr, best_step = -1.0, 0
    recent_losses = deque(maxlen=100)
    if checkpoint:
        state = load_trainer_state(checkpoint)
        if state["signature"] != signature: raise ValueError("checkpoint configuration mismatch")
        optimizer.load_state_dict(state["optimizer"]); move_optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None: scaler.load_state_dict(state["scaler"])
        restore_rng_state(state["rng_state"])
        step, start_epoch = state["global_step"], state["epoch"]
        start_batch = state["batch_index"] + 1
        best_mrr, best_step = state.get("best_mrr", best_mrr), state.get("best_step", 0)
        recent_losses.extend(state.get("recent_losses", []))
        print(f"Resumed from {checkpoint} epoch={start_epoch + 1} batch={start_batch} step={step}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_config.json").write_text(json.dumps(signature, indent=2), encoding="utf-8")
    model.train(); optimizer.zero_grad(set_to_none=True)
    stop = step >= total_steps
    for epoch in range(start_epoch, args.epochs):
        if stop: break
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            pin_memory=device == "cuda",
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        for batch_index, (codes, queries) in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch: continue
            encoded_code = encode(tokenizer, codes, args.max_length, device)
            encoded_query = encode(tokenizer, queries, args.max_length, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                enabled=device == "cuda"):
                code_vec = torch.nn.functional.normalize(model(**encoded_code).last_hidden_state[:, 0, :], dim=1)
                query_vec = torch.nn.functional.normalize(model(**encoded_query).last_hidden_state[:, 0, :], dim=1)
                raw_loss = torch.nn.functional.cross_entropy(
                    query_vec @ code_vec.T, torch.arange(len(codes), device=device))
            if not torch.isfinite(raw_loss): raise FloatingPointError(f"non-finite loss at {step}")
            if use_bf16: raw_loss.backward()
            else: scaler.scale(raw_loss).backward(); scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm): raise FloatingPointError(f"non-finite gradient at {step}")
            if use_bf16: optimizer.step()
            else: scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True); scheduler.step(); step += 1
            recent_losses.append(float(raw_loss.item()))
            if step % 100 == 0:
                print(f"epoch={epoch + 1} step={step}/{total_steps} batch_loss={raw_loss.item():.6f} "
                      f"train_loss_100={np.mean(recent_losses):.6f} lr={scheduler.get_last_lr()[0]:.8g} "
                      f"grad_norm={float(grad_norm):.6f}", flush=True)
            current_mrr = None
            if step % args.validation_steps == 0 or step >= total_steps:
                current_mrr = validate(model, tokenizer, valid_data, args.max_length,
                                       args.validation_batch_size, device, use_bf16, args.k)
                if current_mrr > best_mrr:
                    best_mrr, best_step = current_mrr, step
                    save_best(args.output, model, tokenizer, step, current_mrr, signature)
                print(f"validation step={step} mrr@{args.k}={current_mrr:.6f} "
                      f"best={best_mrr:.6f}@{best_step}", flush=True)
            if step % args.save_steps == 0 or step >= total_steps:
                state = {"global_step": step, "epoch": epoch, "batch_index": batch_index,
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                         "rng_state": capture_rng_state(), "signature": signature,
                         "best_mrr": best_mrr, "best_step": best_step,
                         "validation_mrr@100": current_mrr, "recent_losses": list(recent_losses)}
                saved = save_checkpoint(output=args.output, global_step=step, model=model,
                                        tokenizer=tokenizer, trainer_state=state,
                                        save_total_limit=args.save_total_limit)
                print(f"Saved checkpoint: {saved}", flush=True)
            if step >= total_steps: stop = True; break
        start_batch = 0
    model.save_pretrained(args.output); tokenizer.save_pretrained(args.output)
    result = {"steps": step, "epochs": args.epochs, "best_step": best_step,
              "best_validation_mrr@100": best_mrr,
              "best_model": str((args.output / "best").resolve())}
    (args.output / "training_complete.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
