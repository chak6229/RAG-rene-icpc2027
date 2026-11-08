"""Train the published SSQR pooler objective from CodeBERT base, separately from CLS models."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .common import MODEL_OUTPUT, OUT, ROOT, read_json, sha256, write_json
from .retriever import REPRESENTATION, full_batch_loss, pooler_vectors

CACHE = OUT / "codexglue_tokens"
SOURCE_PATHS = {"train": ROOT / "data/codexglue/training/train.jsonl",
                "valid": ROOT / "data/codexglue/advtest/valid.jsonl"}
COUNTS = {"train": 251820, "valid": 9604}
BASE = ROOT / "models/codebert-base"
MAX_LENGTH = 256


def normalized_tokens(row, key):
    value = row.get(key)
    if not isinstance(value, (list, str)) or not value:
        raise ValueError(f"CodeXGLUE training row lacks {key}")
    result = " ".join(value.split()) if isinstance(value, str) else " ".join(map(str, value))
    if not result.strip():
        raise ValueError(f"empty {key}")
    return result


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / "MANIFEST.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for split, source in SOURCE_PATHS.items():
            if sha256(source) != manifest["sources"][split]["sha256"]:
                raise ValueError("training source changed after token preparation")
        for item in manifest["files"]:
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("existing token cache changed")
        return manifest
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    input_hashes = {}
    for split, source in SOURCE_PATHS.items():
        matrices = {name: np.lib.format.open_memmap(CACHE / f"{split}_{name}.npy", mode="w+", dtype=np.uint16,
                                                  shape=(COUNTS[split], MAX_LENGTH)) for name in ("code", "query")}
        batch, offset = [], 0
        digest = hashlib.sha256()

        def consume():
            nonlocal offset
            if not batch:
                return
            codes = [normalized_tokens(row, "code_tokens" if row.get("code_tokens") else "function_tokens") for row in batch]
            queries = [normalized_tokens(row, "docstring_tokens") for row in batch]
            for name, texts in (("code", codes), ("query", queries)):
                encoded = tokenizer(texts, max_length=MAX_LENGTH, padding="max_length", truncation=True)["input_ids"]
                ids = np.asarray(encoded, dtype=np.int64)
                if ids.min() < 0 or ids.max() > np.iinfo(np.uint16).max:
                    raise ValueError("token cache uint16 range exceeded")
                matrices[name][offset:offset + len(batch)] = ids.astype(np.uint16)
            offset += len(batch)
            batch.clear()

        with source.open("rb") as handle:
            for line in handle:
                digest.update(line)
                if line.strip():
                    batch.append(json.loads(line))
                if len(batch) == 512:
                    consume()
            consume()
        if offset != COUNTS[split]:
            raise ValueError(f"unexpected {split} row count {offset}")
        for matrix in matrices.values():
            matrix.flush()
        input_hashes[split] = {"path": source.relative_to(ROOT).as_posix(), "sha256": digest.hexdigest(), "rows": offset}
        print(json.dumps({"stage": "tokenized", "split": split, "rows": offset}, ensure_ascii=False), flush=True)
    files = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in sorted(CACHE.glob("*.npy"))]
    manifest = {"sources": input_hashes, "files": files, "max_length": MAX_LENGTH,
                "representation": "code_tokens/function_tokens and docstring_tokens from CodeXGLUE train/valid only",
                "cosqa_or_advtest_test_read": False}
    # Bind against the completed data-source audit rather than trusting paths.
    inventory = read_json(ROOT / "results/rene_cosqa_ssqr_audit_20261019/OVERLAP_AUDIT.json")["inputs"]
    for record in input_hashes.values():
        original = next(x for x in inventory if x["path"] == record["path"])
        if original["sha256"] != record["sha256"]:
            raise ValueError("training input differs from audited source")
    write_json(manifest_path, manifest, immutable=True)
    return manifest


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup(precision):
    seed_everything(123456)
    model = AutoModel.from_pretrained(BASE, local_files_only=True).to("cuda")
    # Preserve original B=32 and its 32-way negatives on 8GB hardware.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8, weight_decay=0.0)
    return model, optimizer


def context(precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else contextlib.nullcontext()


def arrays(split):
    return tuple(np.load(CACHE / f"{split}_{name}.npy", mmap_mode="r") for name in ("code", "query"))


def tensors(data, indices):
    return tuple(torch.from_numpy(np.asarray(matrix[indices], dtype=np.int64)).to("cuda") for matrix in data)


def update(model, optimizer, data, indices, precision):
    optimizer.zero_grad(set_to_none=True)
    code, query = tensors(data, indices)
    with context(precision):
        loss, _, _ = full_batch_loss(model, code, query)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite training loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("nonfinite gradient")
    optimizer.step()
    return float(loss), float(grad_norm)


def benchmark(precision, steps):
    if steps < 5:
        raise ValueError("benchmark needs at least five steps, including three warmup steps")
    prepare()
    model, optimizer = setup(precision)
    data = arrays("train")
    times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for index in range(steps):
        ids = np.arange(index * 32, (index + 1) * 32)
        torch.cuda.synchronize()
        began = time.perf_counter()
        loss, _ = update(model, optimizer, data, ids, precision)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        if index >= 3:
            times.append(elapsed)
        losses.append(loss)
    torch.cuda.synchronize()
    validation_began = time.perf_counter()
    validation_metrics = validate(model, arrays("valid"), precision)
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - validation_began
    result = {"device": torch.cuda.get_device_name(), "precision": precision, "full_pair_batch_size": 32,
              "in_batch_scores": [32, 32], "gradient_checkpointing": True, "steps": steps,
              "warmup_steps_discarded": 3, "median_seconds_per_step": float(np.median(times)),
              "mean_seconds_per_step": float(np.mean(times)), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "losses": losses, "projected_update_hours": float(np.mean(times)) * 15740 / 3600,
              "one_validation_seconds": validation_seconds, "validation_metrics": validation_metrics,
              "projected_training_and_validation_hours": (float(np.mean(times)) * 15740 + validation_seconds * 20) / 3600,
              "excludes": "data preparation and checkpoint I/O; benchmark weights discarded before fresh training"}
    write_json(OUT / "TRAINING_BENCHMARK.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def validate(model, data, precision):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for matrix in data:
            vectors = []
            for start in range(0, len(matrix), 32):
                ids = torch.from_numpy(np.asarray(matrix[start:start + 32], dtype=np.int64)).to("cuda")
                with context(precision):
                    hidden = pooler_vectors(model(input_ids=ids, attention_mask=ids.ne(1)))
                vectors.append(hidden.float().cpu().numpy())
            outputs.append(np.concatenate(vectors))
    codes, queries = outputs
    rr_source, rr_stable100 = [], []
    doc_ids = np.arange(len(codes))
    for start in range(0, len(queries), 128):
        scores = queries[start:start + 128] @ codes.T
        if not np.isfinite(scores).all():
            raise FloatingPointError("nonfinite validation scores")
        target_ids = np.arange(start, start + len(scores))
        target_scores = scores[np.arange(len(scores)), target_ids]
        # Original evaluate() pessimistically counts all equal-scored other docs.
        source_ranks = np.count_nonzero(scores >= target_scores[:, None], axis=1)
        stable_ranks = 1 + np.count_nonzero((scores > target_scores[:, None]) |
                        ((scores == target_scores[:, None]) & (doc_ids[None, :] < target_ids[:, None])), axis=1)
        rr_source.extend((1.0 / source_ranks).tolist())
        rr_stable100.extend(np.where(stable_ranks <= 100, 1.0 / stable_ranks, 0.0).tolist())
    model.train()
    return {"source_full_mrr_pessimistic_ties": float(np.mean(rr_source)),
            "stable_mrr_at_100": float(np.mean(rr_stable100)), "queries": len(queries)}


def training(precision):
    data_manifest = prepare()
    output = MODEL_OUTPUT.resolve()
    if output.parent != (ROOT / "models").resolve() or output.name != "codebert-ssqr-pooler-2epoch-20261019":
        raise ValueError("unexpected training output directory")
    if (output / "training_complete.json").exists():
        print("Training already complete; existing checkpoint preserved", flush=True)
        return
    total_steps, save_steps = math.ceil(COUNTS["train"] / 32) * 2, math.ceil(COUNTS["train"] / 32) // 10
    source_script = ROOT / "results/rene_cosqa_ssqr_audit_20261019/selected_source/ssqr/Query Reformulation/code/model.py"
    contract = {**REPRESENTATION, "base_model": "models/codebert-base", "base_weights_sha256": sha256(BASE / "pytorch_model.bin"),
                "train_input": data_manifest["sources"]["train"], "validation_input": data_manifest["sources"]["valid"],
                "batch_size": 32, "epochs": 2, "total_steps": total_steps, "max_length": 256, "seed": 123456,
                "optimizer": "AdamW", "learning_rate": 5e-5, "weight_decay": 0.0, "adam_epsilon": 1e-8,
                "scheduler": "linear_10percent_warmup", "warmup_steps": int(total_steps * .1),
                "gradient_clipping": 1.0, "precision": precision, "gradient_checkpointing": True,
                "validation_every_steps": save_steps, "main_checkpoint": "final_two_epochs",
                "best_checkpoint": "selected only by CodeXGLUE valid source full MRR, never CoSQA",
                "source_model_sha256": sha256(source_script), "trainer_sha256": sha256(Path(__file__)),
                "retriever_sha256": sha256(Path(__file__).with_name("retriever.py")),
                "python_version": sys.version, "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "sampler": "explicit torch.randperm CPU generator; seed=123456+epoch; saved next_batch",
                "paper_source_choice": "published CodeBERT objective and source linear schedule; paper constant-LR discrepancy recorded",
                "runtime_difference": "modern PyTorch/transformers, bf16 and layer activation checkpointing; full BxB objective unchanged"}
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 14 * 2**30:
        raise OSError("at least 14 GiB free is required to retain all new checkpoints and resume state")
    write_json(output / "TRAINING_CONTRACT.json", contract, immutable=True)
    write_json(OUT / "TRAINING_CONTRACT.json", contract, immutable=True)
    model, optimizer = setup(precision)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * .1), num_training_steps=total_steps)
    train_data, valid_data = arrays("train"), arrays("valid")
    step = start_epoch = start_batch = 0
    best_mrr, best_step = -1.0, 0
    recent_losses = deque(maxlen=50)
    state_file = output / "trainer_state.pt"
    if state_file.exists():
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        if state["contract"] != contract:
            raise ValueError("resume training contract mismatch")
        weights_path = output / state["weights_file"]
        if sha256(weights_path) != state["weights_sha256"]:
            raise ValueError("resume checkpoint weights changed")
        model.load_state_dict(load_file(str(weights_path), device="cpu"))
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to("cuda")
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        step, start_epoch, start_batch = state["step"], state["epoch"], state["next_batch"]
        best_mrr, best_step = state["best_mrr"], state["best_step"]
        del state
    began = time.perf_counter()
    write_json(OUT / "TRAINING_STATUS.json", {"status": "running", "step": step, "total_steps": total_steps, "started_utc": datetime.now(timezone.utc).isoformat()})

    def save(epoch, next_batch):
        nonlocal best_mrr, best_step
        metrics = validate(model, valid_data, precision)
        current = metrics["source_full_mrr_pessimistic_ties"]
        if current > best_mrr:
            best_mrr, best_step = current, step
            best = output / "best"
            model.save_pretrained(best)
            tokenizer.save_pretrained(best)
            write_json(best / "ssqr_representation_contract.json", contract)
            write_json(best / "best_checkpoint.json", {"step": step, "validation": metrics})
        checkpoint = output / f"checkpoint-{step}"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint / "ssqr_representation_contract.json", contract)
        # A single resume state is overwritten atomically; checkpoint weights are retained.
        weights_file = f"checkpoint-{step}/model.safetensors"
        state = {"contract": contract, "weights_file": weights_file, "weights_sha256": sha256(output / weights_file),
                 "optimizer": optimizer.state_dict(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                 "step": step, "epoch": epoch, "next_batch": next_batch, "best_mrr": best_mrr, "best_step": best_step}
        pending = output / "trainer_state.pending.pt"
        torch.save(state, pending)
        assert pending.resolve().parent == state_file.resolve().parent == output
        pending.replace(state_file)
        record = {"step": step, "validation": metrics, "best_mrr": best_mrr, "best_step": best_step,
                  "elapsed_seconds": time.perf_counter() - began}
        with (OUT / "TRAINING_VALIDATION.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    for epoch in range(start_epoch, 2):
        generator = torch.Generator().manual_seed(123456 + epoch)
        order = torch.randperm(COUNTS["train"], generator=generator).numpy()
        batch_count = math.ceil(len(order) / 32)
        for batch in range(start_batch if epoch == start_epoch else 0, batch_count):
            ids = order[batch * 32:(batch + 1) * 32]
            loss, grad_norm = update(model, optimizer, train_data, ids, precision)
            scheduler.step()
            step += 1
            recent_losses.append(loss)
            if step % 50 == 0:
                status = {"status": "running", "step": step, "total_steps": total_steps, "epoch": epoch + 1,
                          "loss_last50": float(np.mean(recent_losses)), "gradient_norm": grad_norm,
                          "learning_rate": scheduler.get_last_lr()[0], "elapsed_session_seconds": time.perf_counter() - began,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                write_json(OUT / "TRAINING_STATUS.json", status)
                print(json.dumps(status), flush=True)
            if step % save_steps == 0 or step == total_steps:
                save(epoch, batch + 1)
        start_batch = 0
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    write_json(output / "ssqr_representation_contract.json", contract)
    result = {"status": "complete", "step": step, "total_steps": total_steps, "epochs": 2,
              "final_checkpoint": (output / f"checkpoint-{step}").relative_to(ROOT).as_posix(),
              "final_weights_sha256": sha256(output / "model.safetensors"), "best_step": best_step,
              "best_source_validation_mrr": best_mrr, "elapsed_session_seconds": time.perf_counter() - began,
              "cosqa_effects_evaluated": False, "contract_sha256": sha256(output / "TRAINING_CONTRACT.json")}
    if step != total_steps:
        raise ValueError("training did not complete both epochs")
    write_json(output / "training_complete.json", result, immutable=True)
    write_json(OUT / "TRAINING_STATUS.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "benchmark", "train"))
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--benchmark-steps", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False))
    elif args.mode == "benchmark":
        benchmark(args.precision, args.benchmark_steps)
    else:
        training(args.precision)


if __name__ == "__main__":
    main()
