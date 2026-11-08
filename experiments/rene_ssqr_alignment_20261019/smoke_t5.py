"""Real-weight synthetic smoke; no CoSQA generation, retrieval labels or effect metrics."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import entr, softmax
from transformers import AutoModelForSeq2SeqLM

from .common import OUT, ROOT, sha256, write_json
from .t5 import GenerationSettings, SCORING_CONTRACT, generate_candidates, load_tokenizer, select_candidates, teacher_forced_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = time.perf_counter()
    path = ROOT / "models/t5-cqc-codenn-1m/best"
    tokenizer = load_tokenizer(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to(args.device).eval()
    masked = ["python <extra_id_0> list", "sort dictionary <extra_id_0> keys", "remove <extra_id_0> empty items from a list"]
    spans = ["iterate", "", "all remaining"]
    batch = teacher_forced_scores(model, tokenizer, masked, spans, batch_size=3)
    individual = [teacher_forced_scores(model, tokenizer, [query], [span], batch_size=1)[0]
                  for query, span in zip(masked, spans)]
    max_batch_difference = 0.
    for a, b in zip(batch, individual):
        for key in a:
            if a[key] is not None:
                max_batch_difference = max(max_batch_difference, abs(a[key] - b[key]))
    if max_batch_difference > 5e-5:
        raise AssertionError(f"batch/individual scores differ: {max_batch_difference}")
    manual = []
    with torch.inference_mode():
        for query, span in zip(masked, spans):
            inputs = tokenizer(query, return_tensors="pt").to(args.device)
            labels = tokenizer(f"<extra_id_0> {span} <extra_id_1>", return_tensors="pt").input_ids.to(args.device)
            logits = model(**inputs, labels=labels).logits[0].float().cpu().numpy().astype(np.float64)
            manual.append(float(entr(softmax(logits, axis=-1)).sum(axis=-1).mean()))
    max_manual_difference = max(abs(row["full_vocab_entropy"] - value) for row, value in zip(batch, manual))
    if max_manual_difference > 5e-6:
        raise AssertionError(f"FP64 independent entropy disagrees: {max_manual_difference}")
    settings = GenerationSettings(batch_size=4)
    candidates = generate_candidates(model, tokenizer, "synthetic-smoke-q1", "sort dictionary keys", settings)
    selected = {policy: [r["position"] for r in select_candidates(candidates, policy=policy)] for policy in ("ENTR", "PROB", "RAND")}
    if len(candidates) != 3 or any(len(set(ids)) != 3 for ids in selected.values()):
        raise AssertionError("synthetic position selection lost or duplicated slots")
    report = {"passed": True, "device": args.device, "synthetic_only": True, "cosqa_queries_used": 0,
              "retrieval_effect_evaluated": False, "model": path.relative_to(ROOT).as_posix(),
              "model_weights_sha256": sha256(path / "model.safetensors"), "vocab_size": model.config.vocab_size,
              "tokenizer": type(tokenizer).__name__, "max_batch_individual_difference": max_batch_difference,
              "max_independent_fp64_entropy_difference": max_manual_difference,
              "scoring_contract": SCORING_CONTRACT, "generation": settings.contract(),
              "manual_entropy_cases": [{"span": span, "metrics": row, "independent_entropy": value}
                                       for span, row, value in zip(spans, batch, manual)],
              "candidates": candidates, "selected_positions": selected,
              "elapsed_seconds": time.perf_counter() - start, "scorer_sha256": sha256(Path(__file__).with_name("t5.py")),
              "sentencepiece_sha256": sha256(ROOT / "models/t5-base/spiece.model")}
    write_json(OUT / "T5_REAL_WEIGHT_SMOKE.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "vocab_size", "max_batch_individual_difference", "max_independent_fp64_entropy_difference", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
