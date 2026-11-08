#!/usr/bin/env python3
"""Add exact Top-100 retrieval-feedback features to an RRF utility dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_rewrite_utility_dataset import code_text, encode, feature_row, topk
from vac_mqr import ENHANCED_RETRIEVAL_FEATURES


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def weighted_rank_agreement(left: np.ndarray, right: np.ndarray, k: int = 60) -> float:
    left_rank = {int(index): rank for rank, index in enumerate(left, 1)}
    right_rank = {int(index): rank for rank, index in enumerate(right, 1)}
    common = set(left_rank) & set(right_rank)
    numerator = sum(min(1.0 / (k + left_rank[index]),
                        1.0 / (k + right_rank[index])) for index in common)
    denominator = sum(1.0 / (k + rank) for rank in range(1, len(left) + 1))
    return float(numerator / denominator) if denominator else 0.0


def enhanced_features(query: str, candidate: dict, query_vector: np.ndarray,
                      candidate_vector: np.ndarray, original_scores: np.ndarray,
                      candidate_scores: np.ndarray) -> dict[str, float]:
    values = feature_row(query, candidate, query_vector, candidate_vector,
                         original_scores, candidate_scores)
    original_100, candidate_100 = topk(original_scores, 100), topk(candidate_scores, 100)
    original_10, candidate_10 = original_100[:10], candidate_100[:10]
    original_values, candidate_values = original_scores[original_100], candidate_scores[candidate_100]
    correlation = float(np.corrcoef(original_scores, candidate_scores)[0, 1])
    values.update({
        "top100_overlap": len(set(original_100.tolist()) & set(candidate_100.tolist())) / 100.0,
        "top100_rrf_agreement": weighted_rank_agreement(original_100, candidate_100),
        "top10_rank_stability": weighted_rank_agreement(original_10, candidate_10),
        "score_distribution_correlation": correlation if np.isfinite(correlation) else 0.0,
        "original_top100_mean": float(original_values.mean()),
        "original_top100_std": float(original_values.std()),
        "candidate_top100_mean": float(candidate_values.mean()),
        "candidate_top100_std": float(candidate_values.std()),
        "top100_mean_delta": float(candidate_values.mean() - original_values.mean()),
        "top100_std_delta": float(candidate_values.std() - original_values.std()),
        "top1_document_stable": float(original_100[0] == candidate_100[0]),
    })
    return {name: float(values[name]) for name in ENHANCED_RETRIEVAL_FEATURES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--source-utility", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer

    valid = read_jsonl(args.valid)
    rows = read_jsonl(args.source_utility)
    if args.max_queries:
        rows = rows[:args.max_queries]
    if any(row["split_role"] == "test" for row in rows):
        raise RuntimeError("test rows are forbidden")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    code_vectors = encode(model, tokenizer, [code_text(row) for row in valid],
                          args.batch_size, args.max_length, device)
    query_vectors = encode(model, tokenizer, [row["query"] for row in rows],
                           args.batch_size, args.max_length, device)
    candidates = [candidate for row in rows for candidate in row["candidates"]]
    candidate_vectors = encode(model, tokenizer, [candidate["text"] for candidate in candidates],
                               args.batch_size, args.max_length, device)
    output, offset = [], 0
    for local, row in enumerate(rows):
        query_index = int(row["query_index"])
        original_scores = code_vectors @ query_vectors[local]
        rewritten = []
        for candidate in row["candidates"]:
            vector = candidate_vectors[offset]
            offset += 1
            candidate_scores = code_vectors @ vector
            rewritten.append({**candidate, "features": enhanced_features(
                row["query"], candidate, query_vectors[local], vector,
                original_scores, candidate_scores)})
        reference = dict(row["reference_item"])
        reference_candidate = {"text": row["query"], "information_gain": 0.0,
                               "rank": 0, "constraint_preservation": 1.0}
        reference["features"] = enhanced_features(
            row["query"], reference_candidate, query_vectors[local], query_vectors[local],
            original_scores, original_scores)
        output.append({**row, "candidates": rewritten, "reference_item": reference})
        if len(output) % 250 == 0:
            print(f"enhanced {len(output)}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"format": "vac-mqr-enhanced-retrieval-feedback-v1",
                "query_count": len(output), "features": list(ENHANCED_RETRIEVAL_FEATURES),
                "source_utility_sha256": digest(args.source_utility),
                "valid_sha256": digest(args.valid), "model_config_sha256": digest(args.model / "config.json"),
                "test_used": False, "smoke_max_queries": args.max_queries,
                "output": str(args.output.resolve())}
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
