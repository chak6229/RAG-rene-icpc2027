#!/usr/bin/env python3
"""Build leakage-safe per-query delta-RR labels with a frozen CodeBERT retriever."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vac_mqr import UTILITY_FEATURES


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def code_text(row: dict) -> str:
    tokens = row.get("code_tokens") or row.get("function_tokens")
    if not tokens:
        raise ValueError("missing code tokens; docstring fallback is forbidden")
    return " ".join(map(str, tokens))


def encode(model, tokenizer, texts: list[str], batch_size: int, max_length: int,
           device: str) -> np.ndarray:
    import torch
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda" and torch.cuda.is_bf16_supported()):
                hidden = model(**batch).last_hidden_state[:, 0, :]
            result.append(torch.nn.functional.normalize(hidden.float(), dim=1).cpu().numpy())
    return np.vstack(result).astype(np.float32)


def reciprocal_rank(scores: np.ndarray, relevant: int, k: int) -> float:
    rank = 1 + int(np.count_nonzero(scores > scores[relevant]))
    return 1.0 / rank if rank <= k else 0.0


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    count = min(k, len(scores))
    values = np.argpartition(-scores, count - 1)[:count]
    return values[np.argsort(-scores[values])]


def token_jaccard(left: str, right: str) -> float:
    a, b = set(left.casefold().split()), set(right.casefold().split())
    return len(a & b) / len(a | b) if a or b else 1.0


def score_entropy(scores: np.ndarray) -> float:
    shifted = scores - scores.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def utility_grade(delta_rr: float) -> int:
    if delta_rr >= .05:
        return 2
    if delta_rr > 0.0:
        return 1
    if delta_rr == 0.0:
        return 0
    return -1


def fixed_top3(row: dict) -> list[dict]:
    candidates = list(row.get("candidates", [])[:3])
    while len(candidates) < 3:
        slot = len(candidates) + 1
        candidates.append({
            "text": row["query"], "information_gain": -1e9, "position": -1,
            "span": "", "rank": slot, "quality_pass": False,
            "quality_reasons": ["unavailable_candidate_slot"],
            "constraint_preservation": 1.0, "negation_preserved": True,
            "unavailable": True,
        })
    return candidates


def feature_row(query: str, candidate: dict, query_vector: np.ndarray,
                candidate_vector: np.ndarray, original_scores: np.ndarray,
                candidate_scores: np.ndarray) -> dict[str, float]:
    original_top = topk(original_scores, 10)
    candidate_top = topk(candidate_scores, 10)
    original_margin = float(original_scores[original_top[0]] - original_scores[original_top[1]])
    candidate_margin = float(candidate_scores[candidate_top[0]] - candidate_scores[candidate_top[1]])
    original_tokens, candidate_tokens = query.split(), str(candidate["text"]).split()
    original_top_scores = original_scores[original_top]
    candidate_top_scores = candidate_scores[candidate_top]
    values = {
        "information_gain": float(candidate.get("information_gain", 0.0)),
        "query_candidate_cosine": float(query_vector @ candidate_vector),
        "token_jaccard": token_jaccard(query, candidate["text"]),
        "length_ratio": len(candidate_tokens) / max(len(original_tokens), 1),
        "original_top1": float(original_scores[original_top[0]]),
        "candidate_top1": float(candidate_scores[candidate_top[0]]),
        "top1_delta": float(candidate_scores[candidate_top[0]] - original_scores[original_top[0]]),
        "original_margin": original_margin, "candidate_margin": candidate_margin,
        "top10_mean_delta": float(candidate_scores[candidate_top].mean()
                                  - original_scores[original_top].mean()),
        "top10_overlap": len(set(original_top.tolist()) & set(candidate_top.tolist())) / 10.0,
        "original_top10_mean": float(original_top_scores.mean()),
        "original_top10_std": float(original_top_scores.std()),
        "original_top10_range": float(original_top_scores.max() - original_top_scores.min()),
        "original_top10_entropy": score_entropy(original_top_scores),
        "candidate_top10_mean": float(candidate_top_scores.mean()),
        "candidate_top10_std": float(candidate_top_scores.std()),
        "candidate_top10_range": float(candidate_top_scores.max() - candidate_top_scores.min()),
        "candidate_top10_entropy": score_entropy(candidate_top_scores),
        "top10_std_delta": float(candidate_top_scores.std() - original_top_scores.std()),
        "top10_entropy_delta": score_entropy(candidate_top_scores) - score_entropy(original_top_scores),
        "constraint_preservation": float(candidate["constraint_preservation"]),
        "candidate_rank": float(candidate["rank"]),
    }
    assert tuple(values) == UTILITY_FEATURES
    return values


def upgrade_feature_row(features: dict[str, float]) -> dict[str, float]:
    """Add the post-retrieval QPP fields derivable from legacy Top-10 summaries."""
    original_mean = float(features["candidate_top1"] - features["top1_delta"]
                          - features["top10_mean_delta"])
    candidate_mean = original_mean + float(features["top10_mean_delta"])
    original_std = max(float(features["original_margin"]), 0.0)
    candidate_std = max(float(features["candidate_margin"]), 0.0)
    upgraded = dict(features)
    upgraded.update({
        "original_top10_mean": original_mean,
        "original_top10_std": original_std,
        "original_top10_range": original_std,
        "original_top10_entropy": 0.0,
        "candidate_top10_mean": candidate_mean,
        "candidate_top10_std": candidate_std,
        "candidate_top10_range": candidate_std,
        "candidate_top10_entropy": 0.0,
        "top10_std_delta": candidate_std - original_std,
        "top10_entropy_delta": 0.0,
    })
    return {name: float(upgraded[name]) for name in UTILITY_FEATURES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--validated-rewrites", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--test-used", action="store_true",
                        help="Mark this artifact as test processing; does not affect feature construction.")
    parser.add_argument("--max-per-role", type=int, default=0,
                        help="Smoke-only cap; zero builds all 9,604 queries.")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer
    rows, rewrite_rows = read_jsonl(args.valid), read_jsonl(args.validated_rewrites)
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if len(rows) != len(rewrite_rows):
        raise RuntimeError("valid and rewrite counts differ")
    role_for_index = {}
    selected_indices = []
    for role, detail in split["roles"].items():
        indices = list(range(detail["start"], detail["end_exclusive"]))
        if args.max_per_role:
            indices = indices[:args.max_per_role]
        for index in indices:
            role_for_index[index] = role
        selected_indices.extend(indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    code_vectors = encode(model, tokenizer, [code_text(row) for row in rows],
                          args.batch_size, args.max_length, device)
    queries = [rewrite_rows[index]["query"] for index in selected_indices]
    query_vectors = encode(model, tokenizer, queries, args.batch_size, args.max_length, device)
    fixed_candidates = {index: fixed_top3(rewrite_rows[index]) for index in selected_indices}
    flat_candidates = [candidate["text"] for index in selected_indices
                       for candidate in fixed_candidates[index]]
    candidate_vectors = encode(model, tokenizer, flat_candidates, args.batch_size,
                               args.max_length, device) if flat_candidates else np.empty((0, 768))

    output_rows, offset = [], 0
    for local_index, source_index in enumerate(selected_indices):
        row = rewrite_rows[source_index]
        if row["url"] != rows[source_index]["url"]:
            raise RuntimeError(f"URL alignment mismatch at {source_index}")
        query_vector = query_vectors[local_index]
        original_scores = code_vectors @ query_vector
        original_rr = reciprocal_rank(original_scores, source_index, args.k)
        labelled = []
        candidates = fixed_candidates[source_index]
        for candidate_rank, candidate in enumerate(candidates, 1):
            candidate = {"rank": candidate_rank, **candidate}
            vector = candidate_vectors[offset]; offset += 1
            scores = code_vectors @ vector
            candidate_rr = reciprocal_rank(scores, source_index, args.k)
            delta = candidate_rr - original_rr
            labelled.append({**candidate, "item_type": "rewrite", "is_reference": False,
                             "features": feature_row(row["query"], candidate, query_vector,
                                                     vector, original_scores, scores),
                             "rr_at_100": candidate_rr, "delta_rr": delta,
                             "utility_grade": utility_grade(delta),
                             "harmful": delta < 0.0, "positive_gain": delta > 0.0})
        reference_candidate = {
            "text": row["query"], "information_gain": 0.0, "rank": 0,
            "constraint_preservation": 1.0,
        }
        reference = {**reference_candidate, "item_type": "reference", "is_reference": True,
                     "quality_pass": True, "quality_reasons": [], "negation_preserved": True,
                     "features": feature_row(row["query"], reference_candidate, query_vector,
                                             query_vector, original_scores, original_scores),
                     "rr_at_100": original_rr, "delta_rr": 0.0, "utility_grade": 0,
                     "harmful": False, "positive_gain": False}
        qualified = [item for item in labelled if item["quality_pass"]]
        output_rows.append({"query_index": source_index, "url": row["url"],
                            "query": row["query"], "split_role": role_for_index[source_index],
                            "original_rr_at_100": original_rr,
                            "needs_rewrite": any(item["positive_gain"] for item in qualified),
                            "reference_item": reference, "candidates": labelled})
        if len(output_rows) % 250 == 0:
            print(f"scored {len(output_rows)}/{len(selected_indices)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "format": "vac-mqr-delta-rr-v3", "query_count": len(output_rows),
        "candidate_count": sum(len(row["candidates"]) for row in output_rows),
        "roles": {role: sum(row["split_role"] == role for row in output_rows)
                  for role in split["roles"]},
        "candidate_policy": "exactly three rewrites per query plus the original reference item",
        "label": "graded utility: +2 if delta_rr>=0.05; +1 if 0<delta_rr<0.05; 0 if delta_rr=0; -1 if delta_rr<0",
        "needs_rewrite": "exists quality-passing candidate with delta_rr > 0",
        "features": list(UTILITY_FEATURES),
        "feature_leakage_policy": "URL/relevant index/RR/delta-RR/harmful are labels only and absent from features",
        "code_representation": "code_tokens/function_tokens only; raw function/docstring forbidden",
        "model": str(args.model.resolve()), "model_config_sha256": sha256(args.model / "config.json"),
        "valid_sha256": sha256(args.valid), "rewrites_sha256": sha256(args.validated_rewrites),
        "split_manifest_sha256": sha256(args.split_manifest), "test_used": args.test_used,
        "smoke_max_per_role": args.max_per_role, "output": str(args.output.resolve())}
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
