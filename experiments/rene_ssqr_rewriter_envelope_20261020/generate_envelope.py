"""Multi-rewriter envelope: local CQC T5 vs t5-base on the frozen 512-query subset.

The local arm reuses the frozen bridge candidates/rankings. The t5-base arm
generates at the same positions with the same greedy profile, scores with the
same full-vocabulary teacher-forcing entropy, and ranks with the same CodeBERT
pooler retriever.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = r"."
BRIDGE = os.path.join(ROOT, r"results\rene_ssqr_bridge_20261019")
DEV = os.path.join(ROOT, r"results\rene_cosqa_development_20261019")
OUT = os.path.join(ROOT, r"results\rene_ssqr_rewriter_envelope_20261020")
CODEBERT = os.path.join(ROOT, r"models\codebert-ssqr-pooler-2epoch-20261019")
T5_BASE = os.path.join(ROOT, r"models\t5-base")
T5_LOCAL = os.path.join(ROOT, r"models\t5-cqc-codenn-1m\best")

sys.path.insert(0, ROOT)
from experiments.rene_ssqr_alignment_20261019.t5 import load_tokenizer  # noqa: E402
from experiments.rene_ssqr_alignment_20261019.retriever import encode_texts, dot_product_scores  # noqa: E402
from experiments.rene_ssqr_alignment_20261019.pipeline import load_trained_retriever  # noqa: E402
from experiments.rene_ssqr_bridge_20261019.generate import generate_positions  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rr(ranked, target):
    for position, doc_id in enumerate(ranked, start=1):
        if int(doc_id) == int(target):
            return 1.0 / position
    return 0.0


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else None


def domain_paths(domain):
    if domain == "source":
        return {
            "queries": os.path.join(BRIDGE, "data/source/queries.jsonl"),
            "corpus": os.path.join(BRIDGE, "data/source/corpus.jsonl"),
            "qrels": os.path.join(BRIDGE, "data/private/source_qrels.jsonl"),
            "candidates": os.path.join(BRIDGE, "candidates/source/baseline.jsonl"),
            "rankings": os.path.join(BRIDGE, "rankings/source/CodeBERT.jsonl"),
            "code_vectors": os.path.join(BRIDGE, "retrieval_cache/source/CodeBERT/code_vectors.npy"),
        }
    return {
        "queries": os.path.join(DEV, "data/queries.jsonl"),
        "corpus": os.path.join(ROOT, r"results\rene_ssqr_alignment_20261019\cosqa_adapter\public\corpus.jsonl"),
        "qrels": os.path.join(BRIDGE, "data/private/cosqa_qrels.jsonl"),
        "candidates": os.path.join(DEV, "candidates.jsonl"),
        "rankings": os.path.join(BRIDGE, "rankings/cosqa/CodeBERT.jsonl"),
        "code_vectors": os.path.join(ROOT, r"results\rene_cosqa_pilot_20261019\retrieval_cache\CodeBERT\code_vectors.npy"),
    }


def generate_t5base(domain, ids, queries):
    import torch
    from transformers import AutoModelForSeq2SeqLM
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = load_tokenizer(T5_BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_BASE, local_files_only=True).float().cuda().eval()
    rows = []
    started = time.perf_counter()
    for index, query_id in enumerate(ids, start=1):
        text = queries[query_id]
        positions = set(range(1, len(text.split()) + 1))
        rows.extend(generate_positions(model, tokenizer, query_id, text, positions=positions,
                                      sample=False, max_new_tokens=19))
        if index % 32 == 0 or index == len(ids):
            print({"stage": "generate", "domain": domain, "rewriter": "t5base",
                   "queries": index, "total": len(ids),
                   "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def rank_texts(domain, ids, queries, candidate_rows):
    import torch
    torch.set_num_threads(4)
    paths = domain_paths(domain)
    corpus = load_jsonl(paths["corpus"])
    doc_ids = np.array([row["doc_id"] for row in corpus], dtype=np.int64)
    code_vectors = np.load(paths["code_vectors"]).astype(np.float32)
    if code_vectors.shape != (len(corpus), 768):
        raise ValueError(f"code vector shape mismatch: {code_vectors.shape} vs {len(corpus)}")
    texts = []
    for query_id in ids:
        texts.append(queries[query_id])
    texts.extend(row["rewrite"] for row in candidate_rows)
    unique = list(dict.fromkeys(texts))
    model, tokenizer = load_trained_retriever(Path(CODEBERT), "cuda")
    vectors = encode_texts(model, tokenizer, unique, batch_size=16, device="cuda")
    del model
    torch.cuda.empty_cache()
    ranked = {}
    for offset in range(0, len(unique), 128):
        current = unique[offset:offset + 128]
        scores = dot_product_scores(vectors[offset:offset + len(current)], code_vectors)
        for text, values in zip(current, scores):
            order = np.lexsort((doc_ids, -values))[:100]
            ranked[text_id(text)] = doc_ids[order].tolist()
    return ranked


def ladder(domain, rewriter, ids, queries, qrels, rankings, candidates):
    stats = []
    missing = 0
    for query_id in ids:
        rows = candidates.get(query_id)
        target = qrels.get(query_id)
        if not rows or target is None:
            missing += 1
            continue
        nowrite = rr(rankings.get(text_id(queries[query_id]), []), target)
        scored = []
        for row in rows:
            ranked = rankings.get(text_id(row["rewrite"]))
            if ranked is None:
                continue
            scored.append({
                "entropy": float(row["full_vocab_entropy"]),
                "rr": rr(ranked, target),
            })
        if len(scored) < 3:
            missing += 1
            continue
        by_entropy = sorted(scored, key=lambda item: item["entropy"])
        entropies = [item["entropy"] for item in scored]
        rrs = [item["rr"] for item in scored]
        stats.append({
            "query_id": query_id,
            "nowrite": nowrite,
            "oracle": max(rrs),
            "entr_top1": by_entropy[0]["rr"],
            "entr_best3": max(item["rr"] for item in by_entropy[:3]),
            "spearman_entropy": spearman(entropies, rrs),
        })

    def mean(key):
        values = [row[key] for row in stats if row[key] is not None]
        return sum(values) / len(values) if values else None

    result = {
        "domain": domain,
        "rewriter": rewriter,
        "queries": len(stats),
        "missing": missing,
        "mrr": {
            "nowrite": mean("nowrite"),
            "entr_top1": mean("entr_top1"),
            "entr_best3": mean("entr_best3"),
            "oracle": mean("oracle"),
        },
        "delta": {
            "entr_top1": mean("entr_top1") - mean("nowrite"),
            "entr_best3": mean("entr_best3") - mean("nowrite"),
            "oracle": mean("oracle") - mean("nowrite"),
        },
        "spearman_entropy": mean("spearman_entropy"),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("source", "cosqa"), required=True)
    parser.add_argument("--rewriter", choices=("local", "t5base"), required=True)
    args = parser.parse_args()

    protocol = json.load(open(os.path.join(BRIDGE, "PROTOCOL.json"), encoding="utf-8"))
    ids = protocol["samples"][args.domain]["sensitivity_ids"]
    paths = domain_paths(args.domain)
    queries = {row["query_id"]: row["query"] for row in load_jsonl(paths["queries"])}
    qrels = {row["query_id"]: int(row["doc_id"]) for row in load_jsonl(paths["qrels"])}

    if args.rewriter == "local":
        rows = [row for row in load_jsonl(paths["candidates"]) if row["query_id"] in set(ids)]
        rankings = {row["text_id"]: row["ranked_doc_ids"] for row in load_jsonl(paths["rankings"])}
    else:
        candidate_path = os.path.join(OUT, "candidates", args.domain, f"{args.rewriter}.jsonl")
        if os.path.exists(candidate_path):
            rows = load_jsonl(candidate_path)
            print({"stage": "generate", "cached": candidate_path, "rows": len(rows)}, flush=True)
        else:
            rows = generate_t5base(args.domain, ids, queries)
            os.makedirs(os.path.dirname(candidate_path), exist_ok=True)
            with open(candidate_path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        rankings = rank_texts(args.domain, ids, queries, rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    result = ladder(args.domain, args.rewriter, ids, queries, qrels, rankings, grouped)
    result["sensitivity_queries"] = len(ids)
    result["candidate_rows"] = len(rows)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{args.domain}_{args.rewriter}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
