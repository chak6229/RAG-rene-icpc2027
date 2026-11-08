"""Independent evaluation utilities for RENE replication / adoption analysis.

No imports from the training or scoring pipeline: this module only consumes
qrels + rankings (or per-query decisions) and recomputes metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels(path: str) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in load_jsonl(path):
        qrels[str(row["query_id"])][str(row["doc_id"])] = float(row.get("relevance", 1.0))
    return dict(qrels)


def load_rankings(path: str) -> Dict[str, List[str]]:
    rankings: Dict[str, List[str]] = {}
    for row in load_jsonl(path):
        query_id = str(row["query_id"])
        if query_id in rankings:
            raise ValueError(f"duplicate ranking query: {query_id}")
        docs = [str(doc) for doc in row["doc_ids"]]
        if len(docs) != len(set(docs)):
            raise ValueError(f"duplicate ranked document: {query_id}")
        rankings[query_id] = docs
    return rankings


def rr_at_k(ranked: Sequence[str], relevant: Dict[str, float], k: int = 100) -> float:
    for position, doc_id in enumerate(ranked[:k], start=1):
        if relevant.get(str(doc_id), 0.0) > 0.0:
            return 1.0 / position
    return 0.0


def per_query_rr(rankings: Dict[str, List[str]], qrels: Dict[str, Dict[str, float]], k: int = 100) -> Dict[str, float]:
    return {query_id: rr_at_k(rankings.get(query_id, []), qrels[query_id], k) for query_id in qrels}


def mrr(rankings: Dict[str, List[str]], qrels: Dict[str, Dict[str, float]], k: int = 100) -> float:
    rrs = per_query_rr(rankings, qrels, k)
    return sum(rrs.values()) / len(rrs) if rrs else float("nan")


def paired_delta(
    rankings_a: Dict[str, List[str]],
    rankings_b: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, float]],
    k: int = 100,
) -> Dict[str, float]:
    rr_a = per_query_rr(rankings_a, qrels, k)
    rr_b = per_query_rr(rankings_b, qrels, k)
    return {query_id: rr_a[query_id] - rr_b[query_id] for query_id in qrels}


def cluster_bootstrap_ci(
    values: Dict[str, float],
    groups: Dict[str, str],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 20261019,
) -> Tuple[float, float]:
    """Percentile CI resampling whole groups (clusters)."""
    by_group: Dict[str, List[float]] = defaultdict(list)
    for query_id, value in values.items():
        by_group[groups[query_id]].append(value)
    cluster_ids = list(by_group)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sampled = [cluster_ids[rng.randrange(len(cluster_ids))] for _ in cluster_ids]
        total = 0.0
        count = 0
        for cluster_id in sampled:
            bucket = by_group[cluster_id]
            total += sum(bucket)
            count += len(bucket)
        means.append(total / count)
    means.sort()
    lo = means[int(math.floor((alpha / 2) * n_boot))]
    hi = means[int(math.ceil((1 - alpha / 2) * n_boot)) - 1]
    return lo, hi


def tost(
    values: Dict[str, float],
    groups: Dict[str, str],
    margin: float,
    n_boot: int = 10000,
    seed: int = 20261019,
) -> dict:
    mean = sum(values.values()) / len(values)
    lo90, hi90 = cluster_bootstrap_ci(values, groups, n_boot=n_boot, alpha=0.10, seed=seed)
    return {
        "mean": mean,
        "ci90": [lo90, hi90],
        "margin": margin,
        "equivalent": lo90 > -margin and hi90 < margin,
    }


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for rank, (name, pvalue) in enumerate(items, start=1):
        running = max(running, min(1.0, (m - rank + 1) * pvalue))
        adjusted[name] = running
    return adjusted


def summarize(
    rankings: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, float]],
    k: int = 100,
) -> dict:
    rr = per_query_rr(rankings, qrels, k)
    values = list(rr.values())
    return {
        "queries": len(values),
        "mrr": sum(values) / len(values),
        "nonzero": sum(1 for value in values if value > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--rankings-a", required=True)
    parser.add_argument("--rankings-b", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    rankings_a = load_rankings(args.rankings_a)
    rankings_b = load_rankings(args.rankings_b)
    with open(args.groups, encoding="utf-8") as handle:
        groups = json.load(handle)

    missing_a = sorted(set(qrels) - set(rankings_a))
    missing_b = sorted(set(qrels) - set(rankings_b))
    deltas = paired_delta(rankings_a, rankings_b, qrels, args.k)
    summary_a = summarize(rankings_a, qrels, args.k)
    summary_b = summarize(rankings_b, qrels, args.k)
    ci95 = cluster_bootstrap_ci(deltas, groups, n_boot=args.bootstrap, alpha=0.05)
    equivalence = tost(deltas, groups, args.margin, n_boot=args.bootstrap)

    report = {
        "k": args.k,
        "queries": len(qrels),
        "groups": len(set(groups.values())),
        "missing_rankings_a": len(missing_a),
        "missing_rankings_b": len(missing_b),
        "a": summary_a,
        "b": summary_b,
        "delta_mrr": summary_a["mrr"] - summary_b["mrr"],
        "delta_ci95": list(ci95),
        "tost": equivalence,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
