"""Positive control + signal-level analysis on the frozen SSQR bridge candidates.

No new generation: joins frozen candidates to frozen rankings and qrels, then
computes per-query RR for every position, rank correlations, and the
NoRewrite / Top-1 / best3 / oracle ladder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict

ROOT = r"."
BRIDGE = os.path.join(ROOT, r"results\rene_ssqr_bridge_20261019")
DEV = os.path.join(ROOT, r"results\rene_cosqa_development_20261019")
OUT = os.path.join(ROOT, r"results\rene_ssqr_rewriter_envelope_20261020")
SENS = 512


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


def point_biserial(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def domain_rows(domain, ids):
    protocol = json.load(open(os.path.join(BRIDGE, "PROTOCOL.json"), encoding="utf-8"))
    allowed = set(ids)
    if domain == "source":
        candidates = load_jsonl(os.path.join(BRIDGE, "candidates/source/baseline.jsonl"))
        rankings_path = os.path.join(BRIDGE, "rankings/source/CodeBERT.jsonl")
        qrels_path = os.path.join(BRIDGE, "data/private/source_qrels.jsonl")
        query_path = os.path.join(BRIDGE, "data/source/queries.jsonl")
    else:
        candidates = load_jsonl(os.path.join(DEV, "candidates.jsonl"))
        rankings_path = os.path.join(BRIDGE, "rankings/cosqa/CodeBERT.jsonl")
        qrels_path = os.path.join(BRIDGE, "data/private/cosqa_qrels.jsonl")
        query_path = os.path.join(DEV, "data/queries.jsonl")

    qrels = {row["query_id"]: int(row["doc_id"]) for row in load_jsonl(qrels_path)}
    queries = {row["query_id"]: row["query"] for row in load_jsonl(query_path)}
    rankings = {}
    for row in load_jsonl(rankings_path):
        rankings[row["text_id"]] = row["ranked_doc_ids"]
    grouped = defaultdict(list)
    for row in candidates:
        if row["query_id"] in allowed:
            grouped[row["query_id"]].append(row)
    return queries, qrels, rankings, grouped


def analyze(domain, ids, label):
    queries, qrels, rankings, grouped = domain_rows(domain, ids)
    stats = []
    missing = 0
    for query_id in ids:
        rows = grouped.get(query_id)
        if not rows or query_id not in qrels or query_id not in queries:
            missing += 1
            continue
        target = qrels[query_id]
        nowrite = rr(rankings.get(text_id(queries[query_id]), []), target)
        scored = []
        for row in rows:
            ranked = rankings.get(text_id(row["rewrite"]))
            if ranked is None:
                continue
            scored.append({
                "entropy": float(row["full_vocab_entropy"]),
                "prob": float(row["label_mean_log_probability"]),
                "rr": rr(ranked, target),
            })
        if len(scored) < 3:
            missing += 1
            continue
        entropies = [s["entropy"] for s in scored]
        probs = [s["prob"] for s in scored]
        rrs = [s["rr"] for s in scored]
        by_entropy = sorted(scored, key=lambda s: s["entropy"])
        by_prob = sorted(scored, key=lambda s: -s["prob"])
        rand = random.Random(20261020 + len(stats))
        shuffled = entropies[:]
        rand.shuffle(shuffled)
        stats.append({
            "query_id": query_id,
            "positions": len(scored),
            "nowrite": nowrite,
            "oracle": max(rrs),
            "entr_top1": by_entropy[0]["rr"],
            "entr_best3": max(s["rr"] for s in by_entropy[:3]),
            "prob_top1": by_prob[0]["rr"],
            "prob_best3": max(s["rr"] for s in by_prob[:3]),
            "spearman_entropy": spearman(entropies, rrs),
            "spearman_prob": spearman(probs, rrs),
            "spearman_random": spearman(shuffled, rrs),
            "pb_entropy": point_biserial(entropies, [1.0 if r > 0 else 0.0 for r in rrs]),
        })

    def mean(key):
        values = [s[key] for s in stats if s[key] is not None]
        return sum(values) / len(values) if values else None

    def frac_negative(key):
        values = [s[key] for s in stats if s[key] is not None]
        return sum(1 for value in values if value < 0) / len(values) if values else None

    result = {
        "domain": domain,
        "label": label,
        "queries": len(stats),
        "missing": missing,
        "positions_mean": mean("positions"),
        "mrr": {
            "nowrite": mean("nowrite"),
            "entr_top1": mean("entr_top1"),
            "entr_best3": mean("entr_best3"),
            "prob_top1": mean("prob_top1"),
            "prob_best3": mean("prob_best3"),
            "oracle": mean("oracle"),
        },
        "delta": {
            "entr_top1": mean("entr_top1") - mean("nowrite"),
            "entr_best3": mean("entr_best3") - mean("nowrite"),
            "prob_top1": mean("prob_top1") - mean("nowrite"),
            "prob_best3": mean("prob_best3") - mean("nowrite"),
            "oracle": mean("oracle") - mean("nowrite"),
        },
        "spearman": {
            "entropy": mean("spearman_entropy"),
            "prob": mean("spearman_prob"),
            "random": mean("spearman_random"),
            "entropy_negative_fraction": frac_negative("spearman_entropy"),
            "prob_negative_fraction": frac_negative("spearman_prob"),
        },
        "point_biserial_entropy": mean("pb_entropy"),
    }
    return result


def main():
    os.makedirs(OUT, exist_ok=True)
    protocol = json.load(open(os.path.join(BRIDGE, "PROTOCOL.json"), encoding="utf-8"))
    output = {"schema": "rene-signal-and-positive-control-20261020-v1", "role": "post_hoc_on_frozen_bridge"}
    for domain in ("source", "cosqa"):
        sens = protocol["samples"][domain]["sensitivity_ids"]
        main_ids = protocol["samples"][domain]["main_ids"]
        output[f"{domain}_sensitivity"] = analyze(domain, sens, f"{domain}/sensitivity512")
        output[f"{domain}_main"] = analyze(domain, main_ids, f"{domain}/main")
    with open(os.path.join(OUT, "SIGNAL_POSITIVE_CONTROL.json"), "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
