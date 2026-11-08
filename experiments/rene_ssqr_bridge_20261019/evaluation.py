"""Outcome-blind position selection and explicit best3 evaluation contracts."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import t

from experiments.rene_cosqa_pilot_20261019.selection import select
from .common import require


def selections(rows, query_id, seeds, *, mask="full"):
    require(rows and len({r["position"] for r in rows}) == len(rows), "incomplete/duplicate candidate pool")
    require(all(r["query_id"] == query_id for r in rows), "mixed query pool")
    if mask == "full":
        entr = select(rows, query_id, "ENTR", 3)
    elif mask == "lexical":
        # Empty spans have no lexical-token entropy, and are explicitly placed last.
        def key(row):
            value = row["span_full_vocab_entropy"]
            return (value is None, math.inf if value is None else value, row["position"])
        entr = [r["position"] for r in sorted(rows, key=key)[:min(3, len(rows))]]
    else:
        raise ValueError("unknown entropy mask")
    return {"ENTR": entr, "PROB": select(rows, query_id, "PROB", 3),
            "RAND": [select(rows, query_id, "RAND", 3, seed) for seed in seeds]}


def rr(ranked_doc_ids, relevant, *, depth=100):
    require(len(ranked_doc_ids) == len(set(ranked_doc_ids)), "duplicate document in ranking")
    require(len(ranked_doc_ids) >= depth, "incomplete fixed-depth ranking")
    for rank, doc_id in enumerate(ranked_doc_ids[:depth], 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_query(pool, selected, ranking_lookup, relevant, original_text, text_id):
    positions = {r["position"]: r for r in pool}
    base = rr(ranking_lookup[text_id(original_text)]["ranked_doc_ids"], relevant)
    candidate_rr = {p: rr(ranking_lookup[text_id(row["rewrite"])]["ranked_doc_ids"], relevant)
                    for p, row in positions.items()}
    for policy in ("ENTR", "PROB"):
        require(len(selected[policy]) == min(3, len(pool)) and
                len(set(selected[policy])) == len(selected[policy]) and
                set(selected[policy]) <= set(positions), "selection budget or position mismatch")
    require(len(selected["RAND"]) == 20 and all(len(x) == min(3, len(pool)) and len(set(x)) == len(x)
            and set(x) <= set(positions) for x in selected["RAND"]), "RAND repetitions/budget mismatch")
    result = {"NoRewrite": base,
              "ENTR_top1": candidate_rr[selected["ENTR"][0]],
              "ENTR_best3": max(candidate_rr[p] for p in selected["ENTR"]),
              "PROB_best3": max(candidate_rr[p] for p in selected["PROB"]),
              "RAND20_best3": float(np.mean([max(candidate_rr[p] for p in positions_) for positions_ in selected["RAND"]]))}
    result["ENTR_minus_RAND"] = result["ENTR_best3"] - result["RAND20_best3"]
    result["ENTR_minus_PROB"] = result["ENTR_best3"] - result["PROB_best3"]
    return result


def cluster_interval(values, groups, *, repetitions=10000, seed=261019):
    values = np.asarray(values, dtype=np.float64)
    labels = sorted(set(groups))
    require(len(values) == len(groups) and len(labels) > 1 and np.isfinite(values).all(), "invalid clustered metric")
    label_index = {g: i for i, g in enumerate(labels)}
    codes = np.array([label_index[g] for g in groups])
    sums = np.bincount(codes, weights=values, minlength=len(labels))
    sizes = np.bincount(codes, minlength=len(labels)).astype(float)
    mean = float(values.mean())
    se = math.sqrt(len(labels) / (len(labels) - 1) * float(np.square(sums - mean * sizes).sum()) / len(values)**2)
    half = float(t.isf(.025, len(labels) - 1)) * se
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, repetitions, 200):
        indices = rng.integers(0, len(labels), (min(200, repetitions - start), len(labels)))
        boot.extend((sums[indices].sum(1) / sizes[indices].sum(1)).tolist())
    return {"mean": mean, "queries": len(values), "groups": len(labels),
            "changed_groups": int(np.count_nonzero(np.abs(sums) > 1e-12)),
            "standard_error_cluster": se, "ci95_cluster_t": [mean-half, mean+half],
            "ci95_group_bootstrap": np.quantile(boot, [.025, .975]).tolist(),
            "interpretation": "descriptive post-confirmation diagnostic"}


def domain_difference(source_values, source_groups, cosqa_values, cosqa_groups, *, repetitions=10000, seed=261020):
    """Difference of query-weighted means with shared repository-cluster covariance."""
    source_values, cosqa_values = np.asarray(source_values, float), np.asarray(cosqa_values, float)
    require(len(source_values) == len(source_groups) and len(cosqa_values) == len(cosqa_groups), "domain group mismatch")
    labels = sorted(set(source_groups) | set(cosqa_groups))
    require(len(labels) > 1, "independent groups required")
    label_index = {g: i for i, g in enumerate(labels)}
    def aggregate(values, groups):
        indices = [label_index[g] for g in groups]
        return (np.bincount(indices, weights=values, minlength=len(labels)),
                np.bincount(indices, minlength=len(labels)).astype(float))
    a, na = aggregate(source_values, source_groups)
    b, nb = aggregate(cosqa_values, cosqa_groups)
    ma, mb = float(source_values.mean()), float(cosqa_values.mean())
    influence = (a - ma*na)/na.sum() - (b - mb*nb)/nb.sum()
    se = math.sqrt(len(labels)/(len(labels)-1) * float(np.square(influence).sum()))
    mean = ma - mb
    half = float(t.isf(.025, len(labels)-1)) * se
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, repetitions, 200):
        indices = rng.integers(0, len(labels), (min(200, repetitions-start), len(labels)))
        aa, bb = na[indices].sum(1), nb[indices].sum(1)
        valid = (aa > 0) & (bb > 0)
        boot.extend((a[indices].sum(1)[valid]/aa[valid] - b[indices].sum(1)[valid]/bb[valid]).tolist())
    return {"mean": mean, "source_mean": ma, "cosqa_mean": mb,
            "joint_groups": len(labels), "shared_groups": len(set(source_groups) & set(cosqa_groups)),
            "standard_error_cluster": se, "ci95_cluster_t": [mean-half, mean+half],
            "ci95_group_bootstrap": np.quantile(boot, [.025,.975]).tolist(),
            "bootstrap_valid_draws": len(boot), "query_only_causal_claim": False}
