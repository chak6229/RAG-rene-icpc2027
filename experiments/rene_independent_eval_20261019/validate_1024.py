"""Independently recompute the sealed 1024-query risk-position confirmation.

Reads only the sealed artifacts (private outcomes, retrieval items, decisions)
and recomputes RR/MRR, adoption, harmful counts, and a risk-coverage sweep.
Does not import the pipeline.
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

import rne_eval

BASE = r".\results\rene_uq_independent_proof_20261019\fresh"
OUT_DIR = r".\results\rene_independent_eval_20261019"

EXPECTED = {
    "no_seed_uq/common/threshold075": {"delta_mrr": 0.0023242152270472583, "adopted": 102, "harmful": 3},
    "no_seed_uq/common/fixed50": {"delta_mrr": 0.005158934958642637, "adopted": 256, "harmful": 19},
    "no_seed_uq/full/threshold075": {"delta_mrr": 0.0022445094932083296, "adopted": 115, "harmful": 7},
    "seed_uq/common/threshold075": {"delta_mrr": 0.005109974683139359, "adopted": 89, "harmful": 6},
    "seed_uq/full/threshold075": {"delta_mrr": 0.005415193208605657, "adopted": 125, "harmful": 9},
}


def load_artifacts():
    with open(os.path.join(BASE, "private_outcomes.json"), encoding="utf-8") as handle:
        outcomes = json.load(handle)
    with open(os.path.join(BASE, "retrieval_items.pkl"), "rb") as handle:
        retrieval = pickle.load(handle)
    with open(os.path.join(BASE, "label_free_decisions.pkl"), "rb") as handle:
        decisions = pickle.load(handle)

    target = {row["query_index"]: row["target_doc_id"] for row in outcomes}
    groups = {row["query_index"]: row.get("repo") or str(row["query_index"]) for row in outcomes}
    original = {row["query_index"]: list(row["original_top100"]) for row in retrieval}
    return decisions, target, groups, original


def rr(ranking, target_doc, k=100):
    for position, doc_id in enumerate(ranking[:k], start=1):
        if int(doc_id) == int(target_doc):
            return 1.0 / position
    return 0.0


def recompute_method(rows, target, original, k=100):
    deltas = {}
    adopted = 0
    positive = 0
    harmful = 0
    for row in rows:
        query_id = row["query_index"]
        delta = rr(row["ranking"], target[query_id], k) - rr(original[query_id], target[query_id], k)
        deltas[query_id] = delta
        if row.get("accepted"):
            adopted += 1
        if delta > 1e-10:
            positive += 1
        elif delta < -1e-10:
            harmful += 1
    mean = sum(deltas.values()) / len(deltas)
    return {
        "queries": len(deltas),
        "delta_mrr": mean,
        "adopted": adopted,
        "positive": positive,
        "harmful": harmful,
        "deltas": deltas,
    }


def risk_coverage_sweep(fixed_rows, post_harm_rows, target, original, groups, k=100):
    fixed = {row["query_index"]: list(row["ranking"]) for row in fixed_rows}
    post_harm = {}
    missing_post_harm = 0
    for row in post_harm_rows:
        value = row.get("post_harm")
        if value is None:
            # No pre-gate action was available/accepted; treat as always rejected.
            post_harm[row["query_index"]] = float("inf")
            missing_post_harm += 1
        else:
            post_harm[row["query_index"]] = float(value)
    query_ids = sorted(fixed)
    thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.09, 0.10, 0.125, 0.15, 0.20, 0.30, 0.50]
    points = []
    for tau in thresholds:
        deltas = {}
        adopted = 0
        harmful = 0
        positive = 0
        positive_gain = 0.0
        negative_loss = 0.0
        for query_id in query_ids:
            if post_harm[query_id] <= tau:
                ranking = fixed[query_id]
                adopted += 1
            else:
                ranking = original[query_id]
            delta = rr(ranking, target[query_id], k) - rr(original[query_id], target[query_id], k)
            deltas[query_id] = delta
            if delta > 1e-10:
                positive += 1
                positive_gain += delta
            elif delta < -1e-10:
                harmful += 1
                negative_loss += -delta
        mean = sum(deltas.values()) / len(deltas)
        lo, hi = rne_eval.cluster_bootstrap_ci(deltas, groups, n_boot=2000, alpha=0.05) if adopted else (0.0, 0.0)
        points.append({
            "tau": tau,
            "delta_mrr": mean,
            "adopted": adopted,
            "coverage": adopted / len(query_ids),
            "positive": positive,
            "harmful": harmful,
            "harmful_rate": harmful / adopted if adopted else 0.0,
            "positive_gain": positive_gain,
            "negative_loss": negative_loss,
            "ci95": [lo, hi],
        })
    return points, missing_post_harm


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    decisions, target, groups, original = load_artifacts()
    outputs = decisions["outputs"]

    recomputed = {}
    for method, rows in outputs.items():
        recomputed[method] = {key: value for key, value in recompute_method(rows, target, original).items() if key != "deltas"}

    comparisons = {}
    for method, expected in EXPECTED.items():
        actual = recomputed[method]
        comparisons[method] = {
            "expected": expected,
            "actual": {"delta_mrr": actual["delta_mrr"], "adopted": actual["adopted"], "harmful": actual["harmful"]},
            "delta_mrr_abs_error": abs(actual["delta_mrr"] - expected["delta_mrr"]),
            "adopted_match": actual["adopted"] == expected["adopted"],
            "harmful_match": actual["harmful"] == expected["harmful"],
        }

    fixed_rows = decisions["fixed"]["no_seed_uq"]
    post_harm_rows = outputs["no_seed_uq/common/threshold075"]
    sweep, missing_post_harm = risk_coverage_sweep(fixed_rows, post_harm_rows, target, original, groups)

    payload = {
        "schema": "rene-independent-eval-1024-20261019-v1",
        "source": BASE,
        "queries": len(target),
        "groups": len(set(groups.values())),
        "recomputed": recomputed,
        "comparisons": comparisons,
        "risk_coverage_no_seed_common": sweep,
        "missing_post_harm_no_seed_common": missing_post_harm,
        "note": "Independent recomputation from sealed rankings + private outcomes; no pipeline imports.",
    }
    with open(os.path.join(OUT_DIR, "RECOMPUTE_1024.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(comparisons, ensure_ascii=False, indent=2, sort_keys=True))
    print("\nrisk-coverage (no_seed/common):")
    for point in sweep:
        print(f"tau={point['tau']:.3f} adopted={point['adopted']:4d} harm={point['harmful']:3d} "
              f"harm_rate={point['harmful_rate']:.4f} delta={point['delta_mrr']:+.8f}")


if __name__ == "__main__":
    main()
