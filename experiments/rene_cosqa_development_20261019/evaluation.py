"""RQ1 and same-pool automatic decisions; denominators never depend on success."""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from experiments.rene_cosqa_pilot_20261019.evaluation import (
    per_query_results, summarize, clustered_stats, rr,
)
from .common import (ADAPTER, OUT, now, read, readl, require, sha,
                     verify_decisions, verify_frozen, write, writel)


def subgroup_protocol(protocol, ids):
    result = dict(protocol)
    result["query_ids"] = list(ids)
    result["queries"] = len(ids)
    return result


def subsets(protocol):
    return {"all_development": protocol["query_ids"], "original_pilot": protocol["pilot_query_ids"],
            "remaining_development": protocol["new_query_ids"]}


def positive_qrels(qrels, expected):
    positives = defaultdict(set)
    for row in qrels:
        require(row["query_id"] in expected, "qrels outside frozen D")
        if row.get("relevance", 1) > 0:
            positives[row["query_id"]].add(row["doc_id"])
    require(set(positives) == set(expected), "missing positive qrels")
    return positives


def open_rq1(retriever):
    protocol = verify_frozen(private=True)
    verify_decisions()
    require(read(OUT / "PREOUTCOME_AUDIT.json")["passed"], "preoutcome audit required")
    marker = OUT / f"metrics/{retriever}_OPENED.json"
    require(not marker.exists(), "development effects already opened; use stored results")
    write(marker, {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"), "confirmation_opened": False,
                   "purpose": "remaining-D RQ1 plus full-D summary; original pilot already consumed"}, immutable=True)
    rows = per_query_results(readl(OUT / "data/queries.jsonl"), readl(ADAPTER / "public/corpus.jsonl"),
                             readl(OUT / "candidates.jsonl"), readl(OUT / f"rankings/{retriever}.jsonl"),
                             readl(OUT / "selections.jsonl"), read(OUT / "data/groups.json"), protocol,
                             readl(OUT / "data/private/development_qrels.jsonl"))
    writel(OUT / f"metrics/{retriever}_per_query.jsonl", rows)
    metrics = {}
    for subset, ids in subsets(protocol).items():
        expected = set(ids)
        result = summarize([r for r in rows if r["query_id"] in expected], subgroup_protocol(protocol, ids))
        result["status"] = "development_extension_descriptive"
        metrics[subset] = result
    write(OUT / f"metrics/{retriever}_RQ1.json", {"created_utc": now(), "retriever": retriever,
                                               "protocol_sha256": sha(OUT / "PROTOCOL.json"), "subsets": metrics,
                                               "confirmation_opened": False}, immutable=True)


def rq2_rows(decisions, rankings, positives, protocol, groups, selections=None):
    actions = {(r["query_id"], r["position"]): r for r in rankings}
    policies = {"NoRewrite", "ENTR_top1", "PROB_in_ENTR3"} | {
        name + suffix for name in protocol["direct_gain"]["models"] for suffix in ("_forced", "_selective")}
    expected = {(q, policy) for q in protocol["query_ids"] for policy in policies}
    seen, by_query = set(), defaultdict(dict)
    if selections is None:
        selections = readl(OUT / "selections.jsonl")
    selected = {r["query_id"]: [v["position"] for v in sorted(r["selected"], key=lambda x: x["selected_rank"])]
                for r in selections if r["policy"] == "ENTR"}
    for row in decisions:
        key = row["query_id"], row["policy"]
        require(key in expected and key not in seen, "duplicate/unknown decision query or policy")
        seen.add(key)
        qid, policy = key
        position = row["selected_position"]
        require(position is None or position in selected[qid], "decision escaped the common ENTR3 pool")
        original = actions[qid, None]["ranked_doc_ids"]
        wanted = actions[qid, position]["ranked_doc_ids"]
        adopted = wanted != original
        require(row["adopted"] == adopted and row["ranked_doc_ids"] == wanted, "decision action/ranking inconsistency")
        require(row["action"] == ("adopt" if adopted else "fallback"), "wrong action semantics")
        require(adopted or row["ranked_doc_ids"] == original, "fallback failed to restore original ranking")
        base = rr(original, positives[qid], 100)
        value = rr(wanted, positives[qid], 100)
        by_query[qid][policy] = {"rr": value, "delta": value - base, "adopted": adopted,
                                  "harmful": adopted and value - base < -1e-10,
                                  "selected_position": position, "predicted_gain": row["predicted_gain"]}
    require(seen == expected, "missing expected decision; changing denominator is forbidden")
    rows = []
    for qid in protocol["query_ids"]:
        original_rr = rr(actions[qid, None]["ranked_doc_ids"], positives[qid], 100)
        pool_rr = [rr(actions[qid, pos]["ranked_doc_ids"], positives[qid], 100) for pos in selected[qid]]
        rows.append({"query_id": qid, "group_id": groups[qid], "NoRewrite": original_rr,
                     "oracle_ENTR3_rr": max(pool_rr), "oracle_with_NoRewrite_rr": max([original_rr] + pool_rr),
                     "policies": by_query[qid]})
    return rows


def summarize_rq2(rows, protocol):
    require(len(rows) == len(protocol["query_ids"]), "RQ2 denominator mismatch")
    groups = [r["group_id"] for r in rows]
    kwargs = {"bootstrap": protocol["bootstrap_repetitions"], "seed": protocol["statistics_seed"], "margin": .005}
    base = np.asarray([r["NoRewrite"] for r in rows])
    candidate = np.asarray([r["oracle_ENTR3_rr"] for r in rows])
    oracle = np.asarray([r["oracle_with_NoRewrite_rr"] for r in rows])
    summary = {"queries": len(rows), "groups": len(set(groups)), "NoRewrite_mrr": float(base.mean()),
               "oracle_ENTR3_delta": float(np.mean(candidate - base)),
               "oracle_with_NoRewrite_delta": float(np.mean(oracle - base)), "policies": {}, "contrasts": {},
               "inference": "descriptive development OOF, conditional on fitted fold models"}
    policies = sorted(rows[0]["policies"])
    for policy in policies:
        values = [r["policies"][policy] for r in rows]
        gains = np.asarray([r["delta"] for r in values])
        adopted = sum(r["adopted"] for r in values)
        harmful = sum(r["harmful"] for r in values)
        summary["policies"][policy] = {"mrr": float(base.mean() + gains.mean()),
                                        "delta": clustered_stats(gains, groups, **kwargs),
                                        "adopted": adopted, "coverage": adopted / len(rows), "harmful": harmful,
                                        "harmful_within_adopted": harmful / adopted if adopted else None,
                                        "positive_queries": int((gains > 1e-10).sum()),
                                        "positive_gain_sum": float(gains[gains > 0].sum()),
                                        "damage_sum": float(-gains[gains < 0].sum()),
                                        "net_oracle_realization": float(gains.mean() / np.mean(oracle - base)) if np.mean(oracle - base) > 1e-10 else None}
    contrasts = {
        "GAIN_HGB_forced_minus_ENTR_top1": ("GAIN_HGB_forced", "ENTR_top1"),
        "GAIN_HGB_selective_minus_NoRewrite": ("GAIN_HGB_selective", "NoRewrite"),
        "GAIN_HGB_forced_minus_GEN_HGB_forced": ("GAIN_HGB_forced", "GEN_HGB_forced"),
        "GAIN_HGB_selective_minus_GEN_HGB_selective": ("GAIN_HGB_selective", "GEN_HGB_selective"),
        "GAIN_RIDGE_selective_minus_NoRewrite": ("GAIN_RIDGE_selective", "NoRewrite"),
        "GAIN_HGB_selective_minus_GAIN_HGB_forced": ("GAIN_HGB_selective", "GAIN_HGB_forced"),
    }
    for name, (method, control) in contrasts.items():
        values = [r["policies"][method]["rr"] - r["policies"][control]["rr"] for r in rows]
        summary["contrasts"][name] = clustered_stats(values, groups, **kwargs)
    return summary


def open_rq2(retriever):
    protocol = verify_frozen(private=True)
    seal = read(OUT / f"OOF_SEALED_{retriever}.json")
    for row in seal["files"]:
        require(sha(OUT / row["path"]) == row["sha256"], "OOF artifact changed")
    positives = positive_qrels(readl(OUT / "data/private/development_qrels.jsonl"), protocol["query_ids"])
    rankings = readl(OUT / f"rankings/{retriever}.jsonl")
    rows = rq2_rows(readl(OUT / f"decisions/{retriever}.jsonl"), rankings, positives, protocol, read(OUT / "data/groups.json"))
    writel(OUT / f"metrics/{retriever}_RQ2_per_query.jsonl", rows)
    summaries = {}
    for subset, ids in subsets(protocol).items():
        ids_set = set(ids)
        summaries[subset] = summarize_rq2([r for r in rows if r["query_id"] in ids_set], subgroup_protocol(protocol, ids))
    predictions = readl(OUT / f"oof_predictions/{retriever}.jsonl")
    actions = {(r["query_id"], r["position"]): r for r in rankings}
    labels = np.array([rr(actions[r["query_id"], r["position"]]["ranked_doc_ids"], positives[r["query_id"]], 100)
                       - rr(actions[r["query_id"], None]["ranked_doc_ids"], positives[r["query_id"]], 100) for r in predictions])
    prediction_quality = {}
    for model in protocol["direct_gain"]["models"]:
        p = np.array([r["predictions"][model]["effective"] for r in predictions])
        correlation = float(spearmanr(p, labels).statistic) if np.ptp(p) and np.ptp(labels) else None
        prediction_quality[model] = {"candidate_rows": len(labels), "mae": float(np.mean(np.abs(p - labels))),
                                     "rmse": float(np.sqrt(np.mean((p - labels) ** 2))), "spearman": correlation,
                                     "mean_prediction": float(p.mean()), "mean_label": float(labels.mean())}
    write(OUT / f"metrics/{retriever}_RQ2.json", {"created_utc": now(), "retriever": retriever, "subsets": summaries,
                                               "candidate_prediction_quality": prediction_quality, "confirmation_opened": False,
                                               "protocol_sha256": sha(OUT / "PROTOCOL.json")}, immutable=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("rq1", "rq2"))
    parser.add_argument("retriever", choices=("CodeBERT", "BM25"))
    args = parser.parse_args()
    (open_rq1 if args.mode == "rq1" else open_rq2)(args.retriever)
