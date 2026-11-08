"""Strict independent RR recomputation from sealed action rankings and selections.

No generation, feature fitting, or retrieval-model imports. Query IDs are supplied
by a frozen role manifest; missing data is an error, never a changed denominator.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

from .common import OUT, read, readl, sha, verify_frozen, write, writel
from .selection import select


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inputs(queries, corpus, candidates, rankings, selections, groups, protocol):
    ids = [row["query_id"] for row in queries]
    expected = protocol["query_ids"]
    require(len(ids) == len(set(ids)) and set(ids) == set(expected) and len(ids) == len(expected),
            "queries differ from frozen expected query IDs")
    require(set(groups) == set(expected) and all(isinstance(value, str) and value for value in groups.values()),
            "one connected group is required for every expected query")
    doc_ids = [row["doc_id"] for row in corpus]
    require(len(doc_ids) == len(set(doc_ids)) and all(type(value) is int for value in doc_ids), "duplicate/invalid corpus IDs")
    documents = set(doc_ids)
    query_text = {row["query_id"]: row["query"] for row in queries}
    pools = {query: {} for query in expected}
    for row in candidates:
        query, position = row["query_id"], row["position"]
        require(query in pools and type(position) is int, "unknown candidate query/invalid position")
        require(position not in pools[query], "duplicate candidate position")
        require(all(math.isfinite(float(row[key])) for key in ("full_vocab_entropy", "label_mean_log_probability")),
                "nonfinite candidate score")
        pools[query][position] = row
    for query, pool in pools.items():
        words = query_text[query].split()
        positions = set(range(1, len(words) + 1)) if protocol["generation"]["position_policy"] == "source_1_to_n" else set(range(len(words) + 1))
        require(set(pool) == positions and bool(pool), "candidate pool does not match frozen position domain")

    actions = {}
    expected_keys = {(query, None) for query in expected} | {(query, position) for query, pool in pools.items() for position in pool}
    rank_depth = min(protocol["ranking_depth"], len(documents))
    for row in rankings:
        require(row["position"] is None or type(row["position"]) is int, "invalid action position type")
        key = row["query_id"], row["position"]
        require(key in expected_keys and key not in actions, "duplicate/unknown action or query")
        ranking, scores = row["ranked_doc_ids"], row["scores"]
        require(len(ranking) == rank_depth and len(scores) == rank_depth, "missing or truncated action ranking")
        require(all(type(doc) is int for doc in ranking) and set(ranking) <= documents, "ranked doc outside corpus")
        require(len(set(ranking)) == len(ranking), "duplicate doc in ranked list")
        require(all(math.isfinite(float(score)) for score in scores), "nonfinite ranking score")
        require(all(scores[i] > scores[i + 1] or (scores[i] == scores[i + 1] and ranking[i] < ranking[i + 1])
                    for i in range(len(scores) - 1)), "ranking violates score/tie order")
        wanted_text = query_text[key[0]] if key[1] is None else pools[key[0]][key[1]]["rewrite"]
        require(row["text"] == wanted_text, "action text does not match public query/candidate")
        require(row["policy"] == ("NoRewrite" if key[1] is None else "Candidate"), "invalid action policy")
        actions[key] = row
    require(set(actions) == expected_keys, "missing expected query or action (including NoRewrite)")

    selected = {}
    expected_selection = {(query, policy, seed) for query in expected
                          for policy, seeds in (("ENTR", [None]), ("PROB", [None]), ("RAND", protocol["rand_seeds"]))
                          for seed in seeds}
    for row in selections:
        key = row["query_id"], row["policy"], row["repeat_seed"]
        require(key in expected_selection and key not in selected, "duplicate/unknown policy selection")
        pool = pools[key[0]]
        entries = row["selected"]
        require(len(entries) == min(protocol["candidate_k"], len(pool)), "selection is not min(k, pool size)")
        require(all(type(entry["selected_rank"]) is int for entry in entries), "selected_rank must be explicit integer")
        ordered = sorted(entries, key=lambda entry: entry["selected_rank"])
        require([entry["selected_rank"] for entry in ordered] == list(range(1, len(ordered) + 1)),
                "duplicate or noncontiguous selected_rank")
        positions = [entry["position"] for entry in ordered]
        require(len(set(positions)) == len(positions) and set(positions) <= set(pool), "duplicate/unknown selected position")
        wanted = select(list(pool.values()), key[0], key[1], protocol["candidate_k"], key[2])
        require(positions == wanted, "selection differs from frozen ENTR/PROB/RAND definition")
        selected[key] = positions
    require(set(selected) == expected_selection, "missing query policy/repeat selection")
    return pools, actions, selected


def rr(ranked_doc_ids, positives, k):
    for rank, doc in enumerate(ranked_doc_ids[:k], 1):
        if doc in positives:
            return 1.0 / rank
    return 0.0


def per_query_results(queries, corpus, candidates, rankings, selections, groups, protocol, qrels):
    _, actions, selected = validate_inputs(queries, corpus, candidates, rankings, selections, groups, protocol)
    ids = protocol["query_ids"]
    positives = {query: set() for query in ids}
    unique = set()
    for row in qrels:
        query, doc = row["query_id"], row["doc_id"]
        require(query in positives and type(doc) is int, "qrels outside frozen expected queries")
        require((query, doc) not in unique, "duplicate qrels")
        require(doc in {item["doc_id"] for item in corpus}, "qrels document outside corpus")
        unique.add((query, doc))
        if float(row.get("relevance", 1)) > 0:
            positives[query].add(doc)
    require(all(positives.values()), "missing positive qrels for expected query")
    rows = []
    for query in ids:
        base = actions[query, None]["ranked_doc_ids"]
        values = {position: rr(row["ranked_doc_ids"], positives[query], protocol["metric_k"])
                  for (qid, position), row in actions.items() if qid == query}
        result = {"query_id": query, "group_id": groups[query], "NoRewrite": values[None], "policies": {}}
        for policy, seeds in (("ENTR", [None]), ("PROB", [None]), ("RAND", protocol["rand_seeds"])):
            repetitions = []
            for seed in seeds:
                positions = selected[query, policy, seed]
                top_position = positions[0]
                top_ranked = actions[query, top_position]["ranked_doc_ids"]
                delta = values[top_position] - values[None]
                adopted = top_ranked != base
                repetitions.append({"seed": seed, "top1_rr": values[top_position],
                                    "best3_rr": max(values[position] for position in positions),
                                    "top1_delta": delta, "adopted": adopted,
                                    "harmful": adopted and delta < -1e-10,
                                    "selected_positions": positions})
            result["policies"][policy] = {"top1_rr": float(np.mean([row["top1_rr"] for row in repetitions])),
                                           "best3_rr": float(np.mean([row["best3_rr"] for row in repetitions])),
                                           "repetitions": repetitions}
        rows.append(result)
    return rows


def clustered_stats(values, group_ids, *, bootstrap, seed, margin):
    values = np.asarray(values, dtype=np.float64)
    ids = sorted(set(group_ids))
    codes = np.array([ids.index(group) for group in group_ids], dtype=int)
    require(len(values) == len(group_ids) and len(values) > 0, "invalid paired values/groups")
    count = np.bincount(codes, minlength=len(ids)).astype(np.float64)
    totals = np.bincount(codes, weights=values, minlength=len(ids))
    mean = float(values.mean())
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(ids), size=(bootstrap, len(ids)))
    replicates = totals[samples].sum(1) / count[samples].sum(1)
    ci95 = np.quantile(replicates, [.025, .975]).tolist()
    ci90 = np.quantile(replicates, [.05, .95]).tolist()
    ci_family = np.quantile(replicates, [.05 / 4 / 2, 1 - .05 / 4 / 2]).tolist()
    changed = np.abs(values) > 1e-10
    changed_groups = len({group for group, change in zip(group_ids, changed) if change})
    valid_precision = len(ids) >= 30 and changed_groups >= 20 and int(changed.sum()) >= 20
    se = math.sqrt(len(ids) / (len(ids) - 1) * float(np.square(totals - mean * count).sum())) / len(values) if len(ids) > 1 else None
    p = float(2 * student_t.sf(abs(mean / se), len(ids) - 1)) if se is not None and se > 0 and len(ids) >= 30 else None
    return {"queries": len(values), "groups": len(ids), "mean": mean, "ci95_cluster": ci95,
            "ci90_cluster": ci90, "cluster_robust_se": se, "p_two_sided_cluster_t": p,
            "ci98_75_four_contrast_family": ci_family,
            "positive_queries": int((values > 1e-10).sum()), "negative_queries": int((values < -1e-10).sum()),
            "changed_queries": int(changed.sum()), "changed_groups": changed_groups,
            "equivalence_margin": margin, "precision_eligibility": valid_precision,
            "equivalent": None,
            "descriptive_interval_containment": (ci90[0] > -margin and ci90[1] < margin) if valid_precision else None,
            "equivalence_status": "descriptive_development_only" if valid_precision else "insufficient_independent_changes",
            "inference_assumption": "independent connected groups; cluster t is asymptotic, percentile bootstrap resamples whole groups"}


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    result, running = {}, 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[name]))
        result[name] = running
    return result


def summarize(rows, protocol):
    require(len(rows) == len(protocol["query_ids"]), "denominator does not match frozen expected IDs")
    baseline = np.array([row["NoRewrite"] for row in rows])
    group_ids = [row["group_id"] for row in rows]
    stats_kwargs = {"bootstrap": protocol["bootstrap_repetitions"], "seed": protocol["statistics_seed"], "margin": protocol["planning_effect"]}
    summary = {"status": "development_pilot_descriptive", "queries": len(rows), "groups": len(set(group_ids)),
               "missing_rankings": 0, "metric": f'MRR@{protocol["metric_k"]}', "NoRewrite_mrr": float(baseline.mean()),
               "candidate_set_quality": {}, "automatic_top1": {}, "paired_contrasts": {}}
    for policy in ("ENTR", "PROB", "RAND"):
        best3 = np.array([row["policies"][policy]["best3_rr"] for row in rows])
        top1 = np.array([row["policies"][policy]["top1_rr"] for row in rows])
        summary["candidate_set_quality"][policy] = {"mrr": float(best3.mean()), "delta_vs_NoRewrite": clustered_stats(best3 - baseline, group_ids, **stats_kwargs),
                                                    "label_oracle": True, "deployable_policy": False}
        adopted = float(np.mean([np.mean([repeat["adopted"] for repeat in row["policies"][policy]["repetitions"]]) for row in rows]))
        harmful = float(np.mean([np.mean([repeat["harmful"] for repeat in row["policies"][policy]["repetitions"]]) for row in rows]))
        summary["automatic_top1"][policy] = {"mrr": float(top1.mean()), "delta_vs_NoRewrite": clustered_stats(top1 - baseline, group_ids, **stats_kwargs),
                                           "adoption_rate": adopted, "harmful_per_all_queries": harmful,
                                           "harmful_within_adopted": harmful / adopted if adopted else None}
    entr = np.array([row["policies"]["ENTR"]["best3_rr"] for row in rows])
    for control in ("RAND", "PROB"):
        other = np.array([row["policies"][control]["best3_rr"] for row in rows])
        summary["paired_contrasts"][f"ENTR_minus_{control}_best3"] = clustered_stats(entr - other, group_ids, **stats_kwargs)
    repetitions = len(protocol["rand_seeds"])
    rand_mrr = [float(np.mean([row["policies"]["RAND"]["repetitions"][index]["best3_rr"] for row in rows])) for index in range(repetitions)]
    summary["RAND"] = {"repetitions": repetitions, "seeds": protocol["rand_seeds"], "per_repeat_best3_mrr": rand_mrr,
                       "mc_standard_error_best3_mrr": float(np.std(rand_mrr, ddof=1) / math.sqrt(repetitions)) if repetitions > 1 else None,
                       "estimator": "per-query RR averaged over frozen uniform position subsets; no new generation"}
    return summary


def evaluate_run(retriever, run_dir=OUT):
    run_dir = run_dir.resolve()
    require(run_dir == OUT.resolve(), "this entry point is bound to the frozen development run")
    protocol = verify_frozen(include_private=True)
    audit = read(OUT / "PREOUTCOME_AUDIT.json")
    require(audit["passed"] and audit["protocol_sha256"] == sha(OUT / "PROTOCOL.json"), "preoutcome audit missing or stale")
    seal = read(OUT / "DECISIONS_SEALED.json")
    for record in seal["files"]:
        require(sha(OUT / record["path"]) == record["sha256"], "sealed action data changed")
    require(retriever in protocol["retrievers"], "unknown retriever")
    marker = OUT / "metrics" / f"{retriever}_OPENED.json"
    require(not marker.exists(), "development effect evaluation already recorded; use saved results")
    write(marker, {"protocol_sha256": sha(OUT / "PROTOCOL.json"), "role": "development", "confirmation_opened": False}, immutable=True)
    from .common import ADAPTER, now
    qrels = readl(OUT / "data/private/pilot_qrels.jsonl")
    rows = per_query_results(readl(OUT / "data/queries.jsonl"), readl(ADAPTER / "public/corpus.jsonl"),
                             readl(OUT / "candidates.jsonl"), readl(OUT / f"rankings/{retriever}.jsonl"),
                             readl(OUT / "selections.jsonl"), read(OUT / "data/groups.json"), protocol, qrels)
    writel(OUT / f"metrics/{retriever}_per_query.jsonl", rows)
    result = summarize(rows, protocol)
    result.update({"retriever": retriever, "created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                   "qrels_sha256": sha(OUT / "data/private/pilot_qrels.jsonl"), "confirmation_opened": False})
    write(OUT / f"metrics/{retriever}.json", result, immutable=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=__import__("pathlib").Path, default=OUT)
    parser.add_argument("--retriever", choices=("CodeBERT", "BM25"), required=True)
    args = parser.parse_args()
    result = evaluate_run(args.retriever, args.run_dir)
    print({"retriever": args.retriever, "queries": result["queries"], "groups": result["groups"], "NoRewrite_mrr": result["NoRewrite_mrr"]})


if __name__ == "__main__":
    main()
