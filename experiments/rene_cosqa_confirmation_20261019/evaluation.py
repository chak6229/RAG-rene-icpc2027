"""One outcome opening after all observable decisions and pre-audits are sealed."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from experiments.rene_cosqa_pilot_20261019.evaluation import rr, validate_inputs
from .common import (ADAPTER, OUT, RETRIEVERS, check_seal, now, read, readl, require, sha,
                     status, verify_design, write, writel)
from .statistics import interval, ratio_interval

POLICIES = ("NoRewrite", "ENTR_top1", "PROB_in_ENTR3", "HGB_gain_control", "RIDGE_gain", "RIDGE_verified", "RIDGE_risk")


def primary_conclusion(result):
    a, b = result["ci95_cluster_t"], result["ci95_group_bootstrap"]
    if result["changed_groups"] < 20:
        return "insufficient independent changes; unresolved"
    if a[0] > 0 and b[0] > 0:
        direction = "ENTR advantage over RAND20 supported for the frozen capped sample"
    elif a[1] < 0 and b[1] < 0:
        direction = "ENTR lower than RAND20 for the frozen capped sample"
    else:
        direction = "difference from zero unresolved"
    equivalent = min(a[0], b[0]) > -.005 and max(a[1], b[1]) < .005
    return {"direction": direction, "practical_equivalence_within_prespecified_plusminus005": equivalent,
            "advantage_above005_excluded": bool(max(a[1], b[1]) < .005),
            "does_not_imply_all_UQ_is_ineffective": True}


def build_rows(retriever, p, positives):
    queries = readl(OUT / "public/H_queries.jsonl")
    groups = read(OUT / "public/H_groups.json")
    corpus = readl(ADAPTER / "public/corpus.jsonl")
    candidates = readl(OUT / "candidates.jsonl")
    rankings = readl(OUT / f"rankings/{retriever}.jsonl")
    selections = readl(OUT / "selections.jsonl")
    contract = {**p, "query_ids": p["H_query_ids"]}
    _, actions, selected = validate_inputs(queries, corpus, candidates, rankings, selections, groups, contract)
    rr_actions = {(q, pos): rr(row["ranked_doc_ids"], positives[q], 100) for (q, pos), row in actions.items()}
    decisions = {}
    for r in readl(OUT / f"decisions/{retriever}.jsonl"):
        key = r["query_id"], r["policy"]
        require(key not in decisions and r["policy"] in POLICIES, "duplicate/unknown stage decision")
        require(len(r["ranking"]) == 100 and len(set(r["ranking"])) == 100, "invalid decision ranking")
        decisions[key] = r
    require(set(decisions) == {(q, name) for q in p["H_query_ids"] for name in POLICIES}, "decision denominator differs")
    rows = []
    for q in p["H_query_ids"]:
        base = rr_actions[q, None]
        values = {}
        for name in POLICIES:
            d = decisions[q, name]
            value = rr(d["ranking"], positives[q], 100)
            require(d["accepted"] == (d["ranking"] != actions[q, None]["ranked_doc_ids"]), "adoption is not actual ranking change")
            values[name] = {"rr": value, "delta": value - base, "adopted": bool(d["accepted"]),
                            "verified": bool(d["verified"]), "pairs": d["actual_pairs"], "tokens": d["actual_input_tokens"]}
        quality = {}
        for policy, seeds in (("ENTR", [None]), ("PROB", [None]), ("RAND", p["rand_seeds"])):
            reps = [max(rr_actions[q, pos] for pos in selected[q, policy, seed]) for seed in seeds]
            quality[policy] = float(np.mean(reps))
        rows.append({"query_id": q, "group_id": groups[q], "NoRewrite": base,
                     "candidate_best3": quality, "safe_ENTR_oracle_delta": max(0., quality["ENTR"] - base),
                     "policies": values})
    return rows


def summarize(rows, retriever):
    groups = [r["group_id"] for r in rows]
    baseline = np.array([r["NoRewrite"] for r in rows])
    oracle = np.array([r["safe_ENTR_oracle_delta"] for r in rows])
    result = {"retriever": retriever, "queries": len(rows), "groups": len(set(groups)),
              "NoRewrite_mrr": float(baseline.mean()), "safe_ENTR_oracle_delta": float(oracle.mean()),
              "policy_results": {}, "candidate_quality": {}, "stage_contrasts": {},
              "auxiliary_only": True, "null_rejection_claims": False, "risk_certified": False}
    deltas = {}
    for policy in ("ENTR", "PROB", "RAND"):
        result["candidate_quality"][policy] = {"best3_mrr": float(np.mean([r["candidate_best3"][policy] for r in rows])),
                                                "delta": float(np.mean([r["candidate_best3"][policy] - r["NoRewrite"] for r in rows]))}
    for name in POLICIES:
        delta = np.array([r["policies"][name]["delta"] for r in rows])
        adopted = np.array([r["policies"][name]["adopted"] for r in rows], int)
        harmed = (delta < -1e-10).astype(int)
        require(np.all(harmed <= adopted), "harm without nonidentity adoption")
        deltas[name] = delta
        result["policy_results"][name] = {
            "mrr": float((baseline + delta).mean()), "delta_MRR": interval(delta, groups),
            "adopted": int(adopted.sum()), "coverage": float(adopted.mean()), "harmed": int(harmed.sum()),
            "improved": int(np.count_nonzero(delta > 1e-10)),
            "harm_within_adopted": ratio_interval(harmed, adopted, groups),
            "positive_gain": float(np.maximum(delta, 0.).mean()), "loss_magnitude": float(np.maximum(-delta, 0.).mean()),
            "positive_gain_retention": ratio_interval(np.maximum(delta, 0.), oracle, groups),
            "verified_queries": sum(r["policies"][name]["verified"] for r in rows),
            "CE_pairs_charged": sum(r["policies"][name]["pairs"] for r in rows),
            "CE_input_tokens_charged": sum(r["policies"][name]["tokens"] for r in rows)}
    pairs = (("RIDGE_gain", "NoRewrite"), ("RIDGE_verified", "RIDGE_gain"),
             ("RIDGE_risk", "RIDGE_verified"), ("RIDGE_risk", "NoRewrite"),
             ("RIDGE_gain", "HGB_gain_control"))
    for a, b in pairs:
        result["stage_contrasts"][a + "_minus_" + b] = interval(deltas[a] - deltas[b], groups)
    pre, post = deltas["RIDGE_verified"], deltas["RIDGE_risk"]
    rejected = np.array([r["policies"]["RIDGE_verified"]["adopted"] and not r["policies"]["RIDGE_risk"]["adopted"] for r in rows])
    missed = np.maximum(pre, 0.) * rejected
    avoided = np.maximum(-pre, 0.) * rejected
    error = float(abs(post.mean() - (pre.mean() - missed.mean() + avoided.mean())))
    require(error < 1e-14, "gate gain/loss conservation failed")
    require(result["policy_results"]["RIDGE_verified"]["CE_pairs_charged"] == result["policy_results"]["RIDGE_risk"]["CE_pairs_charged"], "gate refunded sunk cost")
    result["gate_accounting"] = {"before_delta": float(pre.mean()), "after_delta": float(post.mean()),
                                 "rejected_nonidentity_actions": int(rejected.sum()),
                                 "rejected_positive_actions": int(np.sum(rejected & (pre > 1e-10))),
                                 "rejected_harmful_actions": int(np.sum(rejected & (pre < -1e-10))),
                                 "missed_positive_gain": float(missed.mean()), "avoided_loss": float(avoided.mean()),
                                 "identity_max_error": error,
                                 "missed_positive_interval": interval(missed, groups), "avoided_loss_interval": interval(avoided, groups),
                                 "accounting_identity_is_not_a_standalone_novel_algorithm": True}
    return result


def main():
    p = verify_design()
    check_seal("POLICY_SEALED.json")
    check_seal("DECISIONS_SEALED.json")
    require(read(OUT / "PREOUTCOME_AUDIT.json")["passed"], "pre-audit required before opening")
    require(not (OUT / "OUTCOME_OPENING.json").exists(), "H already consumed; replay frozen outputs instead")
    # First mark access, then read annotations. An interrupted opening still consumes H.
    write(OUT / "OUTCOME_OPENING.json", {"opened_utc": now(), "role": "one_shot_independent_confirmation",
                                         "design_sha256": sha(OUT / "DESIGN.json"), "policy_sha256": sha(OUT / "PROTOCOL.json"),
                                         "decisions_sha256": sha(OUT / "DECISIONS_SEALED.json"),
                                         "preaudit_sha256": sha(OUT / "PREOUTCOME_AUDIT.json"),
                                         "H_queries_consumed": 8530, "H_groups_consumed": 1963,
                                         "post_open_policy_selection_forbidden": True}, immutable=True)
    status("H_outcomes_opened_evaluating")
    hset = set(p["H_query_ids"])
    labels = [r for r in readl(ADAPTER / "private/all_positive_qrels.jsonl") if r["query_id"] in hset]
    require(len({(r["query_id"], r["doc_id"]) for r in labels}) == len(labels), "duplicate H qrels")
    positives = {q: set() for q in hset}
    for r in labels:
        if r.get("relevance", 1) > 0:
            positives[r["query_id"]].add(r["doc_id"])
    require(all(positives.values()), "missing H positive qrels")
    writel(OUT / "private/H_qrels.jsonl", labels)
    rows_by_ret, summaries = {}, {}
    primary_ids = set(read(OUT / "PRIMARY_SAMPLE.json")["query_ids"])
    for r in RETRIEVERS:
        rows = build_rows(r, p, positives)
        rows_by_ret[r] = rows
        writel(OUT / f"metrics/{r}_per_query.jsonl", rows)
        summaries[r] = summarize(rows, r)
        write(OUT / f"metrics/{r}_full_H.json", summaries[r], immutable=True)
        print({"evaluated": r, "all_H": len(rows), "final_delta": summaries[r]["policy_results"]["RIDGE_risk"]["delta_MRR"]["mean"]}, flush=True)
    chosen = [r for r in rows_by_ret["CodeBERT"] if r["query_id"] in primary_ids]
    require(len(chosen) == 7718 and len({r["group_id"] for r in chosen}) == 1963, "primary sample drift")
    effect = [r["candidate_best3"]["ENTR"] - r["candidate_best3"]["RAND"] for r in chosen]
    primary = interval(effect, [r["group_id"] for r in chosen])
    primary.update(alpha=.05, comparison=p["primary_comparison"], only_confirmatory_test=True,
                   conclusion=primary_conclusion(primary), precision_plan_sha256=sha(OUT / "PRECISION_PLAN.json"))
    write(OUT / "PRIMARY_RESULT.json", primary, immutable=True)
    aux = {}
    for ret, rows in rows_by_ret.items():
        for sample, current in (("capped16", [x for x in rows if x["query_id"] in primary_ids]), ("full8530", rows)):
            for other in ("RAND", "PROB"):
                aux[f"{ret}:{sample}:ENTR_minus_{other}"] = interval(
                    [x["candidate_best3"]["ENTR"] - x["candidate_best3"][other] for x in current], [x["group_id"] for x in current])
    by_ret = {r: {v["query_id"]: v for v in rows} for r, rows in rows_by_ret.items()}
    groups = [by_ret["CodeBERT"][q]["group_id"] for q in p["H_query_ids"]]
    for name, control in (("RIDGE_gain", "NoRewrite"), ("RIDGE_verified", "RIDGE_gain"), ("RIDGE_risk", "RIDGE_verified")):
        v = [(by_ret["CodeBERT"][q]["policies"][name]["delta"] - by_ret["CodeBERT"][q]["policies"][control]["delta"])
             - (by_ret["BM25"][q]["policies"][name]["delta"] - by_ret["BM25"][q]["policies"][control]["delta"])
             for q in p["H_query_ids"]]
        aux[f"CodeBERT_minus_BM25:{name}_minus_{control}"] = interval(v, groups)
    write(OUT / "AUXILIARY_RESULTS.json", {"comparisons": aux, "predefined_descriptive_only": True,
                                           "no_additional_confirmatory_rejections": True}, immutable=True)
    status("H_evaluation_complete_awaiting_independent_replay")


if __name__ == "__main__":
    main()
