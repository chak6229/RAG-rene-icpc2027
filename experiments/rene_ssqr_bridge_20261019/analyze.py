"""Open only supplementary/development effects after label-free caches are sealed."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .common import (
    OUT, DEVELOPMENT, PROFILES, read, readl, write, writel, now, sha,
    verify_frozen, verify_seal, require, pools_for, text_id,
)
from .evaluation import evaluate_query, cluster_interval, domain_difference


def calculate(directory=OUT):
    """Pure cached computation; used again from a genuinely separate directory."""
    protocol = read(directory / "PROTOCOL.json")
    joint = read(directory / "data/joint_groups.json")
    selected = readl(directory / "selections.jsonl")
    selection_index = {(r["domain"], r["scope"], r["profile"], r["mask"], r["query_id"]): r for r in selected}
    require(len(selection_index) == len(selected), "duplicate selection key")
    by_view = {}
    per_query = []
    for domain in ("source", "cosqa"):
        queries = {q["query_id"]: q["query"] for q in readl(directory / f"data/{domain}/queries.jsonl")}
        qrels = defaultdict(set)
        for row in readl(directory / f"data/private/{domain}_qrels.jsonl"):
            if row["relevance"] > 0:
                qrels[row["query_id"]].add(row["doc_id"])
        bank = {name: readl(directory / f"candidates/{domain}/{name}.jsonl")
                for name in ("baseline", "greedy20_position0", "greedy10", "sample10")}
        rank = {retriever: {r["text_id"]: r for r in readl(directory / f"rankings/{domain}/{retriever}.jsonl")}
                for retriever in protocol["retrievers"]}
        require(set(queries) == set(protocol["samples"][domain]["main_ids"]) == set(qrels), "query/qrel denominator mismatch")
        for scope, profiles in (("main", ["greedy20_source"]), ("sensitivity", list(PROFILES))):
            ids = protocol["samples"][domain]["main_ids" if scope == "main" else "sensitivity_ids"]
            allowed = set(ids)
            for profile in profiles:
                setting = PROFILES[profile]
                current = bank[setting["bank"]]
                if profile == "greedy20_all":
                    current = current + bank["greedy20_position0"]
                pools = defaultdict(list)
                for row in current:
                    if row["query_id"] in allowed and (setting["positions"] == "paper_0_to_n" or row["position"] != 0):
                        pools[row["query_id"]].append(row)
                require(set(pools) == allowed, "view has missing query candidates")
                for mask in ("full", "lexical"):
                    for retriever in protocol["retrievers"]:
                        view = []
                        for qid in ids:
                            sel = selection_index[(domain, scope, profile, mask, qid)]
                            values = evaluate_query(pools[qid], sel, rank[retriever], qrels[qid], queries[qid], text_id)
                            row = {"domain": domain, "scope": scope, "profile": profile, "mask": mask,
                                   "retriever": retriever, "query_id": qid, "group": joint[domain][qid],
                                   "candidate_count": len(pools[qid]),
                                   "empty_span_count": sum(r["span_token_count"] == 0 for r in pools[qid]), **values}
                            view.append(row)
                            per_query.append(row)
                        by_view[(domain, scope, profile, mask, retriever)] = view
    views = []
    for (domain, scope, profile, mask, retriever), rows in by_view.items():
        groups = [r["group"] for r in rows]
        values = {name: np.array([r[name] for r in rows]) for name in
                  ("NoRewrite", "ENTR_top1", "ENTR_best3", "PROB_best3", "RAND20_best3", "ENTR_minus_RAND", "ENTR_minus_PROB")}
        effect_intervals = {name: cluster_interval(values[name], groups) for name in ("ENTR_minus_RAND", "ENTR_minus_PROB")}
        effect_intervals["ENTR_best3_gain"] = cluster_interval(values["ENTR_best3"] - values["NoRewrite"], groups)
        effect_intervals["ENTR_top1_gain"] = cluster_interval(values["ENTR_top1"] - values["NoRewrite"], groups)
        views.append({"domain": domain, "scope": scope, "profile": profile, "mask": mask, "retriever": retriever,
                      "queries": len(rows), "groups": len(set(groups)),
                      "mrr": {k: float(v.mean()) for k,v in values.items() if "minus" not in k},
                      "contrasts": effect_intervals,
                      "candidate_count_mean": float(np.mean([r["candidate_count"] for r in rows])),
                      "empty_spans": sum(r["empty_span_count"] for r in rows)})
    domain_contrasts = []
    for scope, profiles in (("main", ["greedy20_source"]), ("sensitivity", list(PROFILES))):
        for profile in profiles:
            for retriever in protocol["retrievers"]:
                a = by_view[("source", scope, profile, "full", retriever)]
                b = by_view[("cosqa", scope, profile, "full", retriever)]
                contrast = domain_difference([r["ENTR_minus_RAND"] for r in a], [r["group"] for r in a],
                                             [r["ENTR_minus_RAND"] for r in b], [r["group"] for r in b])
                domain_contrasts.append({"scope": scope, "profile": profile, "retriever": retriever, **contrast})
    pairs = [("position_greedy20", "greedy20_all", "greedy20_source"),
             ("length_source", "greedy10_source", "greedy20_source"),
             ("sampling_source", "sample10_source", "greedy10_source"),
             ("sampling_all", "sample10_all", "greedy10_all"),
             ("paper_oriented_joint", "sample10_all", "greedy20_source")]
    sensitivity_contrasts = []
    for domain in ("source", "cosqa"):
        for retriever in protocol["retrievers"]:
            for name, new_profile, reference in pairs:
                a = by_view[(domain,"sensitivity",new_profile,"full",retriever)]
                b = by_view[(domain,"sensitivity",reference,"full",retriever)]
                require([r["query_id"] for r in a] == [r["query_id"] for r in b], "unpaired sensitivity contrast")
                differences = {"ENTR_minus_RAND_change": [x["ENTR_minus_RAND"]-y["ENTR_minus_RAND"] for x,y in zip(a,b)],
                               "ENTR_best3_change": [x["ENTR_best3"]-y["ENTR_best3"] for x,y in zip(a,b)]}
                sensitivity_contrasts.append({"domain": domain, "retriever": retriever, "contrast": name,
                    "new_profile": new_profile, "reference": reference,
                    **{key: cluster_interval(values,[r["group"] for r in a]) for key,values in differences.items()}})
            for profile in PROFILES:
                a = by_view[(domain,"sensitivity",profile,"lexical",retriever)]
                b = by_view[(domain,"sensitivity",profile,"full",retriever)]
                sensitivity_contrasts.append({"domain": domain,"retriever": retriever,"contrast":"lexical_minus_full",
                    "profile":profile, "ENTR_minus_RAND_change":cluster_interval(
                    [x["ENTR_minus_RAND"]-y["ENTR_minus_RAND"] for x,y in zip(a,b)],[r["group"] for r in a])})
    return {"role": protocol["role"], "views": views, "domain_contrasts": domain_contrasts,
            "sensitivity_contrasts": sensitivity_contrasts,
            "historical_confirmation_changed": False, "new_policy_promoted": False}, per_query


def materialize_replay_inputs():
    """Self-contained cached replay tree; no imports of parent result files needed."""
    import shutil
    destinations = [(DEVELOPMENT / "data/queries.jsonl", OUT / "data/cosqa/queries.jsonl"),
                    (DEVELOPMENT / "data/private/development_qrels.jsonl", OUT / "data/private/cosqa_qrels.jsonl"),
                    (DEVELOPMENT / "candidates.jsonl", OUT / "candidates/cosqa/baseline.jsonl")]
    for source, target in destinations:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        require(sha(source) == sha(target), "replay copy changed baseline bytes")


def analyze():
    verify_frozen(private=True)
    verify_seal("CANDIDATES_SEALED.json")
    verify_seal("RANKINGS_SEALED.json")
    preaudit = read(OUT / "PRE_EFFECT_AUDIT.json")
    require(preaudit["passed"], "pre-effect audit failed")
    if (OUT / "metrics/RESULTS.json").exists():
        return read(OUT / "metrics/RESULTS.json")
    materialize_replay_inputs()
    write(OUT / "EFFECT_OPENING.json", {"created_utc": now(), "role": "supplementary_diagnostic",
          "CoSQA_role": "development_only", "protocol_sha256":sha(OUT / "PROTOCOL.json"),
          "earlier_confirmation_unchanged": True})
    result, rows = calculate()
    writel(OUT / "metrics/per_query.jsonl", rows)
    write(OUT / "metrics/RESULTS.json", result)
    return result


if __name__ == "__main__":
    analyze()
