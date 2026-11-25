from __future__ import annotations

import argparse
import importlib.metadata
import platform
import time
from collections import Counter

import numpy as np

from .common import (DENOMINATOR, OUT, PACKAGE, PARENT, PLAN, POLICIES, PROTECTED,
                     RETRIEVERS, ROOT, fuse, load_public, now, read, readl, record,
                     require, sha, unique_text_lists, verify, write, writel)
from .metrics import CONTRASTS, PRIMARY, intervals, rr, utility


def freeze():
    require(not (OUT / "PROTOCOL.json").exists(), "Already frozen")
    paths = {PARENT / name for name in (
        "PROTOCOL.json", "ROLE_MANIFEST.json", "FINAL_ACCEPTANCE.json", "DECISIONS_SEALED.json",
        "CANDIDATES_SEALED.json", "candidates.jsonl", "selections.jsonl", "data/queries.jsonl",
        "data/groups.json", "data/private/development_qrels.jsonl",
    )}
    paths.update([PLAN, ROOT / "results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl"])
    paths.update(PACKAGE.glob("*.py"))
    require(read(OUT / "TEST_RESULTS.json")["passed"], "Contract tests must pass before freeze")
    paths.add(OUT / "TEST_RESULTS.json")
    for retriever in RETRIEVERS:
        path = PARENT / f"rankings/{retriever}.jsonl"
        manifest = PARENT / f"rankings/{retriever}.manifest.json"
        require(read(manifest)["rankings_sha256"] == sha(path), "Ranking manifest differs")
        paths.update([path, manifest])
    parent_seal = read(PARENT / "DECISIONS_SEALED.json")
    require(parent_seal["protocol_sha256"] == sha(PARENT / "PROTOCOL.json"), "Parent protocol differs")
    checked = 0
    for item in parent_seal["files"]:
        path = PARENT / item["path"]
        if path in paths:
            require(sha(path) == item["sha256"], "Sealed parent input changed")
            checked += 1
    require(read(PARENT / "FINAL_ACCEPTANCE.json")["passed"], "Parent acceptance failed")
    write("PROTOCOL.json", {
        "status": "FROZEN_BEFORE_NEW_FUSION_EFFECTS", "created_utc": now(),
        "role": "consumed_development_post_hoc_extension", "new_confirmation": False,
        "prior_A_B_effects_known": True, "external_preregistration": False,
        "scope": "CoSQA D1490/512, two retrievers, no new generation/training/retrieval/CE or H effects",
        "candidate_pool": "exact parent ENTR three positions; same generation cache as A/B",
        "retrievers": list(RETRIEVERS), "executable_policies": list(POLICIES),
        "oracle_diagnostic": "ENTRBest3, maximum candidate RR, no original included; not executable or an upper bound on fused rankings",
        "RRF": {"constant": 60, "depth": 100, "weights": "equal per unique input text",
                "dedup": "exact retrieval input text, including original if present; verify identical-text rankings and scores",
                "RRF3": "three ENTR selected candidate lists, deduplicated",
                "RRF4": "original + three ENTR selected candidate lists, deduplicated",
                "score": "sum 1/(60+rank) over each unique input Top100",
                "precision": "exact rational ordering using shared LCM integer denominator",
                "shared_denominator": str(DENOMINATOR), "ties": "integer document ID ascending",
                "output": "highest 100 from the actual retrieved-document union"},
        "primary_family": "four effects = RRF3/4 vs NoRewrite x CodeBERT/BM25; pointwise95 and Bonferroni98.75 individual intervals",
        "contrast_definitions": CONTRASTS,
        "secondary": "RRF4-RRF3, fusion vs Top1, best3 opportunity; paired cross-retriever effect differences",
        "statistics": {"grouping": "512 existing connected groups", "weighting": "query-weighted means",
                       "bootstrap_repetitions": 10000, "seed": 20261020, "group_bootstrap": "shared draws for paired contrasts",
                       "cluster_t": "group influence standard errors, G-1 df", "equivalence": "no margin/claim assigned"},
        "cost": "unique retrieval-input counts per query (policy standalone and common four-list cache budget); same exhaustive position generation for rewrite arms; no invented wall-clock/token estimates",
        "risk_gate": False, "risk_certified": False, "tuning_or_weight_search": False,
        "no_policy_promotion": True, "original_sealed_files_checked": checked,
        "bound_files": [record(p) for p in sorted(paths)],
        "protected_files": [record(ROOT / name) for name in PROTECTED],
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "scipy")}},
    })
    print({"fusion_frozen": True, "primary_effects": 4, "dedup": "exact text", "precision": "exact rational"}, flush=True)


def construct():
    verify()
    require(not (OUT / "ACTIONS_SEALED.json").exists(), "Fusion actions already sealed")
    start = time.perf_counter()
    metadata, files = {}, []
    for retriever in RETRIEVERS:
        qids, groups, selections, pools, generations = load_public(retriever)
        rows = []
        for q in qids:
            original, *candidates = pools[q]
            fusion3, meta3 = fuse(candidates)
            fusion4, meta4 = fuse([original]+candidates)
            costs = {"NoRewrite": 1, "ENTRTop1": 1, "RRF3": meta3["unique_lists"], "RRF4": meta4["unique_lists"]}
            rows.append({"retriever": retriever, "query_id": q, "group": groups[q],
                         "selected_positions": selections[q], "generation_positions": generations[q],
                         "rankings": {"NoRewrite": original["ranked_doc_ids"], "ENTRTop1": candidates[0]["ranked_doc_ids"],
                                      "RRF3": fusion3, "RRF4": fusion4},
                         "fusion_metadata": {"RRF3": meta3, "RRF4": meta4},
                         "cost_unique_retrieval_inputs": costs,
                         "common_original_plus_ENTR3_inputs": meta4["unique_lists"],
                         "original_plus_top1_inputs": len(unique_text_lists([original,candidates[0]])),
                         "qrels_read": False})
        writel(f"actions/{retriever}.jsonl", rows)
        files.append(OUT / f"actions/{retriever}.jsonl")
        metadata[retriever] = {"queries": len(qids), "groups": len(set(groups.values())),
                               "generation_positions_shared_by_all_rewrite_arms": sum(generations.values()),
                               "unique_list_count_histograms": {p: dict(sorted(Counter(r["cost_unique_retrieval_inputs"][p] for r in rows).items())) for p in POLICIES}}
        print({"constructed": retriever, "queries": len(rows), "labels_read": False}, flush=True)
    write("ACTIONS_SEALED.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                                    "before_new_effects_evaluated": True, "qrels_read_by_action_construction": False,
                                    "metadata": metadata, "construction_seconds": time.perf_counter()-start,
                                    "files": [record(p) for p in files]})


def verify_actions():
    seal = read(OUT / "ACTIONS_SEALED.json")
    require(seal["protocol_sha256"] == sha(OUT / "PROTOCOL.json"), "Protocol/action seal differs")
    for item in seal["files"]:
        require(sha(ROOT / item["path"]) == item["sha256"], "Action ranking changed")
    return seal


def evaluate():
    verify()
    seal = verify_actions()
    require(not (OUT / "RESULTS.json").exists(), "Results exist")
    write("OUTCOME_ACCESS.json", {"created_utc": now(), "purpose": "evaluate sealed label-free policies on consumed D",
                                  "actions_seal_sha256": sha(OUT / "ACTIONS_SEALED.json"), "H_effects_read": False})
    qrels = {}
    for row in readl(PARENT / "data/private/development_qrels.jsonl"):
        if row.get("relevance", 1) > 0:
            qrels.setdefault(row["query_id"], set()).add(row["doc_id"])
    views, per_query, paired = {}, [], {}
    for retriever in RETRIEVERS:
        qids, groups, _, pools, _ = load_public(retriever)
        require(set(qrels) == set(qids), "Wrong qrels role/population")
        actions = readl(OUT / f"actions/{retriever}.jsonl")
        require([r["query_id"] for r in actions] == qids, "Action query order differs")
        values = {p: np.asarray([rr(r["rankings"][p], qrels[r["query_id"]]) for r in actions]) for p in POLICIES}
        values["ENTRBest3"] = np.asarray([max(rr(row["ranked_doc_ids"], qrels[q]) for row in pools[q][1:]) for q in qids])
        effects = {name: values[left]-values[right] for name, (left,right) in CONTRASTS.items()}
        group_list = [groups[q] for q in qids]
        estimates = intervals(np.column_stack(list(effects.values())), group_list, list(effects), family_names=PRIMARY)
        summaries, costs = {}, {}
        for p in values:
            changed = np.asarray([r["rankings"][p] != r["rankings"]["NoRewrite"] for r in actions]) if p in POLICIES else None
            summaries[p] = utility(values[p], values["NoRewrite"], changed)
            if p in POLICIES:
                counts = [r["cost_unique_retrieval_inputs"][p] for r in actions]
                costs[p] = {"standalone_unique_retrieval_inputs_total": sum(counts),
                            "standalone_unique_retrieval_inputs_mean": float(np.mean(counts)),
                            "standalone_unique_retrieval_inputs_min": min(counts),
                            "standalone_unique_retrieval_inputs_max": max(counts),
                            "position_generations_for_this_policy": 0 if p == "NoRewrite" else 9915,
                            "new_generation_or_retrieval_this_replay": 0}
        transition = {"RRF3_better_than_best_single_candidate_queries": int((values["RRF3"] > values["ENTRBest3"]).sum()),
                      "RRF4_better_than_best_single_candidate_queries": int((values["RRF4"] > values["ENTRBest3"]).sum()),
                      "RRF4_beats_RRF3_queries": int((values["RRF4"] > values["RRF3"]).sum()),
                      "RRF4_hurts_RRF3_queries": int((values["RRF4"] < values["RRF3"]).sum())}
        for i, action in enumerate(actions):
            per_query.append({"retriever": retriever, "query_id": action["query_id"], "group": action["group"],
                              "RR": {p: float(v[i]) for p,v in values.items()},
                              "effects": {c: float(v[i]) for c,v in effects.items()},
                              "ranking_changed": {p: action["rankings"][p] != action["rankings"]["NoRewrite"] for p in POLICIES},
                              "cost_unique_retrieval_inputs": action["cost_unique_retrieval_inputs"]})
        views[retriever] = {"population": {"queries": len(qids), "groups": len(set(groups.values()))},
                            "policies": summaries, "estimates": estimates, "costs": costs,
                            "common_four_list_available_cache_budget_total": sum(r["common_original_plus_ENTR3_inputs"] for r in actions),
                            "original_plus_top1_budget_total": sum(r["original_plus_top1_inputs"] for r in actions),
                            "dedup_metadata": seal["metadata"][retriever], "transition_diagnostics": transition}
        paired[retriever] = {"qids": qids, "groups": group_list, "effects": effects}
    require(paired["CodeBERT"]["qids"] == paired["BM25"]["qids"] and paired["CodeBERT"]["groups"] == paired["BM25"]["groups"], "Cross-retriever pairing differs")
    differences = {c: paired["CodeBERT"]["effects"][c]-paired["BM25"]["effects"][c] for c in ("RRF3-NoRewrite", "RRF4-NoRewrite", "RRF4-RRF3")}
    cross = intervals(np.column_stack(list(differences.values())), paired["CodeBERT"]["groups"], list(differences))
    write("RESULTS.json", {"created_utc": now(), "role": "consumed_development_post_hoc_extension",
                            "new_confirmation": False, "risk_certified": False,
                            "views": views, "cross_retriever_CodeBERT_minus_BM25": cross})
    writel("per_query.jsonl", per_query)
    report(views, cross)
    verify()
    verify_actions()
    print({"fusion_effects_evaluated": True, "query_retriever_rows": len(per_query), "official_state_changed": False}, flush=True)


def report(views, cross):
    lines = ['# Multi-candidacy integration: implementable baseline experiment', "",
             'CoSQA D1490 Query/ 512 Associated group, CodeBERT/BM25, the same ENTR candidate for freezing local T5. Compare NoRewrite, ENTRTop1, RRF3 (three candidates) with RRF4 (original query + three candidates), ENTRBest3 only as a labeled opportunity indicator.',
             "", 'RRF Constant 60, enter Top100 to reweight the text by exactly the same input, equal, precise fractional sorting, document ID determine parallel rows, output and set Top100. No action qrels, reassembly after construction. No new generation, training, retrieval, CE or risk doors, no sweep parameters.', ""]
    for retriever, v in views.items():
        lines += [f"## {retriever}", "", 'Programme MRR |Δ MRR Group bootstrap95%', "|---|---:|---:|---|---|---:|---:|"]
        for p, u in v["policies"].items():
            name = f"{p}-NoRewrite"
            e = v["estimates"].get(name)
            ci = e["ci95_group_bootstrap"] if e else [0.,0.]
            fci = e.get("four_effect_family95_bonferroni_group_bootstrap") if e else None
            lines.append(f"|{p}|{u['mrr']:.8f}|{u['delta_mrr']:+.8f}|[{ci[0]:+.8f}, {ci[1]:+.8f}]|" + (f"[{fci[0]:+.8f}, {fci[1]:+.8f}]" if fci else "—") + f"|{u['beneficial_queries']}/{u['harmful_queries']}|{u['harm_mass_per_query']:.8f}|")
        lines += ["", 'Secondary pair comparison MRR Gap group bootstrap95%', "|---|---:|---|"]
        for name in ("RRF4-RRF3", "RRF3-ENTRTop1", "RRF4-ENTRTop1"):
            e=v["estimates"][name]; ci=e["ci95_group_bootstrap"]
            lines.append(f"|{name}|{e['mean']:+.8f}|[{ci[0]:+.8f}, {ci[1]:+.8f}]|")
        lines += ["", 'The only search required for the online strategy is | query averaged | position generation (shared pool) |', "|---|---:|---:|---:|"]
        for p, c in v["costs"].items():
            lines.append(f"|{p}|{c['standalone_unique_retrieval_inputs_total']}|{c['standalone_unique_retrieval_inputs_mean']:.4f}|{c['position_generations_for_this_policy']}|")
        lines += ["", f"The total search input common to the four lists in the controlled comparison is:{v['common_four_list_available_cache_budget_total']};originally query+Top1 with{v['original_plus_top1_budget_total']}。 This is not proof of the same cost of strategy online. The original experiment was actually carried out in bulk/cachel and the table is the only input required per query; Do not delay, token or amount.", ""]
    lines += ['## Interpretation of the boundary', "", 'Integration produces new rankings, which may exceed best3 and best3 cannot be the denominator of integration upward or yield retention rates. RRF4 The inclusion of original queries does not mean that the guarantees are intact; Harmful inquiries and loss ranges are reported. The three candidates are still screened for primary entropy, so the experiment does not require any extra UQ integration of the predictor head, rather than eliminating all the cross-references to entropy information.',
              "", 'The data have been consumed and the agreement has been frozen only before this new integration effect is calculated. (b) Inter-group descriptions of the differences in the development and do not constitute new independent confirmations; Four main effects provide Bonferroni sensitivity. Cross-retrieval is directly modulated at RESULTS.json, and no significant inferences are made by the obvious side. Standard RRF is not a new algorithm; See FINDINGINGS.md and PRIOR_WORK_COMPARISON.md.',
              "", 'Independent acceptance and inspection AUDIT.json, FINAL_ACCEPTANCE json, covering full sorting of Fraction, weighting, action/cost, by queryRR, indicator ranges and freezing of Hashi. Old A/B, old H confirmation and official five pointers.']
    with (OUT / "READOUT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "construct", "evaluate"))
    args = parser.parse_args()
    {"freeze":freeze, "construct":construct, "evaluate":evaluate}[args.stage]()
