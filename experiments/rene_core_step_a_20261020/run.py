"""Freeze and execute step A using immutable source/D caches, with no inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.rene_cosqa_pilot_20261019.selection import select
from .metrics import CONTRASTS, clustered_summary, effects, exact_random_max, rr

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / "results/rene_core_step_a_20261020"
BRIDGE = ROOT / "results/rene_ssqr_bridge_20261019"
DESIGN = ROOT / "ResearchVault/Handoffs/RENE-核心贡献判别验证方案-20261020.md"
PROTECTED = []  # Unrelated author-workspace registries are not part of this release.


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def readl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def record(path):
    return {"path": Path(path).relative_to(ROOT).as_posix(), "sha256": sha(path)}


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def require(value, message):
    if not value:
        raise ValueError(message)


def freeze():
    require(not (OUT / "PROTOCOL.json").exists(), "Step-A protocol already exists")
    bridge = read(BRIDGE / "PROTOCOL.json")
    paths = {BRIDGE / name for name in (
        "PROTOCOL.json", "CANDIDATES_SEALED.json", "RANKINGS_SEALED.json", "FINAL_ACCEPTANCE.json",
        "selections.jsonl", "metrics/RESULTS.json", "metrics/per_query.jsonl",
    )}
    for domain in ("source", "cosqa"):
        paths.update(BRIDGE / name for name in (
            f"data/{domain}/queries.jsonl", f"data/private/{domain}_qrels.jsonl",
            f"candidates/{domain}/baseline.jsonl",
            f"rankings/{domain}/CodeBERT.jsonl", f"rankings/{domain}/BM25.jsonl",
        ))
    paths.update([DESIGN, ROOT / "experiments/rene_cosqa_pilot_20261019/selection.py"])
    paths.update(PACKAGE.glob("*.py"))
    # These are existing-source outcome seals, not a new confirmation or fresh labels.
    seal_checks = []
    for name in ("CANDIDATES_SEALED.json", "RANKINGS_SEALED.json"):
        seal = read(BRIDGE / name)
        require(seal["protocol_sha256"] == sha(BRIDGE / "PROTOCOL.json"), "Source seal protocol differs")
        for item in seal["files"]:
            require(sha(ROOT / item["path"]) == item["sha256"], f"Source input changed: {item['path']}")
        seal_checks.append({"seal": name, "files_verified": len(seal["files"])})
    require(bridge["rand_seeds"] == list(range(2026101900, 2026101920)), "RAND20 contract drift")
    write("PROTOCOL.json", {
        "status": "FROZEN_BEFORE_THIS_REPLAY",
        "created_utc": now(),
        "role": "post_hoc_paired_diagnostic_on_consumed_development_and_source",
        "existing_effects_already_known": True,
        "external_preregistration": False,
        "new_confirmation": False,
        "scope": "step A only; no B/C heads, inference, retraining, new generation or H outcomes",
        "population": {domain: {"queries": len(bridge["samples"][domain]["main_ids"]),
                                "groups": len(set(bridge["samples"][domain]["groups"].values()))}
                       for domain in ("source", "cosqa")},
        "profile": "greedy20_source", "mask": "full", "scope_filter": "main",
        "retrievers": ["CodeBERT", "BM25"],
        "top_k": 3, "depth": 100, "include_original_in_best3": False,
        "rand_seeds": bridge["rand_seeds"],
        "single_random": "uniform expectation over distinct positions, integrated exactly",
        "three_random_primary": "same cached 20 fixed random permutations as original bridge",
        "three_random_sensitivity": "exact expected maximum over uniformly drawn position subsets; no replacement",
        "contrast_definitions": CONTRASTS,
        "groups": "original within-domain source repositories / CoSQA connected groups; query-weighted means",
        "bootstrap": {"repetitions": 10000, "seed": 20261020, "shared_draws_across_contrasts": True},
        "intervals": "pointwise cluster-t95 and group-bootstrap95, descriptive",
        "family_adjustment": "four ENTR j contrasts across 2 datasets x 2 retrievers: Bonferroni 98.75% individual intervals, descriptive robustness",
        "equivalence": "No new equivalence decision/SESOI imposed on D or source; original H .005 rule is unchanged",
        "cross_retriever": "direct paired difference of ENTR j within each dataset; secondary descriptive",
        "no_policy_promotion": True,
        "source_seals_verified": seal_checks,
        "bound_files": [record(path) for path in sorted(paths)],
        "protected_files": [record(ROOT / name) for name in PROTECTED],
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "scipy")}},
    })
    print({"step_A_frozen": True, "role": "post_hoc", "new_models": False}, flush=True)


def verify():
    protocol = read(OUT / "PROTOCOL.json")
    for item in protocol["bound_files"] + protocol["protected_files"]:
        require(sha(ROOT / item["path"]) == item["sha256"], f"Frozen file changed: {item['path']}")
    return protocol


def text_id(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def calculate():
    started = time.perf_counter()
    require(not (OUT / "RESULTS.json").exists(), "Step A results already exist")
    protocol = verify()
    bridge = read(BRIDGE / "PROTOCOL.json")
    selection_rows = readl(BRIDGE / "selections.jsonl")
    selection = {(row["domain"], row["query_id"]): row for row in selection_rows
                 if row["scope"] == "main" and row["profile"] == "greedy20_source" and row["mask"] == "full"}
    require(len(selection) == 3538, "Unexpected main-view selection denominator")
    prior_rows = {(row["domain"], row["retriever"], row["query_id"]): row
                  for row in readl(BRIDGE / "metrics/per_query.jsonl")
                  if row["scope"] == "main" and row["profile"] == "greedy20_source" and row["mask"] == "full"}
    views, ledger, per_domain = [], [], {}
    totals = Counter()
    prior_error = 0.0
    for domain in ("source", "cosqa"):
        ids = bridge["samples"][domain]["main_ids"]
        groups = bridge["samples"][domain]["groups"]
        queries = {row["query_id"]: row["query"] for row in readl(BRIDGE / f"data/{domain}/queries.jsonl")}
        qrels = defaultdict(set)
        for row in readl(BRIDGE / f"data/private/{domain}_qrels.jsonl"):
            if row.get("relevance", 1) > 0:
                qrels[row["query_id"]].add(row["doc_id"])
        pools = defaultdict(list)
        for row in readl(BRIDGE / f"candidates/{domain}/baseline.jsonl"):
            pools[row["query_id"]].append(row)
        require(set(ids) == set(queries) == set(qrels) == set(pools), "Population/qrels/pool mismatch")
        selected = {}
        for qid in ids:
            rows = sorted(pools[qid], key=lambda row: row["position"])
            words = queries[qid].split()
            require([row["position"] for row in rows] == list(range(1, len(words) + 1)), "Incomplete position pool")
            require(all(math.isfinite(row["full_vocab_entropy"]) and math.isfinite(row["label_mean_log_probability"]) for row in rows), "Nonfinite score")
            for row in rows:
                p = row["position"]
                require(row["rewrite"] == " ".join(words[:p] + ([row["span"]] if row["span"] else []) + words[p:]), "Rewrite identity differs")
                require(row["masked_query"] == " ".join(words[:p] + ["<extra_id_0>"] + words[p:]), "Masked input differs")
            entr = select(rows, qid, "ENTR", 3)
            prob = select(rows, qid, "PROB", 3)
            rand = [select(rows, qid, "RAND", 3, seed) for seed in protocol["rand_seeds"]]
            old = selection[domain, qid]
            require(entr == old["ENTR"] and prob == old["PROB"] and rand == old["RAND"], "Selections differ from frozen source")
            selected[qid] = (rows, entr, prob, rand)
            totals["selection_queries_reproduced"] += 1
        for retriever in protocol["retrievers"]:
            rank_rows = readl(BRIDGE / f"rankings/{domain}/{retriever}.jsonl")
            rank = {row["text_id"]: row for row in rank_rows}
            require(len(rank) == len(rank_rows), "Duplicate text IDs in ranking cache")
            for key, row in rank.items():
                require(text_id(row["text"]) == key, "Ranking/text hash mismatch")
                require(len(row["ranked_doc_ids"]) == len(set(row["ranked_doc_ids"])) == len(row["scores"]) == 100, "Top100 completeness differs")
                require(all(math.isfinite(s) for s in row["scores"]), "Nonfinite ranking score")
                require(list(zip(row["ranked_doc_ids"], row["scores"])) == sorted(zip(row["ranked_doc_ids"], row["scores"]), key=lambda item: (-item[1], item[0])), "Ranking tie/sort contract differs")
            totals["ranking_texts_checked"] += len(rank)
            cell = []
            for qid in ids:
                rows, entr, prob, rand = selected[qid]
                base = rr(rank[text_id(queries[qid])]["ranked_doc_ids"], qrels[qid])
                scores = {row["position"]: rr(rank[text_id(row["rewrite"])]["ranked_doc_ids"], qrels[qid]) for row in rows}
                totals["query_position_RR_evaluations"] += len(scores) + 1
                values = {
                    "NoRewrite": base,
                    "ENTR_top1": scores[entr[0]], "PROB_top1": scores[prob[0]],
                    "ENTR_best3": max(scores[p] for p in entr), "PROB_best3": max(scores[p] for p in prob),
                    "Uniform_single_exact": math.fsum(scores.values()) / len(scores),
                    "RAND20_best3": math.fsum(max(scores[p] for p in positions) for positions in rand) / 20,
                    "Uniform_best3_exact": exact_random_max(list(scores.values()), 3),
                }
                contrast = effects(values)
                require(abs(contrast["j_ENTR_contract_difference"] - (contrast["a_ENTR_single_minus_uniform"] - contrast["b_ENTR_best3_minus_RAND20"])) < 1e-14, "Paired j identity differs")
                old = prior_rows[domain, retriever, qid]
                prior_error = max(prior_error, *(abs(values[name] - old[name]) for name in ("NoRewrite", "ENTR_top1", "ENTR_best3", "PROB_best3", "RAND20_best3")))
                row = {"domain": domain, "retriever": retriever, "query_id": qid, "group": groups[qid],
                       "position_count": len(rows), "values": values, "contrasts": contrast,
                       "ENTR_positions": entr, "PROB_positions": prob,
                       "candidate_RR": [[p, score] for p, score in scores.items()]}
                cell.append(row)
                ledger.append(row)
            names = list(CONTRASTS)
            matrix = [[row["contrasts"][name] for name in names] for row in cell]
            summary = clustered_summary(matrix, [groups[qid] for qid in ids], names,
                                        repetitions=protocol["bootstrap"]["repetitions"], seed=protocol["bootstrap"]["seed"])
            means = {key: float(np.mean([row["values"][key] for row in cell])) for key in cell[0]["values"]}
            views.append({"domain": domain, "retriever": retriever, "queries": len(ids), "groups": len(set(groups.values())),
                          "mrr": means, "contrasts": summary,
                          "exact_random3_sensitivity": {"RAND20_minus_exact_mean": summary["RAND20_minus_uniform3_exact"]["mean"],
                                                        "max_absolute_query_MC_difference": max(abs(row["contrasts"]["RAND20_minus_uniform3_exact"]) for row in cell)}})
            per_domain[domain, retriever] = cell
            print({"computed": f"{domain}/{retriever}", "queries": len(cell),
                   "a": summary["a_ENTR_single_minus_uniform"]["mean"],
                   "b": summary["b_ENTR_best3_minus_RAND20"]["mean"],
                   "j": summary["j_ENTR_contract_difference"]}, flush=True)
            del rank, rank_rows
        del selected, pools
    require(prior_error < 1e-12, "Original per-query metrics do not reproduce")
    cross = []
    for domain in ("source", "cosqa"):
        dense, lexical = per_domain[domain, "CodeBERT"], per_domain[domain, "BM25"]
        require([row["query_id"] for row in dense] == [row["query_id"] for row in lexical], "Unpaired retriever comparison")
        values = [[a["contrasts"]["j_ENTR_contract_difference"] - b["contrasts"]["j_ENTR_contract_difference"]] for a, b in zip(dense, lexical)]
        cross.append({"domain": domain, "contrast": "j_CodeBERT_minus_j_BM25", **clustered_summary(
            values, [row["group"] for row in dense], ["difference"],
            repetitions=protocol["bootstrap"]["repetitions"], seed=protocol["bootstrap"]["seed"])["difference"]})
    with (OUT / "per_query.jsonl").open("x", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    verify()
    write("RESULTS.json", {
        "created_utc": now(), "role": protocol["role"], "protocol_sha256": sha(OUT / "PROTOCOL.json"),
        "views": views, "cross_retriever_j_differences": cross,
        "integrity": {**totals, "prior_metric_maximum_error": prior_error,
                      "no_H_qrels_or_rankings_loaded": True, "no_model_inference": True,
                      "protected_files_unchanged": True},
        "seconds": time.perf_counter() - started,
        "new_confirmation": False, "new_method_efficacy_established": False,
        "new_policy_promoted": False, "risk_certified": False,
    })
    print({"step_A_results_saved": True, "seconds": round(time.perf_counter() - started, 2)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "calculate"))
    args = parser.parse_args()
    (freeze if args.phase == "freeze" else calculate)()
