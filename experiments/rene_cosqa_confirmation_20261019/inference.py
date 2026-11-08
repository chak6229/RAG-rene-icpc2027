from __future__ import annotations

import joblib
from collections import Counter, defaultdict

from threadpoolctl import threadpool_limits

from .common import (OUT, RETRIEVERS, check_seal, deny_relevance_reads, read, readl, require, seal,
                     status, verify_design, write, writel)
from .cache import features, generate, rank, score_ce
from .policy import all_decisions, group_items


def load_items(retriever, query_ids):
    model = joblib.load(OUT / f"models/{retriever}.joblib")
    rows = readl(OUT / f"features/{retriever}.jsonl")
    require(all(set(r) == {"query_id", "position", "entr_rank", "values", "structural_zero"} for r in rows), "unexpected feature fields")
    actions = {(r["query_id"], r["position"]): r for r in readl(OUT / f"rankings/{retriever}.jsonl")}
    items = group_items(query_ids, rows, actions, model["gain"])
    return model, items


def decide():
    p = verify_design()
    check_seal("POLICY_SEALED.json")
    require(not (OUT / "OUTCOME_OPENING.json").exists(), "no new decisions after H outcome opening")
    if (OUT / "DECISIONS_SEALED.json").exists():
        check_seal("DECISIONS_SEALED.json")
        return
    models, items = {}, {}
    for r in RETRIEVERS:
        models[r], items[r] = load_items(r, p["H_query_ids"])
    ce = score_ce(items, OUT / "ce_scores.jsonl")
    lengths = {(r["query_id"], r["doc_id"]): r["input_tokens"] for r in readl(OUT / "ce_scores.jsonl")}
    allocation_ledger = {}
    with threadpool_limits(limits=4):
        for r in RETRIEVERS:
            outputs, allocated = all_decisions(items[r], models[r]["verification"], models[r]["risk"])
            for row in outputs:
                row["actual_input_tokens"] = sum(lengths[row["query_id"], d] for d in row.get("verification_doc_ids", []))
            require(Counter(v["policy"] for v in outputs) == Counter({name: 8530 for name in
                    ("NoRewrite", "ENTR_top1", "PROB_in_ENTR3", "HGB_gain_control", "RIDGE_gain", "RIDGE_verified", "RIDGE_risk")}), "stage denominator mismatch")
            writel(OUT / f"decisions/{r}.jsonl", outputs)
            allocation_ledger[r] = {"verified_queries": len(allocated), "query_code_pairs_charged": 8 * len(allocated),
                                     "tokens_charged": sum(v["actual_input_tokens"] for v in outputs if v["policy"] == "RIDGE_verified"),
                                     "H_labels_read": False}
            print({"H_decisions_sealed": r, "queries": 8530, "verified": len(allocated)}, flush=True)
    write(OUT / "INFERENCE_LEDGER.json", {"physical_CE": ce, "per_retriever": allocation_ledger,
                                          "H_labels_read": False, "model_or_threshold_fitted_on_H": False,
                                          "shared_CE_for_verified_and_risk": True}, immutable=True)
    paths = [OUT / name for name in ("candidates.jsonl", "selections.jsonl", "CANDIDATES_SEALED.json", "ce_scores.jsonl", "INFERENCE_LEDGER.json")]
    for r in RETRIEVERS:
        paths += [OUT / f"rankings/{r}.jsonl", OUT / f"rankings/{r}.manifest.json",
                  OUT / f"features/{r}.jsonl", OUT / f"features/{r}.manifest.json", OUT / f"decisions/{r}.jsonl"]
    seal("DECISIONS_SEALED.json", paths, H_outcomes_opened=False, complete_policy_frozen_before_inference=True)


def main():
    verify_design()
    with deny_relevance_reads(), threadpool_limits(limits=4):
        for stage, function in (("H_generation", generate), ("H_CodeBERT", lambda: rank("CodeBERT")),
                                ("H_BM25", lambda: rank("BM25")), ("H_features_CodeBERT", lambda: features("CodeBERT")),
                                ("H_features_BM25", lambda: features("BM25")), ("H_adoption", decide)):
            status(stage)
            function()
    status("H_decisions_complete_awaiting_audit")


if __name__ == "__main__":
    main()
