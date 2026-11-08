from __future__ import annotations

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.rene_cosqa_pilot_20261019.evaluation import rr
from .common import (OUT, PARENT, RETRIEVERS, now, read, readl, require, seal, sha, status,
                     verify_design, write)
from .cache import score_ce
from .policy import (CE_NAMES, RISK_NAMES, allocation, fit_gain, fit_risk, fit_update, group_items,
                     risk_features, trace)


def main():
    p = verify_design()
    require(not (OUT / "POLICY_SEALED.json").exists(), "models already frozen")
    require(not (OUT / "candidates.jsonl").exists(), "fit after H inference is forbidden")
    status("training_D_separate_roles")
    roles = read(OUT / "STAGE_ROLES.json")["queries"]
    positives = {}
    for r in readl(PARENT / "data/private/development_qrels.jsonl"):
        if r.get("relevance", 1) > 0:
            positives.setdefault(r["query_id"], set()).add(r["doc_id"])
    require(set(positives) == set(q for v in roles.values() for q in v), "D labels or stage roles incomplete")
    all_items, gains, ledgers = {}, {}, {}
    with threadpool_limits(limits=4):
        for retriever in RETRIEVERS:
            rows = readl(PARENT / f"features/{retriever}.jsonl")
            actions = {(r["query_id"], r["position"]): r for r in readl(PARENT / f"rankings/{retriever}.jsonl")}
            gain_ids = set(roles["gain_fit"])
            fit_rows = [r for r in rows if r["query_id"] in gain_ids]
            labels = [rr(actions[r["query_id"], r["position"]]["ranked_doc_ids"], positives[r["query_id"]], 100)
                      - rr(actions[r["query_id"], None]["ranked_doc_ids"], positives[r["query_id"]], 100) for r in fit_rows]
            gain_model = fit_gain(fit_rows, labels)
            qids = roles["verification_fit"] + roles["risk_fit"]
            selected = [r for r in rows if r["query_id"] in set(qids)]
            items = group_items(qids, selected, actions, gain_model)
            for q, item in items.items():
                original_rr = rr(item["original"], positives[q], 100)
                item["labels"] = [rr(rank, positives[q], 100) - original_rr for rank in item["rankings"]]
            all_items[retriever], gains[retriever] = items, gain_model
            ledgers[retriever] = {"gain_training_queries": len(gain_ids), "gain_training_actions": len(labels),
                                  "gain_query_ids": sorted(gain_ids)}
        # Scoring superset is D-only training evidence; H charges only its allocated eight-pair plans.
        ce_ledger = score_ce(all_items, OUT / "development/ce_scores.jsonl", all_eligible=True)
        for retriever in RETRIEVERS:
            items = all_items[retriever]
            vfit = [items[q] for q in roles["verification_fit"]]
            update = fit_update(vfit)
            risk_items = {q: items[q] for q in roles["risk_fit"]}
            allocated = allocation(risk_items)
            xx, natives, deltas, fit_qids = [], [], [], []
            for q, item in risk_items.items():
                t = trace(item, update, q in allocated)
                if not t["accepted"]:
                    continue
                x, native = risk_features(item, t)
                xx.append(x)
                natives.append(native)
                deltas.append(item["labels"][t["chosen"]])
                fit_qids.append(q)
            risk = fit_risk(xx, natives, deltas, fit_qids)
            model = {"gain": gains[retriever], "verification": update, "risk": risk,
                     "main_adoption": "RIDGE_gain -> gain-priority eight-pair residual update -> threeway risk<=.075",
                     "feature_names": p["feature_names"], "CE_feature_names": CE_NAMES, "risk_feature_names": RISK_NAMES,
                     "fit_roles": roles, "trained_on_H": False}
            path = OUT / f"models/{retriever}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            require(not path.exists(), "model overwrite forbidden")
            joblib.dump(model, path)
            ledgers[retriever].update(verification_fit_query_ids=update["fit_query_ids"], verification_rows=update["training_rows"],
                                      risk_fit_query_ids=fit_qids, risk_training_actions=len(fit_qids),
                                      risk_training_counts=risk["counts"].tolist(), risk_training_verified=len(allocated),
                                      model_sha256=sha(path))
            print({"training_complete": retriever, "verification_rows": update["training_rows"],
                   "risk_training_actions": len(fit_qids), "H_used": False}, flush=True)
    write(OUT / "TRAINING_LEDGER.json", {"created_utc": now(), "retrievers": ledgers, "ce": ce_ledger,
                                         "roles_disjoint_by_connected_group": True, "D_training_counts_are_not_validation_results": True,
                                         "H_used": False, "threshold_selection": False}, immutable=True)
    final = {"status": "FROZEN", "created_utc": now(), "design_sha256": sha(OUT / "DESIGN.json"),
             "primary_sample_sha256": sha(OUT / "PRIMARY_SAMPLE.json"), "precision_plan_sha256": sha(OUT / "PRECISION_PLAN.json"),
             "single_adoption_strategy": "RIDGE_risk", "primary_retriever": "CodeBERT", "secondary_retriever": "BM25",
             "primary_comparison": p["primary_comparison"], "auxiliary_comparisons": p["auxiliary_comparisons"],
             "models": [{"path": f"models/{r}.joblib", "sha256": sha(OUT / f"models/{r}.joblib")} for r in RETRIEVERS],
             "risk_certified": False, "deployment_policy_promoted": False, "H_predictions_started": False,
             "source_and_protected_hashes_bound_by_design": True}
    write(OUT / "PROTOCOL.json", final, immutable=True)
    seal("POLICY_SEALED.json", [OUT / "PROTOCOL.json", OUT / "TRAINING_LEDGER.json", OUT / "development/ce_scores.jsonl"]
         + [OUT / f"models/{r}.joblib" for r in RETRIEVERS], H_outcomes_opened=False, complete_policy_frozen=True)
    status("complete_policy_frozen", precision_supported=True)


if __name__ == "__main__":
    main()
