"""Frozen pre-outcome supplement; no change to the original primary policy/test."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.rene_cosqa_pilot_20261019.evaluation import rr
from .common import (OUT, PARENT, RETRIEVERS, ROOT, check_seal, deny_relevance_reads, now,
                     read, readl, record, require, sha, verify_design, write, writel)
from .control_model import NoCEPredictor
from .inference import load_items
from .policy import allocation, group_items, regression, trace, witness_matrix
from .statistics import interval

HERE = Path(__file__).parent


def freeze():
    verify_design()
    require(not (OUT / "OUTCOME_OPENING.json").exists(), "cannot add a control after H opening")
    require(not (OUT / "MATCHED_CONTROL_PROTOCOL.json").exists(), "supplement already frozen")
    role = read(OUT / "STAGE_ROLES.json")["queries"]["verification_fit"]
    positives = defaultdict(set)
    for r in readl(PARENT / "data/private/development_qrels.jsonl"):
        if r["query_id"] in set(role) and r.get("relevance", 1) > 0:
            positives[r["query_id"]].add(r["doc_id"])
    ce = defaultdict(dict)
    for r in readl(OUT / "development/ce_scores.jsonl"):
        ce[r["query_id"]][r["doc_id"]] = r["logit"]
    records = {}
    for ret in RETRIEVERS:
        original_model = joblib.load(OUT / f"models/{ret}.joblib")
        rows = [r for r in readl(PARENT / f"features/{ret}.jsonl") if r["query_id"] in set(role)]
        actions = {(r["query_id"], r["position"]): r for r in readl(PARENT / f"rankings/{ret}.jsonl") if r["query_id"] in set(role)}
        items = group_items(role, rows, actions, original_model["gain"])
        xx, yy, weights, qids = [], [], [], []
        for q, item in items.items():
            if item["plan"] is None:
                continue
            item["ce"] = ce[q]
            matrix = witness_matrix(item["original"], item["rankings"], item["rows"], item["mu"], item["plan"], item["ce"])
            base = rr(item["original"], positives[q], 100)
            labels = np.asarray([rr(ranking, positives[q], 100) - base for ranking in item["rankings"]])
            xx.extend(matrix[:, :73])
            yy.extend(labels - item["mu"])
            weights.extend([1. / len(matrix)] * len(matrix))
            qids.append(q)
        require(len(yy) == original_model["verification"]["training_rows"], "matched training action count differs")
        require(sorted(qids) == original_model["verification"]["fit_query_ids"], "matched training query IDs differ")
        weights = np.asarray(weights) / np.mean(weights)
        control = {"model": NoCEPredictor(regression(np.asarray(xx), np.asarray(yy), weights)),
                   "fit_query_ids": sorted(qids), "training_rows": len(yy), "feature_columns": 73}
        destination = OUT / f"models/{ret}_no_CE_control.joblib"
        require(not destination.exists(), "control model overwrite forbidden")
        joblib.dump(control, destination)
        # Check real model invariance to arbitrary replacement of the omitted CE columns.
        real = np.c_[np.asarray(xx[:20]), np.zeros((min(20, len(xx)), 18))]
        altered = real.copy()
        altered[:, 73:] = np.arange(18)[None, :] * 1e6
        difference = float(np.max(np.abs(control["model"].predict(real) - control["model"].predict(altered))))
        require(difference == 0., "noCE control uses omitted CE columns")
        records[ret] = {"training_queries": len(qids), "training_rows": len(yy), "same_supervision_as_full_CE_update": True,
                        "CE_column_poisoning_prediction_difference": difference,
                        "model": record(destination)}
    p = {"status": "FROZEN", "created_utc": now(), "supplement_to_design_sha256": sha(OUT / "DESIGN.json"),
         "supplement_to_policy_sha256": sha(OUT / "PROTOCOL.json"),
         "reason": "separate additional B-role gain supervision/calibration from actual CE relevance evidence",
         "primary_comparison_unchanged": True, "single_final_policy_unchanged": True,
         "H_outcomes_opened": False, "H_effects_used_to_choose_supplement": False,
         "source_files": [record(HERE / name) for name in ("auxiliary_control.py", "control_model.py")],
         "controls": records, "additional_T5_calls": 0, "additional_CE_pairs": 0,
         "comparison": "RIDGE_verified minus matched_noCE_update on the same frozen allocation, ENTR pool, labels and residual coefficient .5; per retriever, allH",
         "analysis_role": "predefined descriptive paired estimation only; no new confirmatory null rejection",
         "cost_scope": "allocated actual CE cost held equal for the evidence ablation; noCE control can run without CE if deployed alone",
         "stage_interval_scope": "all auxiliary stage intervals condition on sealed H batch allocation/actions; not a guarantee for a repeatedly reallocated25%-budget policy",
         "risk_certificate_issued": False}
    write(OUT / "MATCHED_CONTROL_PROTOCOL.json", p, immutable=True)
    print({"matched_control_frozen_before_H_outcomes": True, "training": records}, flush=True)


def verify():
    p = read(OUT / "MATCHED_CONTROL_PROTOCOL.json")
    for row in p["source_files"] + [v["model"] for v in p["controls"].values()]:
        require(sha(ROOT / row["path"]) == row["sha256"], "supplement hash drift")
    return p


def decide():
    p = verify()
    require(not (OUT / "OUTCOME_OPENING.json").exists(), "supplement decisions must precede opening")
    check_seal("DECISIONS_SEALED.json")
    design = read(OUT / "DESIGN.json")
    records = {}
    with deny_relevance_reads(), threadpool_limits(limits=4):
        for ret in RETRIEVERS:
            model, items = load_items(ret, design["H_query_ids"])
            control = joblib.load(OUT / f"models/{ret}_no_CE_control.joblib")
            ce = defaultdict(dict)
            lengths = {}
            for row in readl(OUT / "ce_scores.jsonl"):
                ce[row["query_id"]][row["doc_id"]] = row["logit"]
                lengths[row["query_id"], row["doc_id"]] = row["input_tokens"]
            for q, item in items.items():
                item["ce"] = dict(ce[q])
            allocated = allocation(items)
            original = {r["query_id"]: r for r in readl(OUT / f"decisions/{ret}.jsonl") if r["policy"] == "RIDGE_verified"}
            rows = []
            for q, item in items.items():
                t = trace(item, control, q in allocated)
                t["actual_input_tokens"] = sum(lengths[q, d] for d in t["verification_doc_ids"])
                require(t["verified"] == original[q]["verified"] and t["actual_pairs"] == original[q]["actual_pairs"]
                        and t["verification_doc_ids"] == original[q]["verification_doc_ids"], "matched allocation/budget differs")
                rows.append(t)
            destination = OUT / f"matched_control/{ret}_decisions.jsonl"
            writel(destination, rows)
            # Independent reload and exact trace reproduction on every H query, still no relevance reads.
            reloaded = joblib.load(OUT / f"models/{ret}_no_CE_control.joblib")
            for q, expected in zip(design["H_query_ids"], rows):
                current = trace(items[q], reloaded, q in allocated)
                require(current["ranking"] == expected["ranking"] and current["position"] == expected["position"], "matched control reload differs")
            records[ret] = {"decisions": record(destination), "queries": len(rows), "verified": len(allocated),
                            "model_reload_queries": len(rows), "same_physical_CE_evidence": True}
    require(not (OUT / "OUTCOME_OPENING.json").exists(), "supplement finished after outcome opening")
    write(OUT / "MATCHED_CONTROL_DECISIONS_SEALED.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "MATCHED_CONTROL_PROTOCOL.json"),
                                                        "retrievers": records, "H_outcomes_opened": False}, immutable=True)
    print({"matched_control_H_decisions_sealed": True, "H_effects_opened": False}, flush=True)


def evaluate():
    p = verify()
    seal = read(OUT / "MATCHED_CONTROL_DECISIONS_SEALED.json")
    opening = read(OUT / "OUTCOME_OPENING.json")
    require(p["created_utc"] < seal["created_utc"] < opening["opened_utc"], "supplement not frozen and sealed before effects")
    groups = read(OUT / "public/H_groups.json")
    gold = defaultdict(set)
    for row in readl(OUT / "private/H_qrels.jsonl"):
        if row.get("relevance", 1) > 0:
            gold[row["query_id"]].add(row["doc_id"])
    result = {"descriptive_only": True, "H_used_to_fit_or_select": False, "additional_T5_calls": 0, "additional_CE_pairs": 0,
              "same_training_labels_queries_and_update_rule": True, "intervals_condition_on_sealed_batch_allocation": True,
              "models_reloaded_on_H_without_relevance_access": 17060, "retrievers": {}}
    report = ["# 核验额外监督与CE证据匹配对照", "",
              "在H效果开封前追加冻结的辅助分析，不改变原主检验或唯一完整采纳策略。上下文对照与真实CE分支使用完全相同的B角色训练查询/标签、Ridge正则、.5残差更新、H核验分配及候选池，移除18列CE证据。没有增加T5或CE调用。", "",
              "所有辅助阶段区间以本次已封存批次分配/动作为条件，不作为未来重新分配25%预算策略的总体保证。", "",
              "| 检索器 | 原收益选择Δ | 无CE但同监督更新Δ | 真实CE更新Δ | CE独立差 | 配对95%描述区间 |", "|---|---:|---:|---:|---:|---|"]
    for ret in RETRIEVERS:
        full = {r["query_id"]: r for r in readl(OUT / f"metrics/{ret}_per_query.jsonl")}
        rows = readl(OUT / f"matched_control/{ret}_decisions.jsonl")
        difference, gains, labels_groups, verified_flags = [], [], [], []
        records = []
        for row in rows:
            q = row["query_id"]
            value = rr(row["ranking"], gold[q], 100)
            delta = value - full[q]["NoRewrite"]
            actual = full[q]["policies"]["RIDGE_verified"]["delta"]
            difference.append(actual - delta)
            gains.append(delta)
            labels_groups.append(groups[q])
            verified_flags.append(row["verified"])
            records.append({"query_id": q, "group_id": groups[q], "no_CE_delta": delta,
                            "CE_delta": actual, "CE_increment": actual - delta})
        contrast = interval(difference, labels_groups)
        summary = read(OUT / f"metrics/{ret}_full_H.json")
        result["retrievers"][ret] = {"control_delta_MRR": interval(gains, labels_groups), "CE_independent_increment": contrast,
                                      "verified_queries": sum(verified_flags), "training": p["controls"][ret]}
        writel(OUT / f"matched_control/{ret}_per_query.jsonl", records)
        report += [f"| {ret} | {summary['policy_results']['RIDGE_gain']['delta_MRR']['mean']:+.8f} | {np.mean(gains):+.8f} | {summary['policy_results']['RIDGE_verified']['delta_MRR']['mean']:+.8f} | {contrast['mean']:+.8f} | {contrast['ci95_group_bootstrap']} |"]
    write(OUT / "MATCHED_CONTROL_RESULTS.json", result, immutable=True)
    (OUT / "MATCHED_CONTROL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print({"matched_control_evaluated": result["retrievers"]}, flush=True)


def main():
    with threadpool_limits(limits=4):
        freeze()
    while not (OUT / "DECISIONS_SEALED.json").exists():
        require(not (OUT / "OUTCOME_OPENING.json").exists(), "H opened before supplement decisions")
        time.sleep(5)
    decide()
    while not (OUT / "FINAL_ACCEPTANCE.json").exists():
        time.sleep(5)
    evaluate()


if __name__ == "__main__":
    main()
