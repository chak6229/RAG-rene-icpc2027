"""Freeze, fit forty small linear heads, seal decisions, then evaluate step B."""
from __future__ import annotations

import argparse
import importlib.metadata
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .common import (BLOCKS, CORPUS, DESIGN, FEATURE_NAMES, HEADS, M0, OUT, PACKAGE,
                     PARENT, POLICIES, PROTECTED, R, RETRIEVERS, ROOT, T,
                     load_features, load_outcomes, now, read, readl, record,
                     require, sha, verify, write, writel)
from .metrics import clustered, prediction_quality, utility
from .modeling import decide, fit_fold, predict_saved


def freeze():
    require(not (OUT / "PROTOCOL.json").exists(), "Step B already frozen")
    paths = {PARENT / name for name in (
        "PROTOCOL.json", "FOLDS.json", "ROLE_MANIFEST.json", "FINAL_ACCEPTANCE.json",
        "DECISIONS_SEALED.json", "CANDIDATES_SEALED.json", "selections.jsonl", "candidates.jsonl",
        "data/queries.jsonl", "data/groups.json", "data/private/development_qrels.jsonl",
    )}
    for retriever in RETRIEVERS:
        for kind in ("features", "rankings"):
            paths.update(PARENT / f"{kind}/{retriever}{suffix}" for suffix in (".jsonl", ".manifest.json"))
        manifest = read(PARENT / f"features/{retriever}.manifest.json")
        require(sha(PARENT / f"features/{retriever}.jsonl") == manifest["sha256"], "Original feature seal differs")
        require(manifest["feature_names"] == list(FEATURE_NAMES), "Feature name mismatch")
    parent_seal = read(PARENT / "DECISIONS_SEALED.json")
    require(parent_seal["protocol_sha256"] == sha(PARENT / "PROTOCOL.json"), "Parent protocol seal differs")
    parent_checked = 0
    for item in parent_seal["files"]:
        path = PARENT / item["path"]
        if path in paths:
            require(sha(path) == item["sha256"], "Original decision seal differs")
            parent_checked += 1
    require(read(PARENT / "FINAL_ACCEPTANCE.json")["passed"], "Parent acceptance failed")
    paths.update([CORPUS, DESIGN])
    paths.update(PACKAGE.glob("*.py"))
    for name in ("experiments/rene_cosqa_development_20261019/features.py",
                 "experiments/rene_cosqa_development_20261019/common.py",
                 "experiments/rene_cosqa_pilot_20261019/common.py",
                 "experiments/rene_cosqa_pilot_20261019/ranking.py",
                 "experiments/rene_cosqa_pilot_20261019/evaluation.py"):
        paths.add(ROOT / name)
    write("PROTOCOL.json", {
        "status": "FROZEN_BEFORE_THIS_FIT", "created_utc": now(),
        "role": "consumed_development_OOF_diagnostic", "existing_effects_already_known": True,
        "new_confirmation": False, "external_preregistration": False,
        "scope": "B only, D1490/512; no new T5/generation/retrieval/CE, C or H effects",
        "candidate": "exact existing ENTR Top1 position for every query; no reselection",
        "label": "RR@100(fixed candidate) - RR@100(original), all 1490 training examples, unit weight",
        "folds": "reuse parent 5 connected-group folds, same train/eval rows for all heads",
        "learner": {"name": "StandardScaler + Ridge", "alpha": 1.0, "solver": "svd",
                    "fit_intercept": True, "standardize_training_only": True,
                    "prediction_clipping": False, "tuning": False},
        "feature_schema": list(FEATURE_NAMES),
        "feature_partition": {"M0": M0, "T": T, "R": R},
        "feature_blocks": {h: {"columns": BLOCKS[h], "names": [FEATURE_NAMES[c] for c in BLOCKS[h]]} for h in HEADS},
        "T_interpretation": "single teacher-forcing distribution confidence and ENTR3-relative statistics; not an epistemic/aleatoric decomposition",
        "R_interpretation": "observable original scores/QPP, overlap/rank changes and original-top1 lexical coverage; no relevance/target labels",
        "common_eligibility": "nonidentity text AND full cached Top100 order changes; same label-free gate for every head; both policies use it",
        "eligibility_caveat": "M0/MT learners have no original retrieval feature, but share this observable structural gate with MR/MRT",
        "policies": {
            "positive": "adopt if eligible and raw prediction > 0, no threshold tuning",
            "fixed20": "per held fold, floor(eligible_count/5) highest raw predictions, query-ID ascending tie-break; negative scores still eligible",
        },
        "quota_interpretation": "20% of available ranking-changing nonidentity actions, not all queries; transductive label-free batch-budget diagnostic",
        "primary_comparisons": ["MRT-MT", "MRT-MR"],
        "multiplicity": "family of 8 effects = 2 conditional increments x 2 policies x 2 retrievers; pointwise95 plus Bonferroni99.375 individual intervals",
        "secondary_comparisons": ["MR-M0", "MT-M0", "each policy vs NoRewrite", "prediction error and fixed-score calibration quintiles"],
        "bootstrap": {"repetitions": 10000, "seed": 20261020, "unit": "512 connected groups", "estimand": "query-weighted paired mean"},
        "interval_scope": "descriptive, conditional on fixed OOF predictions; group bootstrap does not refit overlapping training folds or reselect quota; not a new-policy deployment guarantee",
        "risk_gate_in_experiment": False, "risk_certified": False,
        "no_policy_promotion": True, "no_equivalence_margin_or_claim": True,
        "original_sealed_inputs_verified": parent_checked,
        "bound_files": [record(p) for p in sorted(paths)],
        "protected_files": [record(ROOT / name) for name in PROTECTED],
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}},
    })
    print({"step_B_frozen": True, "head_columns": {h: len(BLOCKS[h]) for h in HEADS}, "role": "consumed_D"}, flush=True)


def fit():
    protocol = verify()
    require(not (OUT / "PREDICTIONS_SEALED.json").exists(), "Predictions already sealed")
    started = time.perf_counter()
    files, ledger = [], []
    with threadpool_limits(limits=1):
        for retriever in RETRIEVERS:
            data = load_features(retriever)
            # Global D label access is known/declared; fit_fold slices train before fitting.
            _, _, y = load_outcomes(data)
            predictions = {h: np.full(len(data.qids), np.nan) for h in HEADS}
            for head in HEADS:
                for fold in range(5):
                    train, held = data.folds != fold, data.folds == fold
                    require(not set(data.groups[train]) & set(data.groups[held]), "Training/held groups overlap")
                    model, pred = fit_fold(data.x, y, data.folds, head, fold)
                    model.update({"retriever": retriever, "train_query_ids": list(np.asarray(data.qids)[train]),
                                  "held_query_ids": list(np.asarray(data.qids)[held]),
                                  "train_group_ids": sorted(set(data.groups[train])),
                                  "held_group_ids": sorted(set(data.groups[held]))})
                    error = float(np.max(np.abs(predict_saved(model, data.x[held]) - pred)))
                    require(error < 1e-12, "Saved model prediction mismatch")
                    predictions[head][held] = pred
                    name = f"models/{retriever}/{head}/fold{fold}.json"
                    write(name, model)
                    files.append(OUT / name)
                    ledger.append({"retriever": retriever, "head": head, "held_fold": fold,
                                   "train_queries": int(train.sum()), "held_queries": int(held.sum()),
                                   "train_groups": len(set(data.groups[train])), "held_groups": len(set(data.groups[held])),
                                   "held_labels_in_fit": False, "saved_reload_max_error": error})
            adopted = {policy: {h: decide(predictions[h], data.qids, data.folds, data.eligible, policy) for h in HEADS}
                       for policy in POLICIES}
            rows = []
            for i, q in enumerate(data.qids):
                rows.append({"retriever": retriever, "query_id": q, "group": data.groups[i], "held_fold": int(data.folds[i]),
                             "position": data.positions[i], "eligible": bool(data.eligible[i]),
                             "identity_text": bool(data.identity[i]), "same_ranking": bool(data.same_ranking[i]),
                             "predictions": {h: float(predictions[h][i]) for h in HEADS},
                             "adopted": {p: {h: bool(adopted[p][h][i]) for h in HEADS} for p in POLICIES}})
            name = f"predictions/{retriever}.jsonl"
            writel(name, rows)
            files.append(OUT / name)
            print({"fitted": retriever, "models": 20, "queries": len(rows), "eligible": int(data.eligible.sum())}, flush=True)
    write("TRAINING_LEDGER.json", {"created_utc": now(), "role": protocol["role"],
                                   "model_fits": ledger, "seconds": time.perf_counter() - started,
                                   "D_labels_read": True, "H_outcomes_read": False,
                                   "feature_construction_qrels_read": False,
                                   "all_heads_same_supervision": True, "no_hyperparameter_search": True})
    files.append(OUT / "TRAINING_LEDGER.json")
    write("PREDICTIONS_SEALED.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                                      "role": protocol["role"], "existing_effects_already_known": True,
                                      "before_this_run_aggregate_effect_evaluation": True,
                                      "files": [record(p) for p in files]})


def verify_predictions():
    seal = read(OUT / "PREDICTIONS_SEALED.json")
    require(seal["protocol_sha256"] == sha(OUT / "PROTOCOL.json"), "Prediction protocol mismatch")
    for item in seal["files"]:
        require(sha(ROOT / item["path"]) == item["sha256"], "Sealed prediction/model changed")


def evaluate():
    verify()
    verify_predictions()
    require(not (OUT / "RESULTS.json").exists(), "Results already evaluated")
    views, evaluated = {}, []
    for retriever in RETRIEVERS:
        data = load_features(retriever)
        baseline, candidate, y = load_outcomes(data)
        rows = readl(OUT / f"predictions/{retriever}.jsonl")
        require([r["query_id"] for r in rows] == data.qids, "Prediction order changed")
        vectors, quality, policies = {}, {}, {}
        for h in HEADS:
            pred = np.asarray([r["predictions"][h] for r in rows])
            quality[h] = prediction_quality(y, pred, data.qids)
        for policy in POLICIES:
            policies[policy] = {}
            for h in HEADS:
                adopted = np.asarray([r["adopted"][policy][h] for r in rows])
                policies[policy][h] = utility(y, baseline, adopted, data.folds)
                vectors[f"{policy}:{h}-NoRewrite"] = y * adopted
            for left, right in (("MRT", "MT"), ("MRT", "MR"), ("MR", "M0"), ("MT", "M0")):
                vectors[f"{policy}:{left}-{right}"] = vectors[f"{policy}:{left}-NoRewrite"] - vectors[f"{policy}:{right}-NoRewrite"]
        vectors["ForcedENTRTop1-NoRewrite"] = y
        family = [f"{p}:{c}" for p in POLICIES for c in ("MRT-MT", "MRT-MR")]
        estimates = clustered(np.column_stack(list(vectors.values())), data.groups, list(vectors), family_names=family)
        for policy in POLICIES:
            counts = [policies[policy][h]["adopted"] for h in HEADS]
            if policy == "fixed20":
                require(len(set(counts)) == 1, "Same-count comparison failed")
        for i, row in enumerate(rows):
            evaluated.append({**row, "original_rr": float(baseline[i]), "candidate_rr": float(candidate[i]),
                              "delta_rr": float(y[i]),
                              "policy_gain": {p: {h: float(vectors[f"{p}:{h}-NoRewrite"][i]) for h in HEADS} for p in POLICIES}})
        views[retriever] = {"population": data.metadata,
                            "baselines": {"NoRewrite": utility(y, baseline, np.zeros(len(y), dtype=bool), data.folds),
                                          "ForcedENTRTop1": utility(y, baseline, np.ones(len(y), dtype=bool), data.folds)},
                            "fixed_action_oracle_diagnostic_delta": float(np.maximum(y, 0).mean()),
                            "prediction_quality": quality, "policies": policies, "estimates": estimates}
    write("RESULTS.json", {"created_utc": now(), "role": "consumed_development_OOF_diagnostic",
                            "new_confirmation": False, "risk_certified": False, "views": views,
                            "inference": "Fixed-score descriptive group intervals, not refitted algorithm or conditional-risk coverage guarantees"})
    writel("per_query.jsonl", evaluated)
    report(views)
    verify()
    verify_predictions()
    print({"step_B_evaluated": True, "rows": len(evaluated), "official_configuration_changed": False}, flush=True)


def report(views):
    lines = ["# 第二步B：固定改写动作的生成置信与原查询参照检验", "",
             "本实验只问：同一个ENTR Top1改写是否值得替代原查询，原检索参照R和生成置信T各增加了多少决策价值？",
             "", "CoSQA已消费开发集1490条查询/512关联组，双检索器，沿用5折组隔离OOF。固定候选、标签、学习器和训练折；40个StandardScaler+Ridge(alpha=1, SVD)小模型，无调参。标准化仅拟合训练折，原查询与改写Top100完整排名之间回退。",
             "", "M0为27项共同文本/候选检索/词汇覆盖特征；MT加10项生成置信，MR加34项原查询参照，MRT为71项。token长度属于共同特征；T是单次teacher-forcing概率/熵及ENTR3池内统计，未解耦认知/偶然不确定性。R无qrels、目标ID或真实正确排名。",
             "", "所有头共享可观测的非恒等、Top100排序改变门，M0/MT学习器没有原检索输入，但也使用此共同结构门。固定20%在各评估折内采用floor(可用动作数/5)，同分按query ID，允许负预测。它是批次预算诊断。零阈值则预测ΔRR>0时采用。风险认证未加入本次机制对照。", ""]
    for retriever, view in views.items():
        lines += [f"## {retriever}", "", f"可用动作{view['population']['eligible']}/1490；原文恒等{view['population']['identity_text']}；完整排序相同{view['population']['same_ranking']}。NoRewrite MRR={view['baselines']['NoRewrite']['mrr']:.8f}。强制ENTR Top1 ΔMRR={view['baselines']['ForcedENTRTop1']['delta_mrr']:+.8f}。", "",
                  "|规则|模型|采用数|覆盖率|ΔMRR|描述性组bootstrap95%|有害/采用|每查询损失量|正收益量保留|",
                  "|---|---|---:|---:|---:|---|---:|---:|---:|"]
        for p in POLICIES:
            for h in HEADS:
                u = view["policies"][p][h]
                ci = view["estimates"][f"{p}:{h}-NoRewrite"]["ci95_group_bootstrap"]
                lines.append(f"|{p}|{h}|{u['adopted']}|{u['coverage']:.2%}|{u['delta_mrr']:+.8f}|[{ci[0]:+.8f}, {ci[1]:+.8f}]|{u['harmful']}/{u['adopted']}|{u['harm_mass_per_query']:.8f}|{u['retained_fixed_action_positive_mass']:.2%}|")
        lines += ["", "主条件增量：MRT−MT检验加入原查询参照；MRT−MR检验加入生成置信。八项家族包含两个检索器和两条采用规则。", "",
                  "|规则|比较|ΔMRR差|点式bootstrap95%|八项Bonferroni家族95%区间|",
                  "|---|---|---:|---|---|"]
        for p in POLICIES:
            for c in ("MRT-MT", "MRT-MR"):
                e = view["estimates"][f"{p}:{c}"]
                ci, fc = e["ci95_group_bootstrap"], e["family95_bonferroni_group_bootstrap"]
                lines.append(f"|{p}|{c}|{e['mean']:+.8f}|[{ci[0]:+.8f}, {ci[1]:+.8f}]|[{fc[0]:+.8f}, {fc[1]:+.8f}]|")
        lines += ["", "|预测质量|M0|MT|MR|MRT|", "|---|---:|---:|---:|---:|"]
        for metric in ("mse", "mae", "prediction_spearman", "bias", "mean_absolute_quintile_calibration_gap"):
            lines.append("|" + metric + "|" + "|".join(f"{view['prediction_quality'][h][metric]:.8f}" for h in HEADS) + "|")
        lines += ["", "每折收益、实际有害率/损失幅度、五分位校准明细、次要MR−M0与MT−M0比较均在RESULTS.json；逐查询固定动作、预测、采用与真实RR在per_query.jsonl。", ""]
    lines += ["## 解释范围与验收", "",
              "正的MRT−MT只说明这套固定动作Ridge模型加入可观测原参照后改善所测采用，不证明独特算法或所有检索器泛化。MRT−MR若区间跨零，不能将T写成已验证的额外决策收益，也不能反向宣称全部UQ无效。预测误差改善与实际采用收益分别判定；改善有害Top1不等于胜过NoRewrite。未设置开发集等效阈值。",
              "", "区间对既有OOF预测按关联组重采样，保留query加权与配对；未在每次抽样内重训，也未重排固定预算，故为条件于固定OOF输出的描述性精度，不是学习算法泛化置信区间。开发集已用于提出方案；本次冻结在新拟合前，但不是新独立确认。",
              "", "独立审计结果见AUDIT.json，验收见FINAL_ACCEPTANCE.json：独立Fraction RR、线性代数重拟合、外折标签投毒、独立特征复算/配额/指标区间复放及哈希检查。保持原H确认、正式配置和NoRewrite部署；risk_certified=false。"]
    with (OUT / "READOUT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "fit", "evaluate"))
    args = parser.parse_args()
    {"freeze": freeze, "fit": fit, "evaluate": evaluate}[args.stage]()


if __name__ == "__main__":
    main()
