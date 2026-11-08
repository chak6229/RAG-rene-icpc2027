"""Development evidence, same-pool selection, and prospective precision planning."""
from __future__ import annotations

import copy
import math
from collections import Counter

import numpy as np

from experiments.rene_cosqa_pilot_20261019.analyze import power_plan
from experiments.rene_cosqa_pilot_20261019.evaluation import holm
from .common import (INVENTORY, OUT, PILOT, now, read, readl, require, sha, verify_frozen, write)


def size_diagnostics(roles):
    inventory = {r["idx"]: r for r in readl(INVENTORY)}
    sizes = Counter(inventory[q]["group_id"] for q in roles["roles_by_query"])
    result = {}
    for name, key in (("D", "development_group_ids"), ("H", "confirmation_pool_group_ids"), ("official_test_associated_H", "official_test_group_ids")):
        counts = np.asarray([sizes[g] for g in roles[key]], dtype=float)
        result[name] = {"groups": len(counts), "positive_queries": int(counts.sum()),
                        "group_size_quantiles": dict(zip(("min", "median", "p90", "p95", "p99", "max"), np.quantile(counts, [0, .5, .9, .95, .99, 1]).tolist())),
                        "homogeneous_group_mean_variance_design_factor": float(len(counts) * np.square(counts).sum() / counts.sum() ** 2)}
    result["H_to_D_design_factor_ratio"] = result["H"]["homogeneous_group_mean_variance_design_factor"] / result["D"]["homogeneous_group_mean_variance_design_factor"]
    result["interpretation"] = "known size-imbalance sensitivity under homogeneous independent group means; not a bound on unknown H outcome variance"
    return result


def family(metric, keys, section):
    raw = {}
    for key in keys:
        retriever, contrast = key.split(":")
        value = metric[retriever][section][contrast]["p_two_sided_cluster_t"]
        raw[key] = value if value is not None else 1.0
    adjusted = holm(raw)
    return {key: {"p_raw": raw[key], "p_holm": adjusted[key], "development_only": True} for key in raw}


def main():
    protocol = verify_frozen()
    require(read(OUT / "POSTOUTCOME_AUDIT.json")["passed"], "independent postaudit required")
    roles = read(OUT / "ROLE_MANIFEST.json")
    rq1 = {name: read(OUT / f"metrics/{name}_RQ1.json") for name in protocol["retrievers"]}
    rq2 = {name: read(OUT / f"metrics/{name}_RQ2.json") for name in protocol["retrievers"]}
    families = {subset: family({r: rq1[r]["subsets"][subset] for r in rq1}, protocol["main_contrasts"], "paired_contrasts")
                for subset in ("all_development", "original_pilot", "remaining_development")}
    rq2_families = {subset: family({r: rq2[r]["subsets"][subset] for r in rq2}, protocol["rq2_contrasts"], "contrasts")
                    for subset in families}
    write(OUT / "RQ1_DEVELOPMENT.json", {"created_utc": now(), "retrievers": rq1, "four_contrast_holm": families,
                                        "confirmatory": False, "protocol_sha256": sha(OUT / "PROTOCOL.json")}, immutable=True)
    write(OUT / "RQ2_DEVELOPMENT.json", {"created_utc": now(), "retrievers": rq2, "four_contrast_holm": rq2_families,
                                        "confirmatory": False, "same_candidate_pool": True,
                                        "protocol_sha256": sha(OUT / "PROTOCOL.json")}, immutable=True)
    available = len(roles["confirmation_pool_group_ids"])
    plans, joint_plans, rq2_plans = {}, {}, {}
    joint_protocol = copy.deepcopy(protocol)
    joint_protocol["power_plan"]["family_size"] = 8
    for retriever in protocol["retrievers"]:
        rows = readl(OUT / f"metrics/{retriever}_per_query.jsonl")
        for control in ("RAND", "PROB"):
            values = np.asarray([r["policies"]["ENTR"]["best3_rr"] - r["policies"][control]["best3_rr"] for r in rows])
            groups = [r["group_id"] for r in rows]
            key = f"{retriever}:ENTR_minus_{control}_best3"
            plans[key] = power_plan(values, groups, protocol, available)
            joint_plans[key] = power_plan(values, groups, joint_protocol, available)
        rows = readl(OUT / f"metrics/{retriever}_RQ2_per_query.jsonl")
        for contrast, method, control in (
            ("GAIN_HGB_forced_minus_ENTR_top1", "GAIN_HGB_forced", "ENTR_top1"),
            ("GAIN_HGB_selective_minus_NoRewrite", "GAIN_HGB_selective", "NoRewrite"),
        ):
            key = f"{retriever}:{contrast}"
            values = np.asarray([r["policies"][method]["rr"] - r["policies"][control]["rr"] for r in rows])
            groups = [r["group_id"] for r in rows]
            rq2_plans[key] = power_plan(values, groups, joint_protocol, available)
            rq2_plans[key]["extra_assumption"] = "OOF pilot variance conditional on fold models; future all-D model can differ"
            joint_plans[key] = rq2_plans[key]
    size_info = size_diagnostics(roles)
    size_factor = max(1.0, size_info["H_to_D_design_factor_ratio"])
    primary = []
    for key, plan in plans.items():
        row = next(r for r in plan["planned_rows"] if r["target_effect"] == .005 and r["power"] == .8)
        row = {"contrast": key, "planning_eligible": plan["planning_eligible"], **row}
        row["size_imbalance_sensitivity_required_groups"] = math.ceil(row["conservative_required_groups"] * size_factor) if row["conservative_required_groups"] else None
        row["size_sensitivity_capacity_sufficient"] = bool(row["size_imbalance_sensitivity_required_groups"] and row["size_imbalance_sensitivity_required_groups"] <= available)
        primary.append(row)
    supported = all(r["planning_eligible"] and r["available_group_capacity_sufficient"] for r in primary)
    power = {"created_utc": now(), "previous_pilot_power_plan_sha256": sha(PILOT / "POWER_PLAN.json"),
             "available_confirmation_groups": available, "confirmation_queries": len(roles["confirmation_pool_query_ids"]),
             "development_groups": 512, "development_queries": 1490, "target_effect": .005, "target_power": .8,
             "RQ1_four_comparison_plans": plans, "primary_planning_rows": primary,
             "joint_eight_comparison_sensitivity_plans": joint_plans,
             "D_H_group_size_diagnostics": size_info,
             "all_RQ1_primary_normal_plans_supported": supported,
             "normal_approximation_suggested_RQ1_groups": max(r["conservative_required_groups"] for r in primary) if supported else None,
             "confirmation_sample_size_frozen": False, "confirmation_started": False,
             "reason_not_to_start_confirmation": "complete policy and joint multiplicity still need a separate freeze; inspect size transport and precision first",
             "risk_certified": False}
    write(OUT / "POWER_PLAN.json", power, immutable=True)
    candidates = read(OUT / "CANDIDATES_SEALED.json")
    audit = read(OUT / "POSTOUTCOME_AUDIT.json")
    report = ["# CoSQA 全开发集与同候选直接收益选择", "",
              f"开发集：1,490 条正查询、512 个关联组；新增 1,234 条 / 426 组。候选共 {candidates['candidates']:,}，其中 {candidates['reused_candidates']:,} 个来自原封存试跑。",
              "确认池 8,530 条 / 1,963 组保持未运行。所有结果为开发证据；原试跑与新增组分别保留。", "",
              "## RQ1：候选集质量与自动 Top1", "",
              "| 子集 | 检索器 | 原查询 MRR | ENTR best3 Δ | PROB best3 Δ | RAND20 best3 Δ | ENTR Top1 Δ |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for subset, label in (("all_development", "全部开发"), ("remaining_development", "新增开发"), ("original_pilot", "原试跑")):
        for name in protocol["retrievers"]:
            m = rq1[name]["subsets"][subset]
            b = m["candidate_set_quality"]
            report.append(f'| {label} | {name} | {m["NoRewrite_mrr"]:.8f} | {b["ENTR"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {b["PROB"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {b["RAND"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {m["automatic_top1"]["ENTR"]["delta_vs_NoRewrite"]["mean"]:+.8f} |')
    report += ["", "best3 使用真实相关性标签选最好候选，是候选质量上限；自动收益由下一表的实际决策给出。", "",
               "| 全开发主比较 | 差值 | 95% 组区间 | 四比较同时 98.75% 区间 | Holm p | 变化组 |",
               "|---|---:|---|---|---:|---:|"]
    for key in protocol["main_contrasts"]:
        name, contrast = key.split(":")
        m = rq1[name]["subsets"]["all_development"]["paired_contrasts"][contrast]
        report.append(f'| {key} | {m["mean"]:+.8f} | {m["ci95_cluster"]} | {m["ci98_75_four_contrast_family"]} | {families["all_development"][key]["p_holm"]:.6g} | {m["changed_groups"]} |')
    report += ["", "## RQ2：固定 ENTR 三候选池的自动决策", "",
               f"预先固定 {len(protocol['direct_gain']['feature_names'])} 项特征、五折关联组隔离、模型参数与零收益采纳阈值。GEN_HGB 为生成/文本特征对照，GAIN_RIDGE 与主基线 GAIN_HGB 使用检索证据；全部复用现有缓存。",
               "forced 在同一池内选择候选；selective 把原查询作为已知零收益动作。采用按完整 Top100 排名是否改变计算。", "",
               "| 检索器 | 决策 | ΔMRR | 95% 组区间 | 实际采用 | 有害 | 采用内有害率 |",
               "|---|---|---:|---|---:|---:|---:|"]
    for name in protocol["retrievers"]:
        policies = rq2[name]["subsets"]["all_development"]["policies"]
        for policy in ("ENTR_top1", "PROB_in_ENTR3", "GEN_HGB_forced", "GEN_HGB_selective", "GAIN_RIDGE_forced", "GAIN_RIDGE_selective", "GAIN_HGB_forced", "GAIN_HGB_selective"):
            m = policies[policy]
            harm = f'{100*m["harmful_within_adopted"]:.2f}%' if m["harmful_within_adopted"] is not None else "—"
            report.append(f'| {name} | {policy} | {m["delta"]["mean"]:+.8f} | {m["delta"]["ci95_cluster"]} | {m["adopted"]}/1490 | {m["harmful"]} | {harm} |')
    report += ["", "| 主选择比较 | 配对增量 | 95% 组区间 | 四比较 Holm p |", "|---|---:|---|---:|"]
    for key in protocol["rq2_contrasts"]:
        name, contrast = key.split(":")
        m = rq2[name]["subsets"]["all_development"]["contrasts"][contrast]
        report.append(f'| {key} | {m["mean"]:+.8f} | {m["ci95_cluster"]} | {rq2_families["all_development"][key]["p_holm"]:.6g} |')
    report += ["", "OOF 区间描述固定外折预测的组间变化，未包含完整重训的不确定性。确认时需冻结全开发拟合的单一政策。此轮无风险门或认证。", "",
               "## 确认功效规划", "", "目标差 .005、80% 功效；下表沿用 RQ1 四比较方案。另存 RQ1+RQ2 八比较敏感性。", "",
               "| RQ1 对比 | 原 86 组规划 | 512 组保守规划 | 变化组 | 1,963 组容量够 | 组大小敏感性规划 |",
               "|---|---:|---:|---:|---|---:|"]
    old_power = read(PILOT / "POWER_PLAN.json")
    for row in primary:
        old = next(r for r in old_power["primary_planning_rows"] if r["contrast"] == row["contrast"])
        report.append(f'| {row["contrast"]} | {old["conservative_required_groups"]} | {row["conservative_required_groups"]} | {plans[row["contrast"]]["changed_groups"]} | {row["available_group_capacity_sufficient"]} | {row["size_imbalance_sensitivity_required_groups"]} |')
    report += ["", f'D/H 正查询组大小中位数为 {size_info["D"]["group_size_quantiles"]["median"]:g}/{size_info["H"]["group_size_quantiles"]["median"]:g}，最大为 {size_info["D"]["group_size_quantiles"]["max"]:g}/{size_info["H"]["group_size_quantiles"]["max"]:g}。组均值方差同质假设下，H/D 设计因子比为 {size_info["H_to_D_design_factor_ratio"]:.4f}；这是规划敏感性，不是已知确认方差。',
               "确认规模、联合多重性和完整采纳政策尚未冻结，不能用上述开发点估计作为已确认结论。", "",
               "## 验收与复现", "",
               f'独立排名/全词表 FP64 熵复算通过；{audit["exact_rr_fields_checked"]:,} 个 RR 字段有理数复算，最大误差 {audit["metric_max_error"]:.3g}。30 个外折模型重载与全部预测复算误差 {audit["prediction_max_error"]:.3g}。',
               "原 256 条候选、排名与每查询结果完整保留；语料向量未重算，无新增 T5 seed/mask。五个正式指针与模型权重不变，risk_certified=false，部署仍为 NoRewrite。",
               "本轮是源码主导重建：位置1…n、greedy19、冻结本地权重；原论文的0…n/采样m10等差异继续按原协议披露。单已知正目标与已知上游代码暴露沿用数据审查范围。", "",
               "主要证据：PROTOCOL.json、DECISIONS_SEALED.json、PREOUTCOME_AUDIT.json、POSTOUTCOME_AUDIT.json、RQ1_DEVELOPMENT.json、RQ2_DEVELOPMENT.json、POWER_PLAN.json。", ""]
    with (OUT / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    write(OUT / "FINAL_ACCEPTANCE.json", {"passed": True, "created_utc": now(), "scope": "development extension and same-pool OOF baselines",
                                         "protocol_sha256": sha(OUT / "PROTOCOL.json"), "D_queries": 1490, "D_groups": 512,
                                         "preoutcome_audit_passed": True, "postoutcome_audit_passed": True,
                                         "confirmation_opened": False, "confirmation_sample_size_frozen": False,
                                         "risk_certified": False, "protected_pointers_unchanged": True,
                                         "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": sha(p)} for p in
                                                   [OUT / n for n in ("REPORT.md", "RQ1_DEVELOPMENT.json", "RQ2_DEVELOPMENT.json", "POWER_PLAN.json", "PREOUTCOME_AUDIT.json", "POSTOUTCOME_AUDIT.json")]]}, immutable=True)
    print({"development_complete": True, "normal_RQ1_plans_supported": supported, "confirmation_opened": False}, flush=True)


if __name__ == "__main__":
    main()
