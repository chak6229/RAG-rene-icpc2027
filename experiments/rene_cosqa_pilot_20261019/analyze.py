"""Aggregate the prespecified RQ1 contrasts and plan confirmation precision."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import norm

from .common import OUT, now, read, readl, sha, verify_frozen, write
from .evaluation import holm


def power_plan(values, groups, protocol, available):
    labels = sorted(set(groups))
    index = {group: i for i, group in enumerate(labels)}
    codes = [index[group] for group in groups]
    sums = np.bincount(codes, weights=values, minlength=len(labels))
    sizes = np.bincount(codes, minlength=len(labels)).astype(float)
    mean = float(np.mean(values))
    group_count = len(labels)
    changed = len({group for value, group in zip(values, groups) if abs(value) > 1e-10})
    variance_per_group = group_count ** 2 / (group_count - 1) * float(np.square(sums - mean * sizes).sum()) / float(sizes.sum() ** 2)
    rng = np.random.default_rng(protocol["statistics_seed"] + 1)
    draws = rng.integers(0, group_count, (protocol["power_plan"]["bootstrap_variance_repetitions"], group_count))
    sampled_sums, sampled_sizes = sums[draws], sizes[draws]
    sampled_means = sampled_sums.sum(1) / sampled_sizes.sum(1)
    variance_replicates = group_count ** 2 / (group_count - 1) * np.square(sampled_sums - sampled_means[:, None] * sampled_sizes).sum(1) / sampled_sizes.sum(1) ** 2
    conservative_variance = float(np.quantile(variance_replicates, .95))
    eligible = changed >= protocol["power_plan"]["minimum_changed_groups_for_variance_plan"] and conservative_variance > 0
    z_alpha = float(norm.isf(protocol["power_plan"]["alpha_family"] / protocol["power_plan"]["family_size"] / 2))
    rows = []
    for effect in protocol["power_plan"]["effect_targets"]:
        for power in protocol["power_plan"]["power_targets"]:
            multiplier = (z_alpha + float(norm.ppf(power))) ** 2
            estimated = max(30, math.ceil(multiplier * variance_per_group / effect ** 2)) if variance_per_group > 0 else None
            conservative = max(30, math.ceil(multiplier * conservative_variance / effect ** 2)) if conservative_variance > 0 else None
            rows.append({"target_effect": effect, "power": power, "estimated_required_groups": estimated,
                         "conservative_required_groups": conservative,
                         "available_group_capacity_sufficient": conservative <= available if conservative is not None else None,
                         "authorized_sample_size": conservative if eligible and conservative is not None and conservative <= available else None})
    return {"pilot_groups": group_count, "changed_groups": changed, "variance_per_group": variance_per_group,
            "conservative_variance_per_group_95": conservative_variance, "planning_eligible": eligible,
            "eligibility_note": "variance-based planning, no confirmation inference" if eligible else "too few independent changes; more development needed before binding confirmation size",
            "available_confirmation_groups": available,
            "mde_at_available_groups_80pct": (z_alpha + float(norm.ppf(.8))) * math.sqrt(conservative_variance / available) if conservative_variance > 0 else None,
            "planned_rows": rows,
            "assumptions": "independent representative groups and comparable group-size/effect distribution; no finite population correction; normal planning approximation"}


def main():
    protocol = verify_frozen()
    if not read(OUT / "POSTOUTCOME_AUDIT.json")["passed"]:
        raise ValueError("independent metric replay required before analysis")
    metrics = {name: read(OUT / f"metrics/{name}.json") for name in protocol["retrievers"]}
    pvalues = {}
    for key in protocol["main_contrasts"]:
        retriever, contrast = key.split(":")
        pvalue = metrics[retriever]["paired_contrasts"][contrast]["p_two_sided_cluster_t"]
        pvalues[key] = pvalue if pvalue is not None else 1.0
    adjusted = holm(pvalues)
    family = {key: {"p_raw": pvalues[key], "p_holm": adjusted[key], "pilot_only": True} for key in pvalues}
    roles = read(OUT / "ROLE_MANIFEST.json")
    available = len(roles["confirmation_pool_group_ids"])
    plans = {}
    for retriever in protocol["retrievers"]:
        rows = readl(OUT / f"metrics/{retriever}_per_query.jsonl")
        for control in ("RAND", "PROB"):
            values = np.array([row["policies"]["ENTR"]["best3_rr"] - row["policies"][control]["best3_rr"] for row in rows])
            groups = [row["group_id"] for row in rows]
            key = f"{retriever}:ENTR_minus_{control}_best3"
            plans[key] = power_plan(values, groups, protocol, available)
    write(OUT / "RQ1_PILOT.json", {"created_utc": now(), "role": "development", "confirmatory": False,
                                   "protocol_sha256": sha(OUT / "PROTOCOL.json"), "metrics": metrics,
                                   "four_contrast_family_holm": family}, immutable=True)
    primary_rows = []
    for key, plan in plans.items():
        candidate = next(row for row in plan["planned_rows"] if row["target_effect"] == .005 and row["power"] == .8)
        primary_rows.append({"contrast": key, "planning_eligible": plan["planning_eligible"], **candidate})
    eligible_all = all(row["planning_eligible"] and row["available_group_capacity_sufficient"] for row in primary_rows)
    power = {"created_utc": now(), "available_confirmation_groups": available,
             "confirmation_pool_queries": len(roles["confirmation_pool_query_ids"]), "planning_effect": .005,
             "target_power": .8, "plans": plans, "primary_planning_rows": primary_rows,
             "all_primary_plans_supported": eligible_all,
             "proposed_confirmation_groups": max(row["authorized_sample_size"] for row in primary_rows) if eligible_all else None,
             "confirmation_started": False, "sample_size_frozen": False,
             "next": "bind confirmation size and full adoption policy before any confirmation effects" if eligible_all else "expand development precision or revisit confirmation breadth before opening reserved pool"}
    write(OUT / "POWER_PLAN.json", power, immutable=True)
    report = ["# CoSQA 开发小样本：SSQR 位置评分双检索器试跑", "", f'完成时间：{now()}。开发查询 {protocol["queries"]}，连通组 {protocol["groups"]}，语料6,267文档。',
              "本批永久为开发数据；确认池未运行方法效果。主设置、角色、四主比较均在生成前冻结。", "",
              "| 检索器 | NoRewrite MRR@100 | ENTR best3 Δ | PROB best3 Δ | RAND20均值 best3 Δ | ENTR Top1 Δ | PROB Top1 Δ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for retriever, metric in metrics.items():
        best = metric["candidate_set_quality"]
        top = metric["automatic_top1"]
        report.append(f'| {retriever} | {metric["NoRewrite_mrr"]:.8f} | {best["ENTR"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {best["PROB"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {best["RAND"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {top["ENTR"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {top["PROB"]["delta_vs_NoRewrite"]["mean"]:+.8f} |')
    report += ["", "best3 是使用相关性标签得到的候选集上限，不能当成自动采纳收益。Top1 按显式 selected_rank 选择；RAND 从同一位置池做20次缓存抽样。", "",
               "| 主比较 | 配对 ΔMRR | 95%组bootstrap区间 | 四比较同时98.75%区间 | Holm p（开发诊断） | 变化组 |", "|---|---:|---|---|---:|---:|"]
    for key in protocol["main_contrasts"]:
        retriever, contrast = key.split(":")
        item = metrics[retriever]["paired_contrasts"][contrast]
        report.append(f'| {key} | {item["mean"]:+.8f} | {item["ci95_cluster"]} | {item["ci98_75_four_contrast_family"]} | {family[key]["p_holm"]:.6g} | {item["changed_groups"]} |')
    report += ["", "冻结的是源码主导重建：1…n插入位置、显式greedy、19新token、局部T5 best92000；论文0…n、采样m10、原checkpoint及未公开PROB实现的差异已在PROTOCOL列明，不称精确恢复原作者运行。", "",
               "评估器31项有效/反例测试通过；先封存动作与排名，开结果前独立复算全部排名、BM25公式及真实候选FP64熵；开结果后Fraction RR/MRR独立复算通过。所有方法分母相同。", "",
               f'预留确认池：{len(roles["confirmation_pool_query_ids"])}查询/{available}连通组，其中官方test500的396关联组完整保留。已知代码与仓库不跨角色；未知仓库身份仍未知。', "",
               "| 功效规划比较（目标Δ=.005、80%、四比较校正） | 保守所需组 | 可用组容量够 | 变化事件足够估方差 |", "|---|---:|---|---|"]
    for item in primary_rows:
        report.append(f'| {item["contrast"]} | {item["conservative_required_groups"]} | {item["available_group_capacity_sufficient"]} | {item["planning_eligible"]} |')
    report += ["", "功效是基于开发组分布的规划近似；不足20变化组时不据此冻结确认规模。全零/稀疏结果不直接宣称等效。未执行确认、采纳模型训练或风险认证，正式部署保持原状态。", "",
               "证据：PROTOCOL.json、ROLE_MANIFEST.json、TEST_RESULTS.json、CANDIDATES_SEALED.json、DECISIONS_SEALED.json、PREOUTCOME_AUDIT.json、POSTOUTCOME_AUDIT.json、RQ1_PILOT.json、POWER_PLAN.json。", ""]
    with (OUT / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    print({"analysis_complete": True, "confirmation_started": False, "all_power_plans_supported": eligible_all}, flush=True)


if __name__ == "__main__":
    main()
