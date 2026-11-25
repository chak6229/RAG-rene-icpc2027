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
    report = ['# CoSQA Development of small samples: SSQR Position Rating Double Retrieval Test', "", f'_Other Organiser{now()}. Develop queries {protocol["queries"]}Connect team. {protocol["groups"]}, the medium 6,267 document.',
              'This batch is permanently used to develop data; confirm that the pool does not run its method effects. The main settings, roles, and four comparisons are frozen before they are generated.', "",
              '| Retrieval NoRewrite MRR@100 ENTR best3 | PROB best3 Δ RAND20 Mean value best3 Δ ENTR Top1 Δ PROB Top1 Δ', "|---|---:|---:|---:|---:|---:|---:|"]
    for retriever, metric in metrics.items():
        best = metric["candidate_set_quality"]
        top = metric["automatic_top1"]
        report.append(f'| {retriever} | {metric["NoRewrite_mrr"]:.8f} | {best["ENTR"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {best["PROB"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {best["RAND"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {top["ENTR"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {top["PROB"]["delta_vs_NoRewrite"]["mean"]:+.8f} |')
    report += ["", 'best3 is the upper limit of the pool obtained using a relevant label and cannot be considered an automatic adoption gain. Top1 Select by visible selected_rank; RAND A subcache sample of 20 from the same position pool.', "",
               'Z Main comparison | Match MRR 95% Group bootstrap 98.75% Intersections Holm p', "|---|---:|---|---|---:|---:|"]
    for key in protocol["main_contrasts"]:
        retriever, contrast = key.split(":")
        item = metrics[retriever]["paired_contrasts"][contrast]
        report.append(f'| {key} | {item["mean"]:+.8f} | {item["ci95_cluster"]} | {item["ci98_75_four_contrast_family"]} | {family[key]["p_holm"]:.6g} | {item["changed_groups"]} |')
    report += ["", 'Freezing is source code leading reconstruction: 1 n insert position, visible greedy, 19 new token, local T5 best92000; Papers 0...n, sample m10, original checkpoint and undisclosed PROB the differences realized are listed in PROTOCOL and do not mean to accurately resume the operation of the author.', "",
               'Valid/reverse test for evaluation unit 31 passed; Store action and ranking first, independently recalculating all rankings before results, BM25 formulae and real candidates FP64 entropy; After result Fraction RR / MRR Independent recalculation passed. All methods have the same denominator.', "",
               f'Retention confirmation pool:{len(roles["confirmation_pool_query_ids"])}Query{available}Connecting group, where the official test500 associated group 396 is fully maintained. Known codes do not fit in with repositories; The identity of the unknown repository remains unknown.', "",
               'Comparison of effectiveness planning (target =...) 005, 80%,4comparison correction) | Conservative group | Available group capacity enough | Change event enough to estimate the difference |', "|---|---:|---|---|"]
    for item in primary_rows:
        report.append(f'| {item["contrast"]} | {item["conservative_required_groups"]} | {item["available_group_capacity_sufficient"]} | {item["planning_eligible"]} |')
    report += ["", 'Effectiveness is based on the planning approximation of the distribution of development teams; Not enough 20 change group does not freeze the confirmed size accordingly. Full zero/shrimp results do not directly claim equivalence. The validation, adoption of model training or risk certification was not carried out and the formal deployment remained in place.', "",
               'Evidence: PROTOCOL.json, ROLE_MANIFEST.json, TEST_RESULTS.json, CANDIDATES_SEALED.json, DECISIONS_SEALED.json, PREOUTCOME_AUDIT.json, POSTOUTCOME_AUDIT.json, RQ1_PILOT.json, POWER_PLAN.json.', ""]
    with (OUT / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    print({"analysis_complete": True, "confirmation_started": False, "all_power_plans_supported": eligible_all}, flush=True)


if __name__ == "__main__":
    main()
