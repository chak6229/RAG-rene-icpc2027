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
    report = ['# CoSQA Full development and same candidate direct benefit selection', "",
              f"Developments: 1,490 Articles enquiring, 512 associated groups; Add a new 1,234/426 group. Candidates {candidates['candidates']:,}, where {candidates['reused_candidates']:,} A test run from the original seal.",
              'Confirms that the pool 8,530/1,963 group remains inactive. All results are evidence of development; Keep the original run and the new group separately.', "",
              '## RQ1: Candidate set quality and automatic Top1', "",
              '| Subset | Retrieves | original queries MRR  |  ENTR  best3 Δ |  PROB  best3 Δ | RAND20 best3 Δ |  ENTR  Top1 Δ |',
              "|---|---|---:|---:|---:|---:|---:|"]
    for subset, label in (("all_development", 'Develop All'), ("remaining_development", 'Add Development'), ("original_pilot", 'Original Run')):
        for name in protocol["retrievers"]:
            m = rq1[name]["subsets"][subset]
            b = m["candidate_set_quality"]
            report.append(f'| {label} | {name} | {m["NoRewrite_mrr"]:.8f} | {b["ENTR"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {b["PROB"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {b["RAND"]["delta_vs_NoRewrite"]["mean"]:+.8f} | {m["automatic_top1"]["ENTR"]["delta_vs_NoRewrite"]["mean"]:+.8f} |')
    report += ["", 'best3 Select the best candidate using a true relevance label, which is the ceiling; automatic benefits are given by actual decision-making in the next table.', "",
               '| Total developer comparison 95% ♪ In the middle of a group ♪ ♪ In the same time ♪ 98.75% Intersectional | Holm p | Change Group',
               "|---|---:|---|---|---:|---:|"]
    for key in protocol["main_contrasts"]:
        name, contrast = key.split(":")
        m = rq1[name]["subsets"]["all_development"]["paired_contrasts"][contrast]
        report.append(f'| {key} | {m["mean"]:+.8f} | {m["ci95_cluster"]} | {m["ci98_75_four_contrast_family"]} | {families["all_development"][key]["p_holm"]:.6g} | {m["changed_groups"]} |')
    report += ["", '## RQ2: Fixed ENTR Auto-decision of the Triple Pool', "",
               f"Prefix {len(protocol['direct_gain']['feature_names'])} Item characteristics, 50% correlation group isolation, model parameters and zero revenue adoption threshold... GEN_HGB For the purposes of generating/text characteristics, GAIN_RIDGE uses search evidence against the main baseline GAIN_HGB; Reuse all existing caches.",
               'forced Selection of candidates in the same pool; selective Use the original query as a known zero-gain action. Calculate using a change in ranking by full Top100.', "",
               'Z Retrieves | Decision making Δ MRR | 95% Inter-group | Actual use of | Harmful | Use of internal harmful rate |',
               "|---|---|---:|---|---:|---:|---:|"]
    for name in protocol["retrievers"]:
        policies = rq2[name]["subsets"]["all_development"]["policies"]
        for policy in ("ENTR_top1", "PROB_in_ENTR3", "GEN_HGB_forced", "GEN_HGB_selective", "GAIN_RIDGE_forced", "GAIN_RIDGE_selective", "GAIN_HGB_forced", "GAIN_HGB_selective"):
            m = policies[policy]
            harm = f'{100*m["harmful_within_adopted"]:.2f}%' if m["harmful_within_adopted"] is not None else "—"
            report.append(f'| {name} | {policy} | {m["delta"]["mean"]:+.8f} | {m["delta"]["ci95_cluster"]} | {m["adopted"]}/1490 | {m["harmful"]} | {harm} |')
    report += ["", '| Master choice compares | pair increments | 95% The group size, four-by-four. Holm pium.', "|---|---:|---|---:|"]
    for key in protocol["rq2_contrasts"]:
        name, contrast = key.split(":")
        m = rq2[name]["subsets"]["all_development"]["contrasts"][contrast]
        report.append(f'| {key} | {m["mean"]:+.8f} | {m["ci95_cluster"]} | {rq2_families["all_development"][key]["p_holm"]:.6g} |')
    report += ["", 'OOF Intersections describe inter-group variations for fixed external discount projections and do not contain full retrainment uncertainties. This will require freezing the single policy to be developed in its entirety. This round is open to door or authentication.', "",
               '## Identification of effectiveness planning', "", 'Poor target... 005, 80% efficacy; The following table follows RQ1 of the four ICPs. Save RQ1+RQ2 Eight is more sensitive.', "",
               '| RQ1 Comparison | 86 Group Planning | 512 Group Conservative Planning | Change Group | 1,963 Group Capacity Sufficient | Group Sensitivity Planning',
               "|---|---:|---:|---:|---|---:|"]
    old_power = read(PILOT / "POWER_PLAN.json")
    for row in primary:
        old = next(r for r in old_power["primary_planning_rows"] if r["contrast"] == row["contrast"])
        report.append(f'| {row["contrast"]} | {old["conservative_required_groups"]} | {row["conservative_required_groups"]} | {plans[row["contrast"]]["changed_groups"]} | {row["available_group_capacity_sufficient"]} | {row["size_imbalance_sensitivity_required_groups"]} |')
    report += ["", f'D/H Query group size medium to {size_info["D"]["group_size_quantiles"]["median"]:g}/{size_info["H"]["group_size_quantiles"]["median"]:g}, Max {size_info["D"]["group_size_quantiles"]["max"]:g}/{size_info["H"]["group_size_quantiles"]["max"]:g}. Under the group average variance equal, H/D design factor ratio {size_info["H_to_D_design_factor_ratio"]:.4f}; this is planning sensitivity, not known confirmation variance.',
               'The determination of size, joint multiplicity and complete adoption policy has not yet been frozen and the above development point estimates cannot be used as a confirmed conclusion.', "",
               '## Receiving and Inspection and Restatement', "",
               f'Independent ranking/full wordsheet FP64 recalculated pass;{audit["exact_rr_fields_checked"]:,} RR Fields are rationalized, with maximum error {audit["metric_max_error"]:.3g}. 30 Reloading of a folding model with all predictions due to error {audit["prediction_max_error"]:.3g}。',
               'The original 256 list of candidates, rankings and the results of each query are kept in full; Language vector is not recalculated and T5 seed/mask is not added. Five formal points and model weights are unchanged, risk_certified = false, with deployment remaining NoRewrite.',
               'This round is a source code-led reconstruction: position 1 n, greedy19, freezing local weights; Discrepancies such as 0... n/sampling m10 of the original paper continue to be disclosed according to the original agreement. Only known positive targets and known upstream code exposures follow the data review range.', "",
               'Main evidence: PROTOCOL.json, DECISIONS_SEALED.json, PREOUTCOME_AUDIT.json, POSTOUTCOME_AUDIT.json, RQ1_DEVELOPMENT.json, RQ2_DEVELOPMENT.json, POWER_PLAN.json.', ""]
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
