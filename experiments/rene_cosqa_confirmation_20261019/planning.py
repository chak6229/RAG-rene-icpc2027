"""Outcome-blind H sampling; D variance can choose the query cap, never H effects."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .common import (ADAPTER, INVENTORY, OUT, PARENT, capped_ids, digest, now, read, readl,
                     require, write, writel)
from .statistics import plan


def prepare():
    require(not (OUT / "DESIGN.json").exists(), "already frozen")
    roles = read(PARENT / "ROLE_MANIFEST.json")
    inv = {r["idx"]: r for r in readl(INVENTORY)}
    groups = {q: inv[q]["group_id"] for q in roles["roles_by_query"]}
    d = roles["development_query_ids"]
    h = roles["confirmation_pool_query_ids"]
    require(len(d) == 1490 and len(h) == 8530, "parent roles changed")
    require(not {groups[q] for q in d} & {groups[q] for q in h}, "D/H connected-group overlap")
    metrics = {r["query_id"]: r for r in readl(PARENT / "metrics/CodeBERT_per_query.jsonl")}
    require(set(metrics) == set(d), "planning attempted non-D effects")
    candidates = []
    for cap in (4, 8, 12, 16, 1000000):
        dd, hh = capped_ids(d, groups, cap), capped_ids(h, groups, cap)
        sizes = Counter(groups[q] for q in hh)
        values = np.array([metrics[q]["policies"]["ENTR"]["best3_rr"] - metrics[q]["policies"]["RAND"]["best3_rr"] for q in dd])
        p = plan(values, [groups[q] for q in dd], list(sizes.values()))
        p.update(cap_per_group=cap, H_queries=len(hh),
                 eligible=bool(p["required_groups"] <= .95 * len(sizes) and p["changed_development_groups"] >= 20))
        candidates.append(p)
    feasible = [p for p in candidates if p["eligible"]]
    require(feasible, "precision infeasible; no confirmation authorization")
    chosen = max(feasible, key=lambda r: r["cap_per_group"])
    primary = capped_ids(h, groups, chosen["cap_per_group"])
    require(chosen["planned_power_at_capacity"] >= .8, "80 percent power not supported")
    ordered_d_groups = sorted(roles["development_group_ids"], key=lambda g: digest(g, "rene-cosqa-stage-roles-20261019"))
    stage_groups = {"gain_fit": ordered_d_groups[:307], "verification_fit": ordered_d_groups[307:410],
                    "risk_fit": ordered_d_groups[410:]}
    stage_roles = {name: [q for q in d if groups[q] in set(gs)] for name, gs in stage_groups.items()}
    public = {r["query_id"]: r for r in readl(ADAPTER / "public/all_positive_queries.jsonl")}
    writel(OUT / "public/H_queries.jsonl", [public[q] for q in h])
    write(OUT / "public/H_groups.json", {q: groups[q] for q in h}, immutable=True)
    write(OUT / "PRIMARY_SAMPLE.json", {"query_ids": primary, "group_ids": sorted({groups[q] for q in primary}),
                                         "cap_per_group": chosen["cap_per_group"], "all_H_secondary_query_ids": h,
                                         "selection_without_H_predictions_or_outcomes": True}, immutable=True)
    write(OUT / "STAGE_ROLES.json", {"groups": stage_groups, "queries": stage_roles,
                                      "all_D_previously_consumed": True,
                                      "purpose": "honest training of the sequential study adapter, not new D confirmation",
                                      "H_groups": roles["confirmation_pool_group_ids"]}, immutable=True)
    write(OUT / "PRECISION_PLAN.json", {"created_utc": now(), "candidate_designs": candidates, "selected": chosen,
                                         "selection_rule": "largest cap in [4,8,12,16,all] with conservative required groups <=95% of H capacity and >=20 changed D groups; uses variance, never estimated effect sign",
                                         "primary": "CodeBERT ENTR-minus-RAND20 best3 RR@100; min(3,N), without NoRewrite in best3",
                                         "other_comparisons": "predefined descriptive estimates; no independent null-rejection claims",
                                         "metric_scope": "standard query-weighted MRR in the frozen capped query sample; not all-H MRR or group-balanced MRR",
                                         "full_H": "all8530 conventional MRR retained as predeclared sensitivity, not relabeled as 80%-powered",
                                         "four_equal_primary_comparisons_retained": False,
                                         "target_effect_unchanged": .005, "H_outcomes_opened": False}, immutable=True)
    print({"precision": chosen, "primary_queries": len(primary),
           "D_stage_queries": {k: len(v) for k, v in stage_roles.items()}}, flush=True)


if __name__ == "__main__":
    prepare()
