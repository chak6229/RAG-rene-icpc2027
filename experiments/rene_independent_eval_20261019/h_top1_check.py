"""Post-hoc H-set check: ENTR Top-1 vs NoRewrite (and best-of-3 gain).

H is already consumed; this is a diagnostic, not a new confirmation. It reads
only sealed rankings, selections and qrels, and recomputes RR independently.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import rne_eval

ROOT = r"."
BASE = os.path.join(ROOT, r"results\rene_cosqa_confirmation_20261019")
OUT = os.path.join(ROOT, r"results\rene_independent_eval_20261019\H_TOP1_CHECK.json")


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def rr(ranked, target):
    for position, doc_id in enumerate(ranked, start=1):
        if int(doc_id) == int(target):
            return 1.0 / position
    return 0.0


def main():
    sample = json.load(open(os.path.join(BASE, "PRIMARY_SAMPLE.json"), encoding="utf-8"))
    primary_ids = sample["query_ids"]
    groups = json.load(open(os.path.join(BASE, "public", "H_groups.json"), encoding="utf-8"))
    primary = set(primary_ids)

    qrels = {}
    for row in load_jsonl(os.path.join(BASE, "private", "H_qrels.jsonl")):
        qrels[row["query_id"]] = int(row["doc_id"])

    no_rewrite = {}
    candidates = defaultdict(dict)
    duplicate_positions = 0
    for row in load_jsonl(os.path.join(BASE, "rankings", "CodeBERT.jsonl")):
        query_id = row["query_id"]
        if query_id not in primary:
            continue
        if row["policy"] == "NoRewrite":
            no_rewrite[query_id] = row["ranked_doc_ids"]
        elif row["policy"] == "Candidate":
            position = int(row["position"])
            if position in candidates[query_id]:
                duplicate_positions += 1
            candidates[query_id][position] = row["ranked_doc_ids"]

    entr = {}
    for row in load_jsonl(os.path.join(BASE, "selections.jsonl")):
        if row["policy"] != "ENTR" or row["query_id"] not in primary:
            continue
        ranked = sorted(row["selected"], key=lambda item: item["selected_rank"])
        entr[row["query_id"]] = [int(item["position"]) for item in ranked]

    top1_delta = {}
    best3_delta = {}
    top1_rr = {}
    best3_rr = {}
    base_rr = {}
    missing = 0
    for query_id in primary_ids:
        target = qrels.get(query_id)
        nowrite = no_rewrite.get(query_id)
        positions = entr.get(query_id)
        if target is None or nowrite is None or not positions:
            missing += 1
            continue
        if any(position not in candidates[query_id] for position in positions):
            missing += 1
            continue
        base = rr(nowrite, target)
        top1 = rr(candidates[query_id][positions[0]], target)
        best3 = max(rr(candidates[query_id][position], target) for position in positions)
        base_rr[query_id] = base
        top1_rr[query_id] = top1
        best3_rr[query_id] = best3
        top1_delta[query_id] = top1 - base
        best3_delta[query_id] = best3 - base

    def mean(values):
        return sum(values.values()) / len(values)

    top1_ci = rne_eval.cluster_bootstrap_ci(top1_delta, groups, n_boot=10000, alpha=0.05)
    best3_ci = rne_eval.cluster_bootstrap_ci(best3_delta, groups, n_boot=10000, alpha=0.05)
    payload = {
        "schema": "h-top1-check-20261019-v1",
        "role": "post_hoc_diagnostic_on_consumed_H",
        "queries": len(base_rr),
        "groups": len({groups[query_id] for query_id in base_rr}),
        "missing_or_incomplete": missing,
        "duplicate_candidate_positions": duplicate_positions,
        "mrr": {
            "NoRewrite": mean(base_rr),
            "ENTR_top1": mean(top1_rr),
            "ENTR_best3": mean(best3_rr),
        },
        "ENTR_top1_minus_NoRewrite": {
            "mean": mean(top1_delta),
            "ci95_group_bootstrap": list(top1_ci),
            "positive": sum(1 for value in top1_delta.values() if value > 1e-10),
            "harmful": sum(1 for value in top1_delta.values() if value < -1e-10),
        },
        "ENTR_best3_minus_NoRewrite": {
            "mean": mean(best3_delta),
            "ci95_group_bootstrap": list(best3_ci),
            "positive": sum(1 for value in best3_delta.values() if value > 1e-10),
            "harmful": sum(1 for value in best3_delta.values() if value < -1e-10),
        },
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
