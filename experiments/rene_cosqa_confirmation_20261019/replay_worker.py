"""Standalone clean-directory evaluator. No experiment package imports allowed."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import t


def lines(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    root = Path(sys.argv[1]).resolve()
    sample = json.loads((root / "PRIMARY_SAMPLE.json").read_text("utf-8"))
    groups = json.loads((root / "H_groups.json").read_text("utf-8"))
    gold = defaultdict(set)
    for row in lines(root / "H_qrels.jsonl"):
        if row.get("relevance", 1) > 0:
            gold[row["query_id"]].add(row["doc_id"])
    assert set(gold) == set(groups)

    def reciprocal(q, ranking):
        assert len(ranking) == 100 and len(set(ranking)) == 100
        rank = next((i for i, doc in enumerate(ranking, 1) if doc in gold[q]), None)
        return Fraction(1, rank) if rank is not None else Fraction(0)

    selections = defaultdict(list)
    for row in lines(root / "selections.jsonl"):
        positions = [r["position"] for r in sorted(row["selected"], key=lambda r: r["selected_rank"])]
        selections[row["query_id"]].append((row["policy"], positions))
    output = {"standalone": True, "original_workspace_imported": False, "retrievers": {}}
    checked = 0
    for ret in ("CodeBERT", "BM25"):
        action = {}
        for row in lines(root / f"{ret}_rankings.jsonl"):
            key = row["query_id"], row["position"]
            assert key not in action
            action[key] = reciprocal(key[0], row["ranked_doc_ids"])
            checked += 1
        perquery = {}
        for q in groups:
            quality = defaultdict(list)
            for policy, positions in selections[q]:
                quality[policy].append(max(action[q, p] for p in positions))
            assert len(quality["ENTR"]) == 1 and len(quality["PROB"]) == 1 and len(quality["RAND"]) == 20
            perquery[q] = {name: float(sum(v) / len(v)) for name, v in quality.items()}
        stages, byquery = defaultdict(list), defaultdict(dict)
        for row in lines(root / f"{ret}_decisions.jsonl"):
            q = row["query_id"]
            value = reciprocal(q, row["ranking"])
            delta = value - action[q, None]
            stages[row["policy"]].append((q, delta, row["accepted"], row["actual_pairs"], row["actual_input_tokens"]))
            byquery[q][row["policy"]] = (delta, row)
            checked += 1
        summary = {}
        for name, values in stages.items():
            assert {q for q, *_ in values} == set(groups) and len(values) == len(groups)
            summary[name] = {"delta": float(sum(v[1] for v in values) / len(values)),
                             "adopted": sum(v[2] for v in values), "harmed": sum(v[1] < 0 for v in values),
                             "positive_gain": float(sum(max(v[1], 0) for v in values) / len(values)),
                             "loss_magnitude": float(sum(max(-v[1], 0) for v in values) / len(values)),
                             "pairs": sum(v[3] for v in values), "tokens": sum(v[4] for v in values)}
        missed = avoided = Fraction(0)
        for q, stages_q in byquery.items():
            pre, a = stages_q["RIDGE_verified"]
            post, b = stages_q["RIDGE_risk"]
            assert a["actual_pairs"] == b["actual_pairs"] and a["actual_input_tokens"] == b["actual_input_tokens"]
            if a["accepted"] and not b["accepted"]:
                missed += max(pre, 0)
                avoided += max(-pre, 0)
                assert post == 0
            else:
                assert a["ranking"] == b["ranking"] and pre == post
        summary["gate_missed_positive"] = float(missed / len(groups))
        summary["gate_avoided_loss"] = float(avoided / len(groups))
        retout = {"stages": summary, "per_query": perquery}
        if ret == "CodeBERT":
            values = {q: perquery[q]["ENTR"] - perquery[q]["RAND"] for q in sample["query_ids"]}
            gs, ns = defaultdict(float), defaultdict(int)
            for q, value in values.items():
                gs[groups[q]] += value
                ns[groups[q]] += 1
            n = len(gs)
            mean = sum(values.values()) / len(values)
            se = math.sqrt(n / (n - 1) * sum((gs[g] - mean * ns[g]) ** 2 for g in gs) / len(values) ** 2)
            half = float(t.isf(.025, n - 1)) * se
            output["primary"] = {"mean": mean, "standard_error_cluster": se, "queries": len(values), "groups": n,
                                   "ci95_cluster_t": [mean - half, mean + half]}
        output["retrievers"][ret] = retout
    output["fraction_RR_checks"] = checked
    (root / "REPLAY.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"clean_replay": "PASS", "Fraction_RR_checks": checked, "primary": output["primary"]}), flush=True)


if __name__ == "__main__":
    main()
