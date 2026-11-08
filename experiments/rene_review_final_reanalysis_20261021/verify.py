"""Independently replay paired relevance bounds using panel-state rankings."""

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product

from sklearn.metrics import cohen_kappa_score

from experiments.rene_review_final_reanalysis_20261021.analyze import OUT, put
from experiments.rene_relevance_sensitivity_20261021.common import (
    OUT as SENSITIVITY, read, rows, sha)


def independent(query, reviewer, threshold, cutoff, labels):
    panels = {p["id"]: p for p in query["panels"]}
    by_doc = {doc: pid for pid, panel in panels.items() for doc in panel["doc_ids"]}
    document_lists = {name: docs[:cutoff] for name, docs in query["rankings"].items()}
    panel_lists = {name: [by_doc[doc] for doc in docs] for name, docs in document_lists.items()}
    active = set(pid for ranking in panel_lists.values() for pid in ranking)
    target = set(query["target_doc_ids"])
    positive = set()
    uncertain = []
    for pid in sorted(active):
        panel = panels[pid]
        if target.intersection(panel["doc_ids"]):
            positive.add(pid)
        else:
            label = (panel["prior_labels"][reviewer] if panel["prior_labels"] is not None
                     else labels[query["query_id"], pid])
            if label["relevance"] == "u":
                uncertain.append(pid)
            elif int(label["relevance"]) >= threshold:
                positive.add(pid)

    def reciprocal(ranking, relevant):
        for index, value in enumerate(ranking, 1):
            if value in relevant:
                return Fraction(1, index)
        return Fraction(0)

    original_rr = {name: reciprocal(docs, target) for name, docs in document_lists.items()}
    original_deltas = {policy: original_rr[action] - original_rr["original"]
                       for policy, action in query["actions"].items()}
    original_deltas["SAFE_ENTR_best3_opportunity"] = max(original_rr.values()) - original_rr["original"]
    outcomes = defaultdict(list)
    for assignment in product((False, True), repeat=len(uncertain)):
        relevant = positive | {pid for pid, truth in zip(uncertain, assignment) if truth}
        rr = {name: reciprocal(ranking, relevant) for name, ranking in panel_lists.items()}
        for policy, action in query["actions"].items():
            outcomes[policy].append(rr[action] - rr["original"])
        outcomes["SAFE_ENTR_best3_opportunity"].append(max(rr.values()) - rr["original"])
    return {policy: {"bounds": [min(values), max(values)], "baseline": original_deltas[policy]}
            for policy, values in outcomes.items()}


def main():
    result = read(OUT / "RESULTS.json")
    saved = list(rows(OUT / "query_bounds.jsonl"))
    cases = list(rows(SENSITIVITY / "private/query_pools.jsonl"))
    supplements = {"A": {}, "B": {}}
    for row in rows(OUT / "supplement_labels.jsonl"):
        supplements[row["reviewer"]][row["query_id"], row["panel_id"]] = row["label"]
    independent_rows = {}
    for query in cases:
        for reviewer in ("A", "B"):
            for threshold in (1, 2):
                for cutoff in (1, 3):
                    calculated = independent(query, reviewer, threshold, cutoff, supplements[reviewer])
                    for policy, values in calculated.items():
                        independent_rows[query["query_id"], reviewer, threshold, cutoff, policy] = values
    assert len(independent_rows) == len(saved) == 6144
    checks = Counter()
    groups = defaultdict(list)
    for row in saved:
        key = tuple(row[k] for k in ("query_id", "reviewer", "threshold", "cutoff", "policy"))
        computed = independent_rows[key]
        assert computed["bounds"] == list(map(Fraction, row["expanded_delta_bounds_fraction"]))
        assert computed["baseline"] == Fraction(row["original_qrels_delta_fraction"])
        assert [bound - computed["baseline"] for bound in computed["bounds"]] == list(
            map(Fraction, row["change_from_original_delta_bounds_fraction"]))
        checks["exact_query_bound_baseline_and_change"] += 1
        groups[key[1:]].append(row)
    for summary in result["summary"]:
        key = tuple(summary[k] for k in ("reviewer", "threshold", "cutoff", "policy"))
        records = groups[key]
        weights = [Fraction(row["frame_n"], row["sample_n"]) for row in records]
        assert len(records) == 128 and sum(weights) == 8530
        for field, summary_field in (
            ("expanded_delta_bounds_fraction", "weighted_expanded_delta_identification_bounds"),
            ("change_from_original_delta_bounds_fraction", "weighted_change_from_original_delta_bounds"),
        ):
            for endpoint in (0, 1):
                exact = sum(Fraction(row[field][endpoint]) * weight for row, weight in zip(records, weights)) / 8530
                assert abs(float(exact) - summary[summary_field][endpoint]) < 1e-14
                checks["independent_weighted_endpoint"] += 1
        for endpoint in (0, 1):
            exact = sum(Fraction(row["expanded_delta_bounds_fraction"][endpoint]) for row in records) / 128
            assert abs(float(exact) - summary["unweighted_expanded_delta_bounds"][endpoint]) < 1e-14
            checks["independent_unweighted_endpoint"] += 1
    order = sorted(supplements["A"])
    left, right = [[supplements[reviewer][key]["relevance"] for key in order] for reviewer in ("A", "B")]
    independently_computed = cohen_kappa_score(left, right)
    assert abs(independently_computed - result["supplement_agreement"]["including_unknown"]["kappa"]) < 1e-14
    pairs = [(a, b) for a, b in zip(left, right) if a != "u" and b != "u"]
    ordinal = cohen_kappa_score([a for a, b in pairs], [b for a, b in pairs],
                                labels=["0", "1", "2"], weights="linear")
    assert abs(ordinal - result["supplement_agreement"]["jointly_judged"]["kappa"]) < 1e-14
    assert len(pairs) == result["supplement_agreement"]["jointly_judged"]["n"] == 296
    checks["independent_agreement_statistics"] += 3
    for item in read(OUT / "PROTECTED_FILES.json")["files"]:
        assert sha(item["path"]) == item["sha256"], item["path"]
        checks["source_and_prior_file_unchanged"] += 1
    for item in read(OUT / "RECEIPT_MANIFEST.json")["files"]:
        assert sha(OUT / item["copy"]) == item["sha256"]
        checks["received_snapshot_identical"] += 1
    validation = {
        "passed": True, "scope": "numerical_content_and_file_integrity_only",
        "formal_independence_acceptance_is_separate": True,
        "independent_exact_paired_bounds": 6144, "checks": dict(checks),
        "all_48_summary_combinations_verified": True,
        "source_html_or_code_not_executed": True,
        "results_sha256": sha(OUT / "RESULTS.json"),
        "query_bounds_sha256": sha(OUT / "query_bounds.jsonl"),
    }
    put(OUT / "NUMERICAL_VALIDATION.json", validation)
    print(validation)


if __name__ == "__main__":
    main()
