"""Validate acceptance and manuscript integration against already audited evidence."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from experiments.rene_relevance_sensitivity_20261021.common import ROOT, frozen, put, read, rows, sha
from .prepare import EDITED, OUT, PAPER, PRIOR, binding


def compact(text):
    return re.sub(r"(?<=[+-])0\.", ".", re.sub(r"\s+", "", text))


def bounds(values):
    return "[" + ", ".join(f"{value:+.6f}" for value in values) + "]"


def nominal(pairs):
    n = len(pairs)
    a, b = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    observed = Fraction(sum(x == y for x, y in pairs), n)
    chance = sum(Fraction(a[k] * b[k], n * n) for k in set(a) | set(b))
    return float(observed), float((observed - chance) / (1 - chance))


def ordinal(pairs):
    pairs = [(int(x), int(y)) for x, y in pairs if x != "u" and y != "u"]
    n = len(pairs)
    a, b = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    observed = sum(Fraction(abs(x - y), 2 * n) for x, y in pairs)
    expected = sum(Fraction(abs(x - y) * a[x] * b[y], 2 * n * n)
                   for x in range(3) for y in range(3))
    return n, float(1 - observed / expected)


def main():
    assert not (OUT / "FINAL_ACCEPTANCE.json").exists(), "Preserve an already accepted delivery."
    checks = Counter()

    def require(condition, category):
        if not condition:
            raise AssertionError(category)
        checks[category] += 1

    protocol = frozen()
    require(len(protocol["bound_inputs_and_design"]) == 24, "frozen_design_inputs")
    for item in read(OUT / "PROTECTED_FILES.json")["files"]:
        require(sha(ROOT / item["path"]) == item["sha256"], "protected_prior_files")
    for item in read(OUT / "RECEIPT_MANIFEST.json")["files"]:
        require(sha(Path(item["source"])) == item["sha256"], "original_submitted_files")
        require(sha(OUT / item["copy"]) == item["sha256"], "received_copies")

    result, prior = read(OUT / "RESULTS.json"), read(PRIOR / "RESULTS.json")
    prior_validation = read(PRIOR / "NUMERICAL_VALIDATION.json")
    require(prior_validation["passed"] is True, "prior_numerical_validation")
    require(prior_validation["results_sha256"] == sha(PRIOR / "RESULTS.json"), "prior_results_binding")
    require(prior_validation["query_bounds_sha256"] == sha(OUT / "query_bounds.jsonl"), "exact_bounds_binding")
    numeric_keys = ["original_review_comparison", "original_review_agreement", "supplement_agreement",
                    "pooled_top3_agreement", "coverage", "display_validation", "primary_before_after",
                    "summary", "query_records", "bounds_are_sample_identification_ranges_not_confidence_intervals",
                    "population_gain_or_causal_explanation_percentage_established",
                    "original_agreement_and_mechanism_statistics_unchanged", "original_mainline_factor_results_unchanged"]
    for key in numeric_keys:
        require(result[key] == prior[key], "audited_result_fields_preserved")
    for name in ["query_bounds.jsonl", "supplement_labels.jsonl", "supplement_disagreements.jsonl",
                 "REPRESENTATIVE_DISAGREEMENTS.json"]:
        require(sha(OUT / name) == sha(PRIOR / name), "audited_numeric_files_preserved")

    acceptance, confirmation = read(OUT / "REVIEW_ACCEPTANCE.json"), read(OUT / "INDEPENDENCE_CONFIRMATION.json")
    require(acceptance["passed"] and acceptance["formally_accepted_labels"] == 688, "formal_collection_complete")
    require(result["formal_dual_independent_supplement_completion"] is True, "completed_result_status")
    require(confirmation["answer_verbatim"] == "是，A 确实独立完成，只是漏勾选", "explicit_user_confirmation")
    require(all(confirmation["confirmed_scope"].values()), "confirmation_scope")
    require(sha(ROOT / confirmation["original_return"]["path"]) == confirmation["original_return"]["sha256"],
            "confirmation_bound_to_original_return")
    require(read(ROOT / confirmation["original_return"]["path"])["independence_confirmed"] is False,
            "raw_A_flag_preserved")
    require(acceptance["accepted_by_unchanged_collector"] == 344
            and acceptance["accepted_with_explicit_separate_confirmation"] == 344,
            "separate_acceptance_bases")

    packets = read(ROOT / "results/rene_relevance_sensitivity_20261021/private/reviewer_keys.json")
    label_rows = list(rows(OUT / "supplement_labels.jsonl"))
    labels = {(x["reviewer"], x["query_id"], x["panel_id"]): x["label"] for x in label_rows}
    require(len(labels) == len(label_rows) == 688, "unique_supplementary_labels")
    for reviewer in ("A", "B"):
        payload = read(OUT / f"received/补审{reviewer}/relevance-supplement-{reviewer}-complete.json")
        for case in packets[reviewer]["cases"]:
            for doc, panel in case["document_map"].items():
                require(labels[reviewer, case["query_id"], panel] == payload["answers"][case["case_id"]]["documents"][doc],
                        "unchanged_raw_label_and_reason")
    order = sorted((q, p) for reviewer, q, p in labels if reviewer == "A")
    pairs = [(labels["A", q, p]["relevance"], labels["B", q, p]["relevance"]) for q, p in order]
    agree, kappa = nominal(pairs)
    judged_n, weighted_kappa = ordinal(pairs)
    supplement = result["supplement_agreement"]
    require(math.isclose(agree, supplement["including_unknown"]["agreement"], abs_tol=1e-12)
            and math.isclose(kappa, supplement["including_unknown"]["kappa"], abs_tol=1e-12),
            "independent_supplementary_nominal_agreement")
    require(judged_n == 296 and math.isclose(weighted_kappa, supplement["jointly_judged"]["kappa"], abs_tol=1e-12),
            "independent_supplementary_ordinal_agreement")
    disagreements = list(rows(OUT / "supplement_disagreements.jsonl"))
    require(len(disagreements) == sum(a != b for a, b in pairs) == 154, "supplementary_disagreement_count")
    unknown_pairs = sum(a != b and (a == "u" or b == "u") for a, b in pairs)
    higher = sum(a != "u" and b != "u" and int(b) > int(a) for a, b in pairs)
    lower = sum(a != "u" and b != "u" and int(b) < int(a) for a, b in pairs)
    require((unknown_pairs, higher, lower) == (48, 83, 23), "supplementary_disagreement_components")
    examples = read(OUT / "REPRESENTATIVE_DISAGREEMENTS.json")["examples"]
    for example, category in zip(examples, [("0", "2"), ("u", "2"), ("1", "2")]):
        eligible = [d for d in disagreements if (d["A"]["relevance"], d["B"]["relevance"]) == category]
        require(example["query_id"] == min(eligible, key=lambda d: (d["query_id"], d["panel_id"]))["query_id"]
                and (example["query_id"], example["panel_id"]) == min((d["query_id"], d["panel_id"]) for d in eligible),
                "transparent_example_selection")
        for reviewer in ("A", "B"):
            require(example[reviewer] == labels[reviewer, example["query_id"], example["panel_id"]],
                    "examples_preserve_both_judgments")

    summary = result["summary"]
    index = {(x["reviewer"], x["threshold"], x["cutoff"], x["policy"]): x for x in summary}
    require(len(index) == len(summary) == 48, "all_summary_combinations")
    require(sum(1 for _ in rows(OUT / "query_bounds.jsonl")) == result["query_records"] == 6144,
            "all_paired_query_records")
    preview = read(ROOT / "results/rene_relevance_sensitivity_20261021/preview/RESULTS.json")
    old_index = {(x["reviewer"], x["threshold"], x["cutoff"], x["policy"]): x for x in preview["summary"]}
    for key, item in index.items():
        if item["cutoff"] == 1:
            require(item == old_index[key], "RR1_results_unchanged")
    for reviewer in ("A", "B"):
        for threshold in (1, 2):
            require(index[reviewer, threshold, 3, "RIDGE_verified"]["weighted_expanded_delta_identification_bounds"][0] > 0,
                    "weighted_verified_positive_within_sample")
            require(index[reviewer, threshold, 3, "ENTR_top1"]["weighted_expanded_delta_identification_bounds"][1] < 0,
                    "weighted_direct_ENTR_negative")
            require(index[reviewer, threshold, 3, "SAFE_ENTR_best3_opportunity"]["weighted_expanded_delta_identification_bounds"][0] > 0,
                    "label_assisted_opportunity_positive")
    strict_B = index["B", 2, 3, "RIDGE_verified"]
    require(strict_B["unweighted_expanded_delta_bounds"][0] < 0 < strict_B["unweighted_expanded_delta_bounds"][1],
            "strict_unweighted_B_boundary")
    require(strict_B["weighted_change_from_original_delta_bounds"][0] < 0 < strict_B["weighted_change_from_original_delta_bounds"][1],
            "B_change_from_original_qrels_boundary")
    require(strict_B["weighted_original_qrels_delta"] > 0, "RR3_original_qrels_already_positive")
    require(not all(index[w, t, 3, "RIDGE_risk"]["weighted_expanded_delta_identification_bounds"][0] > 0
                    for w in ("A", "B") for t in (1, 2)), "risk_stage_boundary")
    for item in result["primary_before_after"]:
        old = item["preview"]["weighted_expanded_delta_identification_bounds"]
        new = item["with_all_returned_supplement_labels"]["weighted_expanded_delta_identification_bounds"]
        require(old[0] <= new[0] <= new[1] <= old[1], "primary_ranges_narrow")

    tables = read(OUT / "DIAGNOSTIC_TABLES.json")
    require(tables["primary_before_after"] == result["primary_before_after"], "diagnostic_table_sources")
    for entry in tables["threshold1_weighted"] + tables["strict_policy_rows"]:
        require(entry == index[entry["reviewer"], entry["threshold"], entry["cutoff"], entry["policy"]],
                "diagnostic_table_sources")
    for entry in tables["threshold2_unweighted"]:
        require(entry["bounds"] == index[entry["reviewer"], 2, 3, "RIDGE_verified"]["unweighted_expanded_delta_bounds"],
                "diagnostic_table_sources")

    documents = {name: (PAPER / name).read_text(encoding="utf-8-sig") for name in EDITED}
    documents["FINDINGS.md"] = (OUT / "FINDINGS.md").read_text(encoding="utf-8-sig")
    numbers = []

    def cited(name, expected, pointer):
        require(compact(expected) in compact(documents[name]), "manuscript_numeric_citations")
        pointers = pointer.split(" and ")
        for path in pointers:
            value = result
            for token in path.lstrip("/").split("/"):
                value = value[int(token)] if isinstance(value, list) else value[token]
            require(value is not None, "manuscript_source_pointers")
        numbers.append({"document": name, "expected": expected,
                        "source": "results/rene_review_completion_20261021/RESULTS.json", "pointers": pointers})

    for name in ("relevance_sensitivity_results.md", "integrated_results.md", "FINDINGS.md"):
        for n, item in enumerate(result["primary_before_after"]):
            before, after = item["preview"], item["with_all_returned_supplement_labels"]
            expected = (f'|{item["reviewer"]}|{bounds(before["weighted_expanded_delta_identification_bounds"])}|'
                        f'{bounds(after["weighted_expanded_delta_identification_bounds"])}|'
                        f'{before["point_identified_queries"]} → {after["point_identified_queries"]}/128|')
            cited(name, expected, f"/primary_before_after/{n}")
            for t in (1, 2):
                source = index[item["reviewer"], t, 3, "RIDGE_verified"]
                field = "weighted_expanded_delta_identification_bounds" if t == 1 else "unweighted_expanded_delta_bounds"
                cited(name, bounds(source[field]), f"/summary/{summary.index(source)}/{field}")
            cited(name, bounds(after["weighted_change_from_original_delta_bounds"]),
                  f"/primary_before_after/{n}/with_all_returned_supplement_labels/weighted_change_from_original_delta_bounds")
        cited(name, f'{strict_B["weighted_original_qrels_delta"]:+.6f}',
              f'/summary/{summary.index(strict_B)}/weighted_original_qrels_delta')
    for policy in ("ENTR_top1", "RIDGE_verified", "RIDGE_risk", "SAFE_ENTR_best3_opportunity"):
        left, right = index["A", 2, 3, policy], index["B", 2, 3, policy]
        for name in ("relevance_sensitivity_results.md", "FINDINGS.md"):
            cited(name, f'|{policy}|{bounds(left["weighted_expanded_delta_identification_bounds"])}|'
                       f'{bounds(right["weighted_expanded_delta_identification_bounds"])}|',
                  f"/summary/{summary.index(left)} and /summary/{summary.index(right)}")
    for name in ("main_results_narrative.md", "manuscript.md", "contribution_evidence.md",
                 "knowledge_increment.md", "outline.md", "results_contract.md"):
        for reviewer in ("A", "B"):
            item = index[reviewer, 2, 3, "RIDGE_verified"]
            cited(name, bounds(item["weighted_expanded_delta_identification_bounds"]),
                  f"/summary/{summary.index(item)}/weighted_expanded_delta_identification_bounds")

    stale = [r"await", r"pending", r"No supplementary human returns", r"尚无新增人工返回",
             r"两人各待补", r"344待每人补判", r"补判完成状态需", r"待回复"]
    for p in PAPER.glob("*.md"):
        text = p.read_text(encoding="utf-8-sig")
        require(not any(re.search(pattern, text, re.I) for pattern in stale), "no_stale_current_paper_status")
    require("not confidence intervals" in documents["relevance_sensitivity_results.md"], "bounds_distinguished_from_CI")
    require("sampling and annotation uncertainty" in documents["relevance_sensitivity_results.md"], "uncertainty_scope_reported")
    require("unweighted" in documents["main_results_narrative.md"], "weighting_sensitivity_in_main_text")
    old_discussion = (OUT / "before/review_disagreement_discussion.md").read_text(encoding="utf-8-sig")
    original_cases = old_discussion.split("## Where the judgments differ", 1)[1].split("## What is resolved", 1)[0]
    require(original_cases in documents["review_disagreement_discussion.md"], "original_disagreement_counts_and_cases_preserved")
    for example in examples:
        require(example["query_id"] in documents["review_disagreement_discussion.md"], "supplementary_cases_in_paper")

    sections = ["abstract.md", "intro_draft.md", "related_work_draft.md", "study_design.md",
                "main_results_narrative.md", "discussion.md", "conclusion.md"]
    expected = "# Position Advantage and Executable Utility: An External Replication of SSQR for Code Search\n\n"
    expected += "\n\n".join((PAPER / n).read_text(encoding="utf-8-sig").rstrip() for n in sections)
    expected += "\n\n# References and Supplementary Material\n\nBibliography: references_verified.bib. Full result tables: integrated_results.md. Statistical/source contract: results_contract.md. Contribution mapping: contribution_evidence.md. Review examples and disagreements: review_disagreement_discussion.md. Earlier decision-specific uncertainty comparisons: supplementary_uncertainty.md (T7). Additive relevance sensitivity: relevance_sensitivity_results.md (separate post-hoc diagnostic). Machine source: results/rene_manuscript_integration_20261021/TABLES.json.\n"
    require(documents["manuscript.md"] == expected, "assembled_manuscript_matches_7_sources")
    original_tables = read(ROOT / "results/rene_manuscript_integration_20261021/TABLES.json")["tables"]
    require(len(original_tables) == 15 and sum(len(v) for v in original_tables.values()) == 176,
            "original_15_tables_176_rows")

    changes = []
    for item in read(OUT / "PAPER_BEFORE.json")["files"]:
        current = ROOT / item["path"]
        before = OUT / "before" / current.name
        require(sha(before) == item["sha256"], "paper_before_snapshot_preserved")
        changes.append({"path": item["path"], "before_sha256": item["sha256"],
                        "after_sha256": sha(current), "changed": sha(current) != item["sha256"]})
    put(OUT / "MANUSCRIPT_CHANGES.json", {"files": changes, "assembly_sources": sections})
    put(OUT / "MANUSCRIPT_NUMBERS.json", {"checked_citations": numbers})
    put(OUT / "VALIDATION.json", {
        "passed": True, "checks": dict(checks), "total_checks": sum(checks.values()),
        "prior_independent_exact_bound_records": prior_validation["independent_exact_paired_bounds"],
        "prior_independent_numerical_validation": binding(PRIOR / "NUMERICAL_VALIDATION.json"),
        "numeric_values_and_exact_bound_files_unchanged": True,
        "formal_independence_acceptance": "REVIEW_ACCEPTANCE.json",
        "new_training_generation_or_retrieval": False,
    })
    files = [p for p in OUT.rglob("*") if p.is_file()]
    files += [PAPER / name for name in EDITED]
    files += list(Path(__file__).parent.glob("*.py"))
    put(OUT / "DELIVERY_MANIFEST.json", {"scope": "completed supplementary review and current manuscript integration",
                                         "files": [binding(p) for p in sorted(files)]})
    put(OUT / "FINAL_ACCEPTANCE.json", {
        "passed": True, "formal_dual_independent_supplement_completion": True,
        "returned_and_formally_accepted_supplement_labels": 688,
        "raw_A_independence_flag": False, "original_A_flag_preserved": True,
        "A_acceptance_basis": "separate explicit human-user confirmation",
        "judgments_reasons_unknowns_and_disagreements_preserved": True,
        "original_15_tables_and_prior_experiments_preserved": True,
        "manuscript_and_7_sources_synchronized": True,
        "primary_endpoint": {"policy": "RIDGE_verified", "threshold": 2, "cutoff": 3,
                             "weighting": "original_eight_stratum", "interval": "sample identification bounds, not CI"},
        "numerical_scope": "previously independently replayed exact results retained without recomputation",
        "validation": binding(OUT / "VALIDATION.json"),
        "delivery_manifest": binding(OUT / "DELIVERY_MANIFEST.json"),
        "report": "results/rene_review_completion_20261021/FINDINGS.md",
        "manuscript": "paper/rene2027/manuscript.md", "pending_collection_items": [],
    })
    print({"passed": True, "checks": sum(checks.values()), "accepted_supplement_labels": 688,
           "numeric_citations": len(numbers), "paper_sources_checked": len(EDITED),
           "original_tables": 15, "original_table_rows": 176})


if __name__ == "__main__":
    main()
