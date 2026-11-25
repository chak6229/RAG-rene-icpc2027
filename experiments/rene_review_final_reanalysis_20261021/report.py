"""Write a readable assessment of the reviewed labels and fixed sensitivity."""

from experiments.rene_review_final_reanalysis_20261021.analyze import OUT, put, put_rows
from experiments.rene_relevance_sensitivity_20261021.common import read, rows, sha


def interval(values):
    return "[" + ", ".join(f"{value:+.6f}" for value in values) + "]"


def selected(result, reviewer, policy, threshold=2, cutoff=3):
    return next(row for row in result["summary"] if
                (row["reviewer"], row["policy"], row["threshold"], row["cutoff"]) ==
                (reviewer, policy, threshold, cutoff))


def main():
    result = read(OUT / "RESULTS.json")
    validation = read(OUT / "NUMERICAL_VALIDATION.json")
    assert validation["passed"] and validation["results_sha256"] == sha(OUT / "RESULTS.json")
    assert result["supplement_status"]["A"]["independence_confirmed_in_return"] is False
    counts = result["supplement_agreement"]["label_counts"]
    original = result["original_review_agreement"]
    supplement = result["supplement_agreement"]
    pooled = result["pooled_top3_agreement"]
    lines = [
        '# Resubmission of review A/B and re-examination A/B: verification and interpretation of results',
        "",
        "The entries for the four sets of materials match the regular review package and the web page, text version and JSON's 2,108 judgements and justifications have been checked."
        'All the data fields of the original review A/B were identical to those of the previous edition; this time, two additional sections 344 were added.'
        'After the retrospect, both examiners at their main diagnostic end-points were on the positive side of the expanded benefit community.'
        'This has resulted in enhanced evaluation calibration and automatic admission of diagnostic evidence.',
        "",
        'Status: Additional review B adopts the original recovery rules; Additional review A is complete, but returns the file with an independent completion confirmation value of false,'
        'The original recovery program refuses to receive it formally. The user has been asked if the tick is missing and an answer remains pending.'
        'The return status of A is maintained below, with two returned labels being analysed for description and not written as a double independent review having been formally accepted.',
        "",
        '## Comparison of material and original evaluation',
        "",
        'Query of material Query Question Quest Quest Quest Quest',
        "|---|---:|---:|---:|---|",
        '| A 128 | 383 | 327 | is the same itemized as previous data fields',
        '| B 128 | 383 | 327 | is the same itemized as previous data fields',
        'A | 126 | 344 Project integrity; independent confirmation to be completed',
        'Supplementary review B|126|-|344| complete; original recovery rules adopted',
        "",
        'The original evaluation document contained only a sequencing format difference and was not a new sample or two additional reviewers.'
        'Semantic Consistency Rate remains 76.50%,κ= 0.525; Consistency rate for code contains unknown 53.82%,κ= 0.371;'
        'Common legible code 260 / 327, linear weights 0.569...old weighted error and disagreement statistics maintained...',
        "",
        '## Consistency of new additions',
        "",
        'A B',
        "|---|---:|---:|",
    ]
    for label, name in (("2", 'Satisfied'), ("1", 'Partially satisfied'), ("0", 'Unsatisfactory'), ("u", 'cannot determine')):
        lines.append(f"|{name}|{counts['A'][label]}|{counts['B'][label]}|")
    lines += [
        "",
        f"344 Accurate Consistency Rate for New Panels {supplement['including_unknown']['agreement']:.2%}，"
        f"Nominal 100={supplement['including_unknown']['kappa']:.3f}；"
        f"We’re gonna get it together. {supplement['jointly_judged']['n']}/344"
        f"（{supplement['jointly_judged_coverage']:.2%}），"
        f"Linear weight 100 ={supplement['jointly_judged']['kappa']:.3f}。",
        "",
        'Of the differences 154, 48 is unknown of A, which differs from B in its determination and 106 is one that can be judged by both parties but has different degrees.'
        'in the latter group B is higher than A to 83 and less than A to 23. The difference between the old and new batches is not directly explained by the increase or decrease in quality.'
        'Nor can a person be used to consistently loosely or narrowly summarize all differences.',
        "",
        f"The Top-3 pool is composed of 610 different code panels, of which 266 succeeds the original review and 344 comes from a complementary review."
        f"Inner pool A/B with 49 each unknown; both parties are jointly liable {pooled['jointly_judged']['n']}/610，"
        f"Linear weight 100 ={pooled['jointly_judged']['kappa']:.3f}。"
        'The re-examination entries are reported separately for all known relevance, and unknown is not entered as irrelevant or automatic.',
        "",
        '## Predicate main diagnostic endpoints',
        "",
        'Follow the original 128 Query/ 120 group, eight layer weights and fixed actions with main endpoints of RIDGE_verified, RR@3, rating only 2 to include additional relevant pools.'
        'Expand the collection to retain the original qrels text of the code and the same code in the pool; The two reviewers calculate separately, and the unknown panel is exhaustive on shared text variables.'
        'The following values are the weighted sample identification community for policy RR minus query RR,'
        'MRR@100 is not a confidence interval, nor does it replace all H.',
        "",
        'The scope of the retrofitted pretrial trials is determined by the examiner.',
        "|---|---|---|---:|",
    ]
    for row in result["primary_before_after"]:
        before, after = row["preview"], row["with_all_returned_supplement_labels"]
        lines.append(
            f"|{row['reviewer']}|{interval(before['weighted_expanded_delta_identification_bounds'])}|"
            f"{interval(after['weighted_expanded_delta_identification_bounds'])}|"
            f"{before['point_identified_queries']} → {after['point_identified_queries']}/128|"
        )
    lines += [
        "",
        'The same RR@3 endpoint below the original qrels is itself +0.001594 and this table cannot be described as a turning point from negative to positive.'
        'Changes relative to the original qrels margin, A is '
        + interval(selected(result, "A", "RIDGE_verified")["weighted_change_from_original_delta_bounds"])
        + ', B '
        + interval(selected(result, "B", "RIDGE_verified")["weighted_change_from_original_delta_bounds"])
        + '; B still contains 0.',
        "",
        'Sensitivity caliber A: RIDGE_verified Range | B: RIDGE_verified Gap |',
        "|---|---|---|",
        '| Rating 1 or 2 are incorporated, original layer weighted RR@3 |'
        + interval(selected(result, "A", "RIDGE_verified", threshold=1)["weighted_expanded_delta_identification_bounds"])
        + "|"
        + interval(selected(result, "B", "RIDGE_verified", threshold=1)["weighted_expanded_delta_identification_bounds"])
        + "|",
        '| Rating only 2 Inclusion, not authorized RR@3 |'
        + interval(selected(result, "A", "RIDGE_verified")["unweighted_expanded_delta_bounds"])
        + "|"
        + interval(selected(result, "B", "RIDGE_verified")["unweighted_expanded_delta_bounds"])
        + "|",
        "",
        'The two examiners under both relevant thresholds are on the positive side, weighted by a predetermined layer; The unweighted scope of B under the strict relevant criteria still extends 0.'
        'The conclusions are therefore limited to fixed samples and their original weighted analysis, which cannot be expanded to benefit all sample calibres.',
        "",
        '## Impact on main lines and contributions',
        "",
        '| Fixed Policy, Rating 2, Weighted RR@3 | A Difference range B',
        "|---|---|---|",
    ]
    for policy in ("ENTR_top1", "RIDGE_verified", "RIDGE_risk", "SAFE_ENTR_best3_opportunity"):
        lines.append("|" + policy + "|"
                     + interval(selected(result, "A", policy)["weighted_expanded_delta_identification_bounds"])
                     + "|" + interval(selected(result, "B", policy)["weighted_expanded_delta_identification_bounds"]) + "|")
    lines += [
        "",
        '1. Directly select ENTR the first recast weighted margin remains negative and both relevant thresholds and examiners maintain that direction.'
        'The security of tag-aided best3 opportunities are still on the right side, continuing to support candidate opportunities and assigning automatic selections as separate measures.',
        '2. Executable utility readings at the verification stage are materially sensitive to relevance clusters.'
        'Losses previously included in a single target ranking cannot all be equated with search meaning damage or reduced user utility.',
        '3. The risk retreat has not been followed by two reviewers, and both relevant thresholds can be identified as positive results.'
        'A complete automatic policy stability cannot be declared beneficial under this re-examination.',
        '4. This is a major complement to the evaluation calibration and fixed policy diagnosis of the third contribution.'
        'B/G Controlled comparison, StaQC recognition, searcher and weight control were not re-engineered and its original conclusions maintained.'
        'Semantics, relevance coverage and benefit learning errors are still not recognized to explain how much loss.',
        "",
        '## Differences in representation',
        "",
        'The following cases are taken from the three categories of disagreement 0/2, u/2 and 1/2 respectively, each taking the first in the dictionary order of query_id/panel_id;'
        'It is used to describe the boundary, keep a label on both sides and do not make third-party decisions.',
        "",
    ]
    disagreements = list(rows(OUT / "supplement_disagreements.jsonl"))
    examples = []
    for pair, description in (
        (("0", "2"), 'Functional Object for Request and Response'),
        (("u", "2"), 'Missing auxiliary and visible functions'),
        (("1", "2"), 'String conversion and finished output'),
    ):
        options = sorted((r for r in disagreements if (r["A"]["relevance"], r["B"]["relevance"]) == pair),
                         key=lambda r: (r["query_id"], r["panel_id"]))
        example = options[0]
        examples.append({"category": description, **example})
        lines += [
            "### " + description,
            "",
            'Query:' + example["original"] + "（" + example["query_id"] + "）。",
            "",
            "A：" + example["A"]["relevance"] + "；" + example["A"]["note"],
            "",
            "B：" + example["B"]["relevance"] + "；" + example["B"]["note"],
            "",
        ]
    put(OUT / "REPRESENTATIVE_DISAGREEMENTS.json", {"selection": "lexicographic first in fixed label-pair categories",
                                                   "examples": examples, "labels_adjudicated": False})
    lines += [
        '## Completion status and handover',
        "",
        'Four content verifications, 344 x 2 new judgement mapping, 610 panel merger pool analysis, 48 combination and independent recalculation of 6,144 precise matching boundaries have been completed.'
        'Received 688 complementary review labels; The original recovery rules can currently receive 344 of B and 344 of A independently confirmed pending response.'
        'The original document, the original preview, the original statistics and the source of the paper were retained.',
        "",
        'To-do to confirm the actual independent completion status of A, and then update uniformly the compensatory status in the master draft, the table of relevance sensitivity and the statement of evaluation differences.'
        'Do not rewrite the original judgement with a view to increasing it. Values, search-by-source and differences are specified in RESULTS.json,'
        "query_bounds.jsonl、supplement_disagreements.jsonl。",
        "",
    ]
    (OUT / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    put(OUT / "DELIVERY_MANIFEST.json", {
        "scope": "new_review_and_supplement_analysis_without_replacing_prior_evidence",
        "files": [{"path": str(p.relative_to(OUT)), "sha256": sha(p)}
                  for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in ("DELIVERY_MANIFEST.json", "ACCEPTANCE.json")],
    })
    acceptance = {
        "numeric_and_content_checks_passed": True,
        "independent_query_bound_replays": 6144,
        "display_judgments_and_reasons_checked": 2108,
        "returned_supplement_labels": 688,
        "supplement_content_complete": {"A": True, "B": True},
        "supplement_independence_confirmation_in_return": {"A": False, "B": True},
        "formal_dual_independent_supplement_completion": False,
        "pending": ["A actual independence confirmation; original return flag preserved"],
        "numerical_validation_sha256": sha(OUT / "NUMERICAL_VALIDATION.json"),
        "delivery_manifest_sha256": sha(OUT / "DELIVERY_MANIFEST.json"),
        "old_results_and_source_files_unchanged": True,
        "report": "FINDINGS.md",
    }
    put(OUT / "ACCEPTANCE.json", acceptance)
    print(acceptance)


if __name__ == "__main__":
    main()
