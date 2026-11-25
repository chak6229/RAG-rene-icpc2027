from __future__ import annotations

from .common import OUT, now, read, require, sha, status, write


def put(name, lines):
    path = OUT / name
    require(not path.exists(), "report already exists")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    require(read(OUT / "FINAL_ACCEPTANCE.json")["passed"], "independent acceptance required")
    design, policy = read(OUT / "DESIGN.json"), read(OUT / "PROTOCOL.json")
    precision = read(OUT / "PRECISION_PLAN.json")
    primary = read(OUT / "PRIMARY_RESULT.json")
    ledger = read(OUT / "TRAINING_LEDGER.json")
    acceptance = read(OUT / "FINAL_ACCEPTANCE.json")
    selected = precision["selected"]
    s1 = ['# First step: Single adoption strategy and independent confirmation', "",
          'Only complete adoption strategy frozen before confirmation RIDGE_risk: 71 Direct proceeds selection, priority for real returns CE Validation, three categories risk retreat. CodeBERT The primary searcher, BM25 performs the same algorithm as a boundary analysis.', "",
          'This is a new CoSQA research adapter selected on the basis of developing evidence; The HGB identity of the original development protocol remains unchanged, and HGB is retained as against the same input. The old strong84, the integration process and the five official pointers were not replaced.', "",
          'Training roles cross-repeated by associated groups: revenue 953 query/307 group; Verify update 288 query/103 group; Risk 249 Query/ 102 group. All three players are from the consumer development D and cannot be identified as new. H Only freezing strategy reasoning.', "",
          'Check that each searcher has the most floor (n/4) of the highest positive yield priority, and each 8 actual query-code query; yes; Use the original user query evaluation code. Update selection only in the same ENTR candidate pool. The end risk threshold is fixed. 075, refusal to restore Top100 and no re-selection or refund of the cost of verification.', "",
          'Unique validation comparison: CodeBERT ENTR−RAND20 candidate best3, tested on frozen capped16 samples. PROB, BM25, HGB and stage difference are predefined description estimates without new confirmation rejection claims.', "",
          f"Design SHA256: `{sha(OUT / 'DESIGN.json')}`. Full-policy SHA256: `{sha(OUT / 'PROTOCOL.json')}`。", "",
          'The opening chain has been reconciled: the design of a model/strategy actually confirms that pre-independence audits have produced clean directories. All 8530 queries/1963 groups are now consumed and may no longer be used to select new strategies.']
    s2 = ['# Step two, confirm accuracy.', "",
          'The original four corporate needs 2475 group; Inclusion of the original D/H group size sensitivity required 3662 group, which cannot be completed with the existing 1963 group. This round explicitly replaces the confirmed design and retains the old plan.', "",
          'Back to original unified scheme CodeBERT ENTR - RAND Main issues; All 1963 associated groups are retained, each group up to 16 query and select 7718 article in the order scheduled for SHA256. The indicator is still weighted by the usual query in the sample MRR; The target population is designed for this frozen sample and does not pretend to be full of 8530 queries, etc.', "",
          'The upper limit within the group is based on D variance feasibility options: pre-listed 4 / 8 / 12 / 16 / All, select to meet conservative group needs up to capacity 95% The maximum limit. H forecasts or labels were not used, and no positive or negative selection was made for D returns. All 8530 results assign sensitivity in advance.', "",
          f"Target difference remains... 005, double side alpha =... 05, efficacy 80%. D Difference 10000 Sub-group bootstrap 95% Upper bounds are{selected['variance_bootstrap_upper95']:.10f};group size sensitivity multipliers{selected['size_imbalance_multiplier_at_least1']:.6f}. Conservative need{selected['required_groups']}Group, Capacity 1963 Group, Planning Effectiveness{selected['planned_power_at_capacity']:.2%}, 80% Minimum calibration{selected['mde80']:.6f}。", "",
          'The plan relies on D to H conditions differentials and group independence assumptions; Not a mathematical guarantee of unknown H differences. Non-retrovalence replaces actual compartments.', "",
          f"Actual master comparison MRR ={primary['mean']:+.8f};t95% slots{primary['ci95_cluster_t']};groups bootstrap95%{primary['ci95_group_bootstrap']};change groups{primary['changed_groups']}。", "",
          f"Interpret by freezing rules: `{primary['conclusion']}`。", "",
          '(b) Intersectional breach 0 indicates only that direction is not determined; The material equivalence requires that both 95% compartments are located entirely in ±. 005 and at least 20 groups of changes. This rule only binds the selection of negative entropy candidate positions and does not extend to all UQ conclusions.']
    s3 = ['# Step 3 CoSQA Complete evidence of validation and risk retreat', "",
          'All 8530 Article queries, 1963 associated groups; Each searcher uses the same ENTR three candidates (when insufficient) min. 3 N) and frozen strategy. The following phases, as well as between impact zones, are pre-defined descriptive estimates.', "",
          '| Retrieval | Stage | MRR@ 100 | Use/harmful use of internal hazard rate | positive gain retention CE logarithms', "|---|---|---:|---:|---:|---:|---:|"]
    for ret in ("CodeBERT", "BM25"):
        summary = read(OUT / f"metrics/{ret}_full_H.json")
        s3 += [f"| {ret} ENTR Candidate Security oracle {summary['safe_ENTR_oracle_delta']:+.8f} | — | — | 100% | 0 |"]
        for name in ("ENTR_top1", "HGB_gain_control", "RIDGE_gain", "RIDGE_verified", "RIDGE_risk"):
            m = summary["policy_results"][name]
            harm = m["harm_within_adopted"]["value"]
            retention = m["positive_gain_retention"]["value"]
            h = 'Undefined' if harm is None else f"{harm:.2%}"
            r = 'Undefined' if retention is None else f"{retention:.2%}"
            s3.append(f"| {ret} | {name} | {m['delta_MRR']['mean']:+.8f} | {m['adopted']}/{m['harmed']} | {h} | {r} | {m['CE_pairs_charged']} |")
        g = summary["gate_accounting"]
        s3 += ["", f"{ret}The door.{g['before_delta']:+.8f} - Misdirection of positive gains{g['missed_positive_gain']:.8f} + Avoidance of loss{g['avoided_loss']:.8f} = Behind the door{g['after_delta']:+.8f}. Constant equation error{g['identity_max_error']:.3g}。", "",
               f"{ret}Phase pair 95% describes the interval:"]
        for name, m in summary["stage_contrasts"].items():
            s3 += [f"- {name}：{m['mean']:+.8f}，{m['ci95_group_bootstrap']}。"]
        s3 += [""]
    s3 += ['(b) The positive rate of retention is the denominator of the total positive income in the same search security oracle; Not proportional to net gains close to zero. Verification may change the course of action and report separately on its poor stage; Risk accounting compares only fixed post-check actions.', "",
           'Use of internal harmful frequency by association group bootstrap; Do not use group dependence as a separate Bernoulli risk certificate. risk_certified = false, with full deployment remaining NoRewrite.', "",
           'CE Enter token, for the only cost of physics versus each strategy see INFERENCE_LEDGER.json and metrics/* _full_H.json.']
    put("STEP1_POLICY_FREEZE.md", s1)
    put("STEP2_PRECISION.md", s2)
    put("STEP3_STAGE_RESULTS.md", s3)
    put("REPORT.md", ['# RENE Independent confirmation Accuracy and completeness phase acceptance', "", *s2[2:], "", *s3[2:], "",
                      f"Independent inspection:{acceptance['model_decisions_reloaded']}Reloading of project models;{acceptance['Fraction_RR_checks']}entry Fraction RR restart; maximum value difference{acceptance['max_metric_error']:.3g}。", "",
                      'Clean cataloguing and acceptance are independent value reconstructions for cache ranking/decision-making to the master table, not posing as re-training neuromodels. Real T5 Full vocabulary FP64, retrieve rankings, CE cross-check with model reasoning.', "",
                      'For a complete and single strategy see STEP1_POLICY_FREEZE.md; Accuracy STEP2_PRECISION.md; Phase breakdown. STEP3_STAGE_RESULTS.md'])
    write(OUT / "EXPOSURE_LEDGER.json", {"created_utc": now(), "former_role": "unopened_H",
                                         "current_role": "consumed_independent_confirmation", "queries": 8530, "groups": 1963,
                                         "one_shot_opening_sha256": sha(OUT / "OUTCOME_OPENING.json"),
                                         "not_available_for_future_policy_selection": True}, immutable=True)
    write(OUT / "REPORT_RECEIPT.json", {"created_utc": now(), "reports": {name: sha(OUT / name) for name in
                                         ("REPORT.md", "STEP1_POLICY_FREEZE.md", "STEP2_PRECISION.md", "STEP3_STAGE_RESULTS.md")},
                                         "final_acceptance_sha256": sha(OUT / "FINAL_ACCEPTANCE.json")}, immutable=True)
    status("complete", acceptance_passed=True)


if __name__ == "__main__":
    main()
