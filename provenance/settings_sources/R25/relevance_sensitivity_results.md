# Supplementary Diagnostic: Additive Relevance Sensitivity

## Question and fixed analysis

How sensitive are the sampled policies' measured utilities to adding relevant code outside the original target labels? This post-hoc diagnostic uses the same 128 queries in 120 groups, the original eight-stratum weights and fixed CodeBERT actions. Prior method outcomes and original human labels were visible before the diagnostic was designed. Its protocol, supplementary packets and primary endpoint were locally fixed before supplementary judgments; it is not an independent method-effect confirmation.

The reference remains the original query. Original qrels are preserved. For each reviewer separately, an expanded relevant set includes original positive targets, identical-text counterparts within the pooled lists, and code that reviewer grades fully relevant (2). Including partial relevance (grades 1/2) is a separate sensitivity. These additive rules test dependence on observed relevance coverage; neither reviewer's expanded labels replace the original ground truth.

We evaluate RR@1 and RR@3 with the same cutoff for original and rewritten lists. A pooled panel is the union of the original and fixed three candidate Top-3 lists, merged by exact code text. Unknown judgments remain unknown. Shared text-level assignments yield exact paired identification bounds. These are descriptive sample ranges, not confidence intervals; they omit sampling and annotation uncertainty. The separate sampling analysis below and in relevance_sampling_uncertainty.md adds design-based uncertainty without changing those ranges. The original full-H MRR@100 estimates retain their original scope.

## Completed collection and agreement

Both reviewers independently complete all 344 supplementary code judgments across 126 of the original 128 queries. The complete pooled Top-3 set contains 610 distinct query/code-text panels: 266 inherit original judgments and 344 receive new judgments. Original assessments of 383 rewrites and 327 code panels per reviewer remain unchanged; resubmission does not add queries or reviewers.

|Supplementary relevance grade|A|B|
|---|---:|---:|
|2: fully relevant|21|58|
|1: partially relevant|103|146|
|0: not relevant|172|140|
|u: unknown/context insufficient|48|0|

Supplementary exact agreement is 55.23% and nominal Cohen kappa is 0.321. The jointly judged subset covers 296/344 panels (86.05%), with linear weighted kappa 0.484. Of 154 disagreements, 48 pair A's unknown with B's numeric grade and 106 differ between numeric grades; B is higher in 83 of those 106 and lower in 23. These are displayed-panel counts, separate from the original frame-weighted query-event rates.

In the 610-panel pool, each reviewer retains 49 unknown judgments. Jointly judged coverage is 513/610 (84.10%), with linear weighted kappa 0.527; full-pool nominal kappa including unknown is 0.340. Completion concerns returned assessments, not the elimination of unknown relevance or disagreement. Original and supplementary panels differ, so their kappas do not measure a change in reviewer quality. Detailed label pairs and three additional functional-boundary examples are in review_disagreement_discussion.md.

## Primary diagnostic: weighted RIDGE_verified RR@3

The primary endpoint specified before supplementary judgments is RIDGE_verified minus the original query at RR@3, using grade-2 additions and the original eight-stratum weights. The original-qrels value for this same sampled endpoint is +0.001594. Supplementation narrows the expanded-relevance bounds as follows.

|Reviewer|Before supplementary judgments|After supplementary judgments|Identified query contrasts|
|---|---|---|---:|
|A|[-0.017841, +0.041899]|[+0.005843, +0.018263]|102 → 121/128|
|B|[-0.032886, +0.059318]|[+0.001279, +0.016604]|96 → 122/128|

Both completed ranges are positive under this rule. This is range narrowing within the sampled endpoint; its original-qrels value was already positive. The change relative to that original-qrels contrast is A [+0.004250, +0.016669]; B [-0.000315, +0.015011]. Reviewer B's change range includes zero even though B's expanded contrast is positive.

|RR@3 sensitivity for RIDGE_verified|A: policy minus original|B: policy minus original|
|---|---|---|
|Grades 1/2, original stratum weights|[+0.003297, +0.005007]|[+0.001948, +0.010069]|
|Grade 2, unweighted|[+0.003906, +0.023438]|[-0.006510, +0.011719]|

Including grades 1/2 gives positive weighted bounds for both reviewers, identifying 126 query contrasts for A and 124 for B. Under strict grade-2 relevance without the original stratum weights, B's range still includes zero. The positive direction therefore concerns the declared weighted diagnostic and does not hold across all weighting choices.

## Sampling uncertainty around the primary ranges

We retain the original eight-stratum, 16-queries-per-stratum design and frame weights, treating each reviewer's final relevance rule and fixed actions as given. Under the pseudorandom hash-as-SRSWOR interpretation, one-sided hypergeometric CDF count limits on the full RR support yield conservative 95% finite-frame envelopes. A stratified Neyman variance estimate with finite-population correction and Welch–Satterthwaite t critical values supplies an approximate precision reference.

|Grade-2 RIDGE_verified RR@3|Conservative finite-frame 95% envelope|Stratified-t approximate 95% envelope|
|---|---|---|
|A|[-0.400821, +0.417409]|[-0.008645, +0.033373]|
|B|[-0.401856, +0.420320]|[-0.009318, +0.033184]|

Both methods span zero for A and B at both relevance thresholds; the additional conservative envelopes simultaneous across all four reviewer/threshold combinations also span zero. The exact construction is wide because it covers both endpoints, all eight strata and all full-support CDF thresholds with 16 observations per stratum. The t reference is approximate. Neither method includes reviewer error or a correction for choosing the diagnostic rule after original outcomes and labels were visible. Positive sample identification ranges therefore do not establish positive utility across the H8530 frame. The complete methods, threshold-1 and simultaneous results are in relevance_sampling_uncertainty.md; all 48 combinations are in results/rene_relevance_sampling_uncertainty_20261021/RESULTS.json.

## Fixed-policy comparison under expanded relevance

|Grade 2, original stratum weights, RR@3|A: policy minus original|B: policy minus original|
|---|---|---|
|ENTR_top1|[-0.040360, -0.006583]|[-0.045258, -0.017126]|
|RIDGE_verified|[+0.005843, +0.018263]|[+0.001279, +0.016604]|
|RIDGE_risk|[-0.000177, +0.000324]|[+0.001755, +0.001962]|
|SAFE_ENTR_best3_opportunity|[+0.093181, +0.118574]|[+0.047352, +0.103379]|

ENTR Top1 has negative weighted bounds for both reviewers at both relevance thresholds; label-assisted safe best3 opportunity has positive bounds in all four cases. This preserves the distinction between available candidate opportunity and direct adoption. The verified-policy diagnostic shows how relevance coverage can change the utility reading of a fixed action. Risk fallback does not have positive bounds across both reviewers and both thresholds. These sample results do not establish a positive full-pipeline deployment effect or allocate exclusive causal shares to semantic drift, relevance coverage and decision-learning error.

## Earlier RR@1 result, unchanged by supplementation

The original Top-1 panels were already covered, with unknown labels retained. For RIDGE_verified at threshold 2, the weighted 128-query original-qrels RR@1 difference is -0.018318. Reviewer A's expansion gives +0.007547, a change of +0.025865. Reviewer B gives [-0.000440,+0.007547], with change [+0.017878,+0.025865]. The paired contrasts are identified for all 128 queries under A and 127 under B. Supplementation adds no Top-1 labels, so these values remain unchanged. Their cutoff and relevance definition differ from the original full-H MRR@100 result.

## Artifacts and reporting role

The supplementary packets show only the original query and code in independent anonymous orders, hiding old labels, method names, rewrite texts, ranks, known-target roles, strata and the other reviewer's judgments. Original labels, reasons, unknowns and disagreements remain intact.

The fixed design is experiments/rene_relevance_sensitivity_20261021/DESIGN.md. Completed evidence is in results/rene_review_completion_20261021/: RESULTS.json contains all 48 reviewer/threshold/cutoff/policy combinations; query_bounds.jsonl contains 6,144 exact paired query records; supplement_labels.jsonl and supplement_disagreements.jsonl preserve all new labels and disagreements. REVIEW_ACCEPTANCE.json records completed collection, and FINAL_ACCEPTANCE.json validates this manuscript integration.

The original existing-label analysis remains in results/rene_relevance_sensitivity_20261021/preview/. The earlier content/numerical audit remains in results/rene_review_final_reanalysis_20261021/ as a historical version. The sampling-uncertainty extension is results/rene_relevance_sampling_uncertainty_20261021/ and is integrated into the current manuscript under results/rene_contribution_strength_20261021/. These diagnostics remain separate from the frozen 15-table, 176-row bundle and do not rerun or revise its B/G, confirmation, retriever, weight or full-H effect estimates.
