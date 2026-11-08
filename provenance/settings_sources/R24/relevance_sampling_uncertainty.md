# Supplementary Diagnostic: Sampling Uncertainty in Expanded Relevance

## Target and design

The completed relevance diagnostic in relevance_sensitivity_results.md reports identification ranges over unknown code relevance in a reviewed sample. This analysis separately quantifies sampling uncertainty in those ranges. Its target is the finite CodeBERT H8530 query frame, with each reviewer's final relevance rule, policy and cutoff treated as fixed. The original eight strata and their frame sizes are retained; each contributes 16 queries selected by salted hash order. Inference interprets this procedure as pseudorandom simple random sampling without replacement within strata. Queries are the sampling units; the 120 associated groups are not a replacement sampling design.

For each query, shared-text unknown assignments produce lower and upper paired RR contrasts, l(q) and u(q). We estimate each finite-frame endpoint by the original weighted mean, sum_h (N_h/8530) mean_h. Reviewer A and B and relevance thresholds 2 and 1 remain separate. The intervals include sampling uncertainty and unknown-relevance identification uncertainty. They do not estimate annotation error or a population of reviewers, replace the original qrels, or change the full-H MRR@100 result.

This is an analysis after the original effects, labels and sample ranges were visible. The relevance diagnostic was specified before supplementary judgments, not before the original sample was reviewed. We therefore report design-based intervals conditional on treating the final rule as fixed; the nominal coverage is not a guarantee adjusted for data-informed choice of that rule. This diagnostic is not a new method-effect confirmation.

## Conservative finite-frame envelope

RR@1 has support {0,1}; RR@3 has support {0,1/3,1/2,1}. Query-level contrast endpoints lie in the full difference support. For RR@3 this is {-1,-2/3,-1/2,-1/3,-1/6,0,1/6,1/3,1/2,2/3,1}. Unobserved support values are retained.

At each support threshold in each stratum, we invert a one-sided hypergeometric tail for the finite-population count at or below that threshold. For the lower endpoint, upper CDF limits yield a lower mean limit; for the upper endpoint, lower CDF limits yield an upper mean limit. The discrete identity E[X] = x_min + sum_j (x_(j+1)-x_j) P(X > x_j) combines the counts into endpoint limits. Original frame weights combine the stratum means. Bonferroni controls both endpoint directions, eight strata and all support thresholds. At RR@3 this allocates 0.05/(2 × 8 × 10) to each one-sided tail. Under the stated fixed-rule sampling interpretation, the resulting 95% outer envelope has conservative finite-sample design coverage for both population endpoints.

This construction is intentionally conservative. Sixteen queries per stratum, full-support count limits and simultaneous tail control allow substantial mass at values absent from the sample. A wide envelope describes the resolution of this procedure; it does not establish a large underlying loss or gain.

## Approximate precision reference

We also report the Neyman variance estimate with finite-population correction:

```text
Vhat(endpoint) = sum_h (N_h/8530)^2 (1 - n_h/N_h) s_h^2/n_h
```

Welch–Satterthwaite degrees of freedom and one-sided t critical values at alpha/2 combine the lower and upper endpoint limits into an approximate 95% outer envelope. With 16 queries per stratum and sparse nonzero contrasts, this is a precision reference with approximate coverage. It does not replace the conservative finite-frame method or provide a finite-sample guarantee.

## Results for the declared verified-policy diagnostic

All rows use RIDGE_verified minus the original query at RR@3 with the original stratum weights. Each 95% envelope below covers the two endpoints within one reviewer/threshold combination under its stated assumptions. Threshold 2 is the primary relevance diagnostic; threshold 1 includes partially relevant code.

|Relevance threshold|Reviewer|Sample identification range|Conservative finite-frame 95% envelope|Stratified-t approximate 95% envelope|
|---:|---|---|---|---|
|2|A|[+0.005843, +0.018263]|[-0.400821, +0.417409]|[-0.008645, +0.033373]|
|2|B|[+0.001279, +0.016604]|[-0.401856, +0.420320]|[-0.009318, +0.033184]|
|1|A|[+0.003297, +0.005007]|[-0.395174, +0.400332]|[-0.006890, +0.014480]|
|1|B|[+0.001948, +0.010069]|[-0.406116, +0.413443]|[-0.014874, +0.025913]|

We additionally control the four reviewer-by-threshold combinations together by using alpha=0.05/4 for each conservative envelope. The four-combination family was declared for this post-hoc uncertainty report; it does not amend the original diagnostic protocol or the B/G comparison families.

|Relevance threshold|Reviewer|Conservative envelope with simultaneous 95% coverage across these four combinations|
|---:|---|---|
|2|A|[-0.450782, +0.466530]|
|2|B|[-0.451641, +0.469207]|
|1|A|[-0.445096, +0.450156]|
|1|B|[-0.455725, +0.462720]|

The sample identification ranges are positive in all four weighted comparisons. Every conservative and approximate sampling envelope spans zero. These observations support sensitivity of the sampled fixed-policy utility to relevance coverage; they do not establish positive utility in H8530, equivalence to zero, or a stable deployment benefit. Strict unweighted reviewer-B results, the changes relative to original qrels, and risk-stage sensitivities retain their original interpretation. No interval includes annotation error or generalizes beyond the fixed frame.

## Evidence and validation

The input is the unchanged 6,144 query-level records in results/rene_review_completion_20261021/query_bounds.jsonl. All 48 existing reviewer/threshold/cutoff/policy combinations are evaluated, without selecting by interval sign. Each row has its own 95% envelope; there is no claim of simultaneous 95% coverage across all 48 combinations. The four-combination family above is the separately reported simultaneous result.

Methods and code are in experiments/rene_relevance_sampling_uncertainty_20261021/. RESULTS.json, FINDINGS.md and NUMERICAL_VALIDATION.json are in results/rene_relevance_sampling_uncertainty_20261021/. Numerical replay separately checks the original weighted ranges, Fraction-based variances, hypergeometric inversions and aggregate envelopes for all combinations. Small-population enumeration checks the finite-frame coverage and variance contracts. Original labels, unknowns, disagreements, qrels and the accepted 15-table bundle are preserved.
