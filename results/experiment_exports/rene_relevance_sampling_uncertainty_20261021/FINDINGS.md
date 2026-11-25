# Human relevance sensitivity: finite-frame sampling uncertainty

The analysis uses the original eight strata, with 16 queries sampled without replacement per stratum and the original weights. It inverts one-sided hypergeometric distributions for within-stratum CDF counts over the full discrete RR support. Bonferroni correction covers both endpoints, all eight strata and all thresholds, and the resulting bounds are combined for the stratified mean. This yields a conservative 95% design-based finite-frame envelope. A Neyman variance/t approximation with finite-population correction is included for comparison. Neither method accounts for error in the reviewers' judgments.

The primary policy is RIDGE_verified with RR@3. For each combination below, the primary envelope covers both finite-frame endpoints of the H8530 identification interval. Results are reported by reviewer A/B and threshold. `RESULTS.json` also contains a simultaneous 95% envelope for these four combinations. No simultaneous 95% coverage is claimed for all 48 combinations. The analysis is a diagnostic conducted after inspection of earlier results.

|Threshold|Reviewer|Sample identification bounds|Conservative finite-frame 95% envelope|Stratified t approximation, 95% envelope|
|---:|---|---|---|---|
|1|A|[+0.003297, +0.005007]|[-0.395174, +0.400332]|[-0.006890, +0.014480]|
|2|A|[+0.005843, +0.018263]|[-0.400821, +0.417409]|[-0.008645, +0.033373]|
|1|B|[+0.001948, +0.010069]|[-0.406116, +0.413443]|[-0.014874, +0.025913]|
|2|B|[+0.001279, +0.016604]|[-0.401856, +0.420320]|[-0.009318, +0.033184]|

Positive sample identification bounds do not establish positive utility in the finite frame. Both the exact conservative envelopes and t approximations span zero for the two reviewers and both thresholds at the primary endpoint. The population direction is unresolved. These intervals establish neither a null effect nor equivalence. The original sample, strict unweighted results, unknown judgments and original qrels were retained.

`RESULTS.json` records the full stratified hypergeometric CDF count bounds, variances, degrees of freedom, finite-population corrections and 48 combinations. The simultaneous 95% primary envelopes for the four focal combinations also span zero. The input was `results/rene_review_completion_20261021/query_bounds.jsonl`.
