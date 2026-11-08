# Matched T5-weight sensitivity of candidate-count and aggregation effects

We extended the two-weight Top1/best3 comparison to a complete candidate-count × aggregation analysis using the existing per-candidate retrieval cache. Both the local CQC T5 and public T5-base were evaluated on exactly the same 512 AdvTest queries (512 repository groups) and 512 CoSQA queries (512 connected groups). Original-query rankings, corpus, qrels, the CodeBERT retriever, and original insertion-position counts were held fixed. No training, candidate generation, model inference, or retrieval was repeated.

For ENTR and PROB, we selected the first k∈{1,2,3} positions by the original generation score with position-based tie breaking. Exact uniform sampling without replacement over positions retained duplicate-text probability mass. The mean aggregation is the expected RR@100 under uniform adoption within the selected prefix; best-k is a label-assisted opportunity measure. Writing S(k,a) for the selector advantage over exact random, we used B=S(1,mean)−S(3,mean), G=S(3,mean)−S(3,best3), and J=B+G. These quantities specify an arithmetic path through the evaluation contracts; k denotes selected count from an already generated pool.

The locally sealed sensitivity design specified 24 primary comparisons: two datasets × two selectors × two components × {local, base, paired base−local}. We report two-sided Bonferroni-adjusted cluster-t intervals and 20,000 whole-group percentile bootstrap intervals, using shared draws across both weights. A direction is established only when both adjusted intervals exclude zero on the same side and at least 20 groups have nonzero contributions. Other contrasts, including ENTR−PROB and J, use descriptive pointwise 95% intervals.

| Dataset | Weight | Selector | B (adjusted CI) | G (adjusted CI) |
|---|---|---|---|---|
| AdvTest | local | ENTR | -0.005239 [-0.019681, +0.009203] | +0.003479 [-0.006850, +0.013808] |
| AdvTest | local | PROB | -0.002036 [-0.016130, +0.012057] | +0.005028 [-0.005560, +0.015615] |
| AdvTest | t5base | ENTR | +0.001237 [-0.011825, +0.014298] | +0.005483 [-0.003986, +0.014952] |
| AdvTest | t5base | PROB | +0.007238 [-0.005819, +0.020295] | +0.009344 [+0.000480, +0.018208] |
| CoSQA | local | ENTR | +0.005815 [-0.013606, +0.025237] | +0.009869 [-0.001979, +0.021718] |
| CoSQA | local | PROB | +0.011552 [-0.007936, +0.031039] | +0.011625 [-0.000438, +0.023688] |
| CoSQA | t5base | ENTR | +0.018967 [-0.003537, +0.041472] | +0.012087 [-0.001941, +0.026114] |
| CoSQA | t5base | PROB | +0.014316 [-0.006975, +0.035607] | +0.018360 [+0.003595, +0.033124] |

On CoSQA, both weights yielded positive ENTR point estimates for B and G: local B=+0.005815 and G=+0.009869; base B=+0.018967 and G=+0.012087. The ENTR advantage followed +0.021066→+0.015251→+0.005381 for local and +0.034753→+0.015786→+0.003699 for base, along k1-mean→k3-mean→k3-best. None of these eight within-weight ENTR component tests across the two datasets was resolved after the full correction. AdvTest B changed point-estimate sign between weights, from −0.005239 to +0.001237. PROB G was positive after correction under T5-base in both datasets (+0.009344 and +0.018360); the remaining primary comparisons were unresolved.

All eight paired weight differences in B/G were unresolved after correction. For example, CoSQA ENTR ΔB=+0.013152 with adjusted cluster-t CI [−0.016157,+0.042461], and ΔG=+0.002217 with CI [−0.016541,+0.020975]. These intervals do not establish equivalence or weight invariance. ENTR−PROB component contrasts and their weight interactions had pointwise intervals spanning zero, providing no established entropy-specific increment.

This analysis completes the matched two-weight B/G coverage and characterizes its precision. It extends the descriptive CoSQA pattern to the second T5 weight, while leaving corrected cross-weight robustness unresolved. The consumed samples and known earlier endpoint results make this a post-hoc sensitivity analysis, separate from the StaQC frozen confirmation. The comparison concerns complete weight conditions within one model family; it does not isolate the causal effect of CQC training, recover unavailable author artifacts, or establish cross-family or full-policy transfer.

Validation: nine contract tests passed. All 16,384 old per-query anchors were recovered exactly. An independent replay used persisted Top100 rankings and qrels to recompute 18,680 candidate and 2,048 original-query reciprocal ranks as exact fractions, checked random maxima using an independent CDF formula, verified 196,608 factorial/paired fields and all 384 reported means, and independently reproduced the 24 primary statistics. The scientific result and inputs are hashed in results/rene_t5_weight_factors_20261020/.

Figures: selection_curves.pdf (pointwise cluster-t intervals) and factor_contrasts.pdf (24-comparison adjusted cluster-t intervals). Complete bootstrap intervals and all 72 factorial cells are available in the CSV exports.

## Outcome-blind new StaQC paired confirmation (2026-10-21)

A separate locally sealed study used 25,600 previously unused StaQC query/connection groups and a fixed 32,768-snippet corpus. It regenerated the complete insertion-position pools for the local CQC T5 and public T5-base weights, then used the same CodeBERT retriever and exact-random reference. The six primary contrasts were the two within-weight ENTR components under each weight and the paired t5base-minus-local B/G differences. This is a new sample and a new fixed corpus; it does not replace the historical 512-query sensitivity or the earlier 13,312-group StaQC confirmation.

| Weight condition | B | Six-comparison adjusted t interval | G | Six-comparison adjusted t interval |
|---|---:|---|---:|---|
| local | +0.002803 | [+0.001565, +0.004041] | +0.003241 | [+0.002409, +0.004073] |
| t5base | +0.004282 | [+0.002941, +0.005624] | +0.003520 | [+0.002636, +0.004404] |
| t5base − local | +0.001479 | [−0.000325, +0.003282] | +0.000279 | [−0.000909, +0.001467] |

The corresponding 20,000-draw paired component-bootstrap intervals agree with the t intervals. All four within-weight ENTR B/G directions satisfy the predeclared two-interval rule, while both direct weight differences remain unresolved. Thus the B/G direction extends to both tested T5 weight conditions in this new StaQC/title-snippet setting and fixed CodeBERT contract. The result does not establish weight equivalence, isolate CQC-training causality, establish cross-family generality, or imply automatic-policy gain. The 0.005 planning resolution for the weight differences is a precision target rather than an equivalence margin.

Eligibility, outcome opening, complete per-query results, independent numerical replay, model reload replay and final acceptance are in `results/rene_weight_confirmation_20261021/` (`REPORT.md`, `RESULTS.json`, `NUMERICAL_AUDIT.json`, `MODEL_AUDIT.json`, `FINAL_ACCEPTANCE.json`).
