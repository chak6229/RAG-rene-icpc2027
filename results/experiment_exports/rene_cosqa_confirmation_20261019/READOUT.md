# CoSQA independent confirmation, precision and full-stage analysis

The experiment, independent primary-pipeline verification and additional final checks are complete. All H results were inspected. The H effects did not determine the models, thresholds or primary comparison.

**Step 1: freezing and independent confirmation**

Development D contains 1490 queries in 512 linked groups. Gain, verification and risk fitting used 953/307, 288/103 and 249/102 queries/groups, respectively, with disjoint linked groups. Full H contains 8530 queries in 1963 groups against a fixed corpus of 6267 documents.

The full policy selects positive predicted gain with Ridge71. It allocates verification to at most 25% of queries by descending positive gain and scores each original query against 8 distinct real documents with a cross-encoder. A 91-feature residual model updates the scores with coefficient .5. The 20-feature three-class risk head applies a fixed .075 gate and restores the full original Top100 after rejection. The HGB control uses the same inputs.

The CoSQA implementation adapts the rewrite study to single-pass features and does not impute absent across-seed features with zeros. It does not establish equivalence to the historical strong84 or fused production pipeline. The previous production version and five CURRENT/STABLE pointers were retained.

The sole confirmation contrast is CodeBERT ENTR−RAND20 best3 RR@100. Best3 excludes NoRewrite, and the selected count is min(3,N). The 20 random orders share one generation and retrieval cache and are repeated baselines on the same queries. PROB, BM25, stage differences and matched controls were prespecified as descriptive estimates.

Effects were opened at `2026-10-19T13:37:05.741621+00:00`. The matched control for additional supervision was sealed earlier at `2026-10-19T13:36:08.968117+00:00`.

| Main comparative calibration | Query/group | ENTR−RAND20 ΔMRR | Group t 95% | Group bootstrap 95% |
|---|---:|---:|---|---|
| Freezing of master samples, maximum in each group 16 | 7718/1963 | -0.00083652 | [-0.00320993, +0.00153688] | [-0.00325479, +0.00154146] |
| All H Presensitive | 8530/1963 | -0.00106661 | [-0.00333560, +0.00120237] | [-0.00337214, +0.00123976] |

The primary contrast has 1500 nonzero groups. The frozen decision was `{"direction": "difference from zero unresolved", "practical_equivalence_within_prespecified_plusminus005": true, "advantage_above005_excluded": true, "does_not_imply_all_UQ_is_ineffective": true}`.

An interval spanning zero leaves direction unresolved. Practical equivalence for the specified contrast requires both 95% intervals to be fully within the prespecified ±.005 margin and at least 20 nonzero groups. The inference applies to the designated contrast.

**Step 2: addressing limited precision**

The earlier plan for four primary contrasts required 2475 groups and had only 1963 available. The revised design uses one primary contrast, retains every group, caps each group at 16 queries using a fixed hash order, and yields 7718 queries. The full 8530-query sensitivity is reported separately without claiming the same 80% power.

Candidate caps were 4/8/12/16/all. The largest cap with a conservative group requirement within 95% of capacity and at least 20 nonzero D groups was selected using D variance. The rule used neither H predictions and effects nor the sign of D effects. The estimand is query-weighted MRR under the frozen sampling design.

The design retains δ=.005, two-sided α=.05 and 80% target power. The bootstrap 95% upper bound on D variance was 0.00553101176. The group-size multiplier was 1.05585196. The design required 1836 groups out of 1963 available. Planned power was 82.58% and MDE80=0.00483432. `PREOPEN_NUMERICAL_CHECK` and `PREOPEN_VARIANCE_CHECK` document independent noncentral t and D variance checks.

Planning assumes independent linked groups and transferable conditional variance from D to H. It does not guarantee the realized H precision or 80% power for the equivalence test. The observed intervals determine the decision.

**Step 3: full-stage evidence**

Both retrievers run the same frozen pipeline separately and use the full 8530 queries as denominator. Stage results are prespecified descriptive estimates. The safe oracle takes the label-assisted maximum across the same ENTR candidates and NoRewrite and measures available opportunity.

**CodeBERT**: NoRewrite MRR=0.51832406. Same-candidate safe-oracle Δ=+0.06609266.

| Phase | ΔMRR | Use/harmful | Coverage | Use of internal harmful rates | Retention of positive proceeds | CE Counting pairs |
|---|---:|---:|---:|---:|---:|---:|
| Negative entropy Top1 | -0.02110467 | 8530/2328 | 100.00% | 27.29% | 42.52% | 0 |
| Compare with input HGB returns | -0.00287610 | 5305/1277 | 62.19% | 24.07% | 27.02% | 0 |
| Ridge Proceed selection | +0.00007864 | 6034/1317 | 70.74% | 21.83% | 23.04% | 0 |
| True CE | +0.00084705 | 5890/1280 | 69.05% | 21.73% | 24.35% | 17056 |
| Risk back. | -0.00017777 | 1006/60 | 11.79% | 5.96% | 0.77% | 17056 |

| Phase Pair | ΔMRR | Group bootstrap 95% Description interval |
|---|---:|---|
| Validation - selection of proceeds | +0.00076841 | [-0.00013923, +0.00169727] |
| Risk retreat - verification | -0.00102482 | [-0.00336725, +0.00133546] |
| Full Process - NoRewrite | -0.00017777 | [-0.00055106, +0.00019457] |

Risk fallback accounting: before the gate +0.00084705 − forgone positive gain 0.01558429 + avoided loss 0.01455946 = after the gate -0.00017777.

The gate rejected 4884 actions that changed ranks, including 1090 with positive gains and 1220 with losses. Accounting error was 1.65e-18. Zero-gain actions remain in the coverage denominator. The decomposition describes the observed stages.

**BM25**: NoRewrite MRR=0.19814905. Same-candidate safe-oracle Δ=+0.01941318.

| Phase | ΔMRR | Use/harmful | Coverage | Use of internal harmful rates | Retention of positive proceeds | CE Counting pairs |
|---|---:|---:|---:|---:|---:|---:|
| Negative entropy Top1 | -0.01443510 | 7955/2929 | 93.26% | 36.82% | 39.10% | 0 |
| Compare with input HGB returns | -0.00271496 | 2615/853 | 30.66% | 32.62% | 11.80% | 0 |
| Ridge Proceed selection | -0.00207943 | 3559/1072 | 41.72% | 30.12% | 5.63% | 0 |
| True CE | -0.00171681 | 2926/906 | 34.30% | 30.96% | 4.43% | 17056 |
| Risk back. | +0.00000000 | 0/0 | 0.00% | Undefined | 0.00% | 17056 |

| Phase Pair | ΔMRR | Group bootstrap 95% Description interval |
|---|---:|---|
| Validation - selection of proceeds | +0.00036263 | [-0.00017839, +0.00091982] |
| Risk retreat - verification | +0.00171681 | [+0.00115385, +0.00228134] |
| Full Process - NoRewrite | +0.00000000 | [+0.00000000, +0.00000000] |

Risk fallback accounting: before the gate -0.00171681 − forgone positive gain 0.00085910 + avoided loss 0.00257590 = after the gate +0.00000000.

The gate rejected 2926 actions that changed ranks, including 78 with positive gains and 906 with losses. Accounting error was 4.34e-19. Zero-gain actions remain in the coverage denominator. The decomposition describes the observed stages.

**Control for additional supervision**

The matched control was fixed before effect inspection. It uses the same verification training queries, labels, row weights, Ridge regularization, .5 residual coefficient, H allocation and ENTR candidates. The control removes only 18 CE columns. The direct contrast compares actual CE with the equally supervised update without CE.

| retriever | CE Update MRR | Real CE Update Δ MRR | CE Increase in condition | Match 95% Description interval |
|---|---:|---:|---:|---|
| CodeBERT | +0.00055187 | +0.00084705 | +0.00029518 | [-0.00050908, +0.00110818] |
| BM25 | -0.00171310 | -0.00171681 | -0.00000371 | [-0.00029280, +0.00029104] |

The auxiliary-stage intervals condition on the sealed verification allocations and actions for this batch. They do not provide a population guarantee for repeatedly allocating a 25% budget. No confirmatory superiority or equivalence tests were added.

**Workload and acceptance**

The two retrievers share 33631 distinct physical pairs in the CE cache, with 2522373 input tokens. The table reports per-policy charges separately by retriever. For each retriever, the verified and risk-gated stages have the same incurred cost. The matched comparison holds paid cost fixed, although the standalone no-CE policy could skip CE inference.

The primary chain passed 32 contract checks and reloaded 119420 model decisions. The independent clean-directory replay recomputed 249102 Fraction RRs with maximum metric discrepancy 2.22e-16.

The final matched-control audit independently recomputed 51180 Fraction RRs and their intervals, and rechecked source, model, result and report hashes. Recomputing 64 original-query/code pairs directly from the CE weights gave maximum logit discrepancy 0. This verification was audit work outside the policy budget.

The clean-directory replay reconstructs metrics from cached rankings and decisions. T5 full-vocabulary FP64, the complete dense ranking cache and the independent BM25 formula were checked in the pre-opening audit. Training the models was outside the cached replay.

The comparison is bounded by the single known CoSQA positive, fixed corpus, disclosed upstream code overlap and missing repository mappings. Independence refers to separation of documented linked groups in downstream fitting and confirmation. It does not establish absence of all upstream exposures.

Empirical harm frequencies and group intervals do not certify population risk. `risk_certified=false`. The deployment decision remains as recorded. H8530/1963 is a previously used evaluation set.

Machine-readable evidence: `PRIMARY_RESULT.json`, `AUXILIARY_RESULTS.json`, `metrics/*_full_H.json`, `MATCHED_CONTROL_RESULTS.json`, `FINAL_ACCEPTANCE.json`, and `../rene_cosqa_confirmation_review_20261019/FINAL_REVIEW.json`.
