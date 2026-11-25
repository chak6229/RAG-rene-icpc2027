# CoSQA Development of small samples: SSQR Position Rating Double Retrieval Test

Completion time: 2026 - 10 - 19 T08: 45: 14.065374 + 00: 00. Develop queries 256, connect groups 86, syntax 6,267 documents.
This batch is permanently used to develop data; confirm that the pool does not run its method effects. The main settings, roles, and four comparisons are frozen before they are generated.

| retriever | NoRewrite MRR@100 | ENTR best3 Δ | PROB best3 Δ | RAND20 Mean value best3 | ENTR Top1 Δ | PROB Top1 Δ |
|---|---:|---:|---:|---:|---:|---:|
| CodeBERT | 0.49695906 | +0.07054904 | +0.07100508 | +0.06523072 | -0.02201640 | -0.01840287 |
| BM25 | 0.19238386 | +0.02120211 | +0.02036177 | +0.01638454 | -0.02485738 | -0.02048444 |

best3 is the upper limit of the pool obtained using a relevant label and cannot be considered an automatic adoption gain. Top1 Select by visible selected_rank; RAND A subcache sample of 20 from the same position pool.

| Main comparison | Matches Δ MRR | 95% Group bootstrap | Four compares with 98.75% of the area. | Holm p (development of diagnostics) | Change Group |
|---|---:|---|---|---:|---:|
| CodeBERT:ENTR_minus_RAND_best3 | +0.00531832 | [-0.007711749754408486, 0.018561619688993114] | [-0.011302681457875696, 0.022231747263962827] | 1 | 58 |
| CodeBERT:ENTR_minus_PROB_best3 | -0.00045604 | [-0.01018419463632538, 0.008557311074196921] | [-0.013485399632706608, 0.011443686279712127] | 1 | 20 |
| BM25:ENTR_minus_RAND_best3 | +0.00481757 | [-0.0026380017662490695, 0.012907991335171256] | [-0.004737096601127205, 0.015491210631924521] | 0.890902 | 50 |
| BM25:ENTR_minus_PROB_best3 | +0.00084034 | [-0.002408861339108449, 0.005663237262122122] | [-0.0029748712348980234, 0.007189033053738892] | 1 | 14 |

Freezing is source code leading reconstruction: 1 n insert position, visible greedy, 19 new token, local T5 best92000; Papers 0...n, sample m10, original checkpoint and undisclosed PROB the differences realized are listed in PROTOCOL and do not mean to accurately resume the operation of the author.

Valid/reverse test for evaluation unit 31 passed; Store action and ranking first, independently recalculating all rankings before results, BM25 formulae and real candidates FP64 entropy; After result Fraction RR / MRR Independent recalculation passed. All methods have the same denominator.

Retain confirmation pool: 8530 Query/1963 Connecting group, where the official test500 396 association group is fully retained. Known codes do not fit in with repositories; The identity of the unknown repository remains unknown.

| Power-planning contrast (target Δ=.005, 80% power, four-comparison correction) | Conservative group requirement | Available group capacity meets requirement | Sufficient nonzero groups to estimate variance |
|---|---:|---|---|
| CodeBERT:ENTR_minus_RAND_best3 | 2448 | False | True |
| CodeBERT:ENTR_minus_PROB_best3 | 1617 | True | True |
| BM25:ENTR_minus_RAND_best3 | 1059 | True | True |
| BM25:ENTR_minus_PROB_best3 | 433 | True | False |

Power is a planning approximation based on the development-group distribution. The confirmation sample size was not fixed using fewer than 20 nonzero groups. All-zero or sparse estimates do not establish equivalence. This pilot did not run confirmation, train adoption models or certify risk. The recorded deployment decision was retained.

Evidence: PROTOCOL.json, ROLE_MANIFEST.json, TEST_RESULTS.json, CANDIDATES_SEALED.json, DECISIONS_SEALED.json, PREOUTCOME_AUDIT.json, POSTOUTCOME_AUDIT.json, RQ1_PILOT.json, POWER_PLAN.json.

## This round of judgment and next steps

- This round generates 1,734 place candidates, actual 1990 actions/researchers (256 original query +1734 candidate, 1981 unique search text; Generate 47.66 seconds, CodeBERT ranking 116.4 seconds, BM25 ranking 0.7 seconds. RAND20 Select only the cache position, without increasing the number of times generated.
- The maximum label of the candidate group is positive, while the direct Top1 is negative and the loss has occurred at the automatic selection stage. The phenomenon cannot be detected for the first time alone; The main comparison of this item remains on the new query ENTR−RAND and ENTR− PROB.
- The 95% intervals for the four primary comparisons, including their simultaneous intervals, all span zero. This pilot does not establish a stable incremental benefit of negative entropy over random or probability scoring, nor does it establish equivalence or a null effect. ENTR and PROB select the same Top3 set for 149/256 queries and the same Top1 for 211/256, motivating the PROB control.
- Audit before outcome PASS: 3980 Recalculate action ranking, dense point error 0, BM25 Independent formula error 0; 16 A true development candidate for independence FP64 has the greatest error in entropy 5.475e-7. After the results, 3072 individual RR field independently rationalized, maximum error 1.11e-16. Subsequent normative query confirmation D/H text intersection 0.
- Target margin.005, 80% Effect, four-comparison correction, CodeBERT ENTR - RAND General planning 1662 Group, Conservative planning for variance uncertainty 2448 Groups; Existing confirmation pool 1963 group, conservative targets have not been met. BM25  ENTR - PROB Only 14 Change group, before we get there. 20 The group's differential planning conditions. Cannot commit to official 500 inquiries enough.
- The following shall be used to supplement the evidence of variance/change events with evidence of 426 development teams that have been permanently returned to D and are not yet operational, keeping their main settings unchanged; At the same time, direct benefit options for candidates are used as RQ2 baseline development. Confirms that the pool and its sampling sequence are maintained, and assesses once again the efficacy and complete adoption strategy have been locked in. This round did not initiate the subsequent expansion or confirmation.
- All CoSQA on-going queries remain single known related targets; The external recurrence of new queries does not amount to an upstream lack of code. Role segregation is targeted at restored connectivity groups, with unknown repository identities still unknown and extending the scope of prior data review.
