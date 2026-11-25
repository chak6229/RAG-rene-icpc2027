# Completed original and supplemental reviews: results and manuscript integration

The author confirmed that reviewer A completed the supplemental judgments independently without seeing the experimental effects, prior scores or reviewer B's judgments. The separately recorded confirmation for the unchecked box is in `INDEPENDENCE_CONFIRMATION.json`. Original returns, labels and rationales were retained. B met the original intake declaration, and 344 supplemental judgments per reviewer were accepted, for 688 total. See `REVIEW_ACCEPTANCE.json`.

## Supplemental evidence

Each original reviewer assessed 128 queries, 383 rewrites and 327 code panels, with the original fields unchanged. The supplement adds 344 panels per reviewer across 126 queries in the original sample, retaining the query count and reviewer count. We retained 19 returned files, checked 2,108 judgments and rationales across web, text and JSON formats, and independently replayed 48 analysis combinations and 6,144 exact paired bounds. This step uses verified results without new training, generation or retrieval.

Supplemental exact agreement was 55.23%, with nominal κ=.321. Both reviewers supplied known grades on 296/344 panels (86.05%), with linear-weighted κ=.484. Of 154 disagreements, 48 involved an unknown and 106 involved different known grades. In the merged pool of 610 panels, each reviewer retained 49 unknowns without automatic adjudication. Original semantic agreement remained 76.50% (κ=.525), and original code agreement including unknowns remained 53.82% (κ=.371). Original weighted event statistics were retained.

## Primary diagnostic and sensitivities

The prespecified endpoint uses RIDGE_verified, threshold 2, and RR@3 weighted over the original eight strata. The contrast subtracts original-query RR from policy RR. The sample identification bounds reflect unknown relevance and exclude sampling and annotation error.

|Reviewer|Pre-trial rehabilitation|After re-trial|Openable Query|
|---|---|---|---:|
|A|[-0.017841, +0.041899]|[+0.005843, +0.018263]|102 → 121/128|
|B|[-0.032886, +0.059318]|[+0.001279, +0.016604]|96 → 122/128|

Under the original qrels, the same endpoint is +0.001594. Supplementation narrows the unknown range and shows sensitivity to the relevance set. The change relative to original qrels is [+0.004250, +0.016669] for A and [-0.000315, +0.015011] for B. The B change still includes 0.

|Sensitivity settings|A|B|
|---|---|---|
|Rating 1 / 2 Inclusion, original layer weighting RR@3|[+0.003297, +0.005007]|[+0.001948, +0.010069]|
|Rating 2 Incorporated, unweighted RR@3|[+0.003906, +0.023438]|[-0.006510, +0.011719]|

## Interpretation for the study

|Threshold 2, original layer weighted RR@3|A|B|
|---|---|---|
|ENTR_top1|[-0.040360, -0.006583]|[-0.045258, -0.017126]|
|RIDGE_verified|[+0.005843, +0.018263]|[+0.001279, +0.016604]|
|RIDGE_risk|[-0.000177, +0.000324]|[+0.001755, +0.001962]|
|SAFE_ENTR_best3_opportunity|[+0.093181, +0.118574]|[+0.047352, +0.103379]|

The supplemental review strengthens the diagnostic of the evaluation contract and fixed automatic policy. In the prespecified weighted sample, the post-verification bounds are positive for both reviewers at both thresholds, direct ENTR single selection remains negative and the safe best3 opportunity remains positive. A positive effect after risk fallback is not established jointly across the reviewers and thresholds. B's strict unweighted bounds still span 0.

The separate B/G, StaQC and paired retriever results retain their original interpretation. The supplement does not establish a positive full-H MRR@100 result, a stable full-pipeline gain, or mutually exclusive explanations for losses. The recovery limits for the historical weights, training CSV, production aggregation and full-training attribution remain in force.

## Historical manuscript and delivery records

The historical draft was `paper/rene2027/manuscript.md`. The relevance supplement, six associated source sections, integrated results, outline, contribution map and evaluation contracts were synchronized at that stage. The original 106 rewrite, 151 code and 35 clarity disagreements and three examples were retained. The supplement adds 154 disagreements, grade distributions and three examples about request versus response, missing helper context and completeness of output. Both reviewers' recorded judgments were retained.

See `RESULTS.json`, `DIAGNOSTIC_TABLES.json` and `query_bounds.jsonl` for the numerical evidence, and `supplement_disagreements.jsonl` and `REPRESENTATIVE_DISAGREEMENTS.json` for disagreements. The 15 earlier tables and their 176 machine-source rows were preserved. `before/` was the historical manuscript backup. This report supersedes the earlier provisional intake status.

`FINAL_ACCEPTANCE.json` records the verification. Manuscript review, conference formatting and the final artifact were subsequent submission steps.
