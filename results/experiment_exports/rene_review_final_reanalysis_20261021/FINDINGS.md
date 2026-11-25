# Resubmitted reviews A/B and supplemental reviews A/B: verification and interpretation

All four sets matched the fixed review packets. We checked 2,108 judgments and rationales across the web, text and JSON forms. All fields of the original A/B reviews matched the prior version. Each reviewer supplied 344 new supplemental judgments. After supplementation, both reviewers' weighted relevance-expansion bounds for the prespecified diagnostic endpoint were positive in the sample.

At the time of this report, supplemental review B passed the original intake rule. The A return was complete, but its independence checkbox was `false`, so the original intake program did not accept it. Confirmation was pending. The analysis below describes both returned sets of labels and records that historical status. See the later review-completion report for the final intake status.

## Materials and initial reviews

|Materials|Query Number|Rewrite the judgement|Code judgement|Results|
|---|---:|---:|---:|---|
|Review A|128|383|327|Same item by item as the previous data field|
|Review B|128|383|327|Same item by item as the previous data field|
|Supplementary review A|126|—|344|Project complete; independently confirmed to be completed|
|Supplementary review B|126|—|344|Project integrity; adoption of original recovery rules|

The original review files differed only in serialization. The sample and the two reviewers were unchanged. Semantic agreement remained 76.50% with κ=0.525. Code relevance agreement including unknowns remained 53.82% with κ=0.371. Both reviewers could judge 260/327 panels, with linear-weighted κ=0.569. The existing weighted error shares and disagreement statistics were retained.

## Agreement on supplemental judgments

|Compensatory trial.|A|B|
|---|---:|---:|
|Satisfied|21|58|
|Partially satisfied|103|146|
|Unsatisfactory|172|140|
|cannot determine|48|0|

Exact agreement on the 344 new panels was 55.23%, with nominal κ=0.321. Both reviewers assigned known grades to 296/344 panels (86.05%), with linear-weighted κ=0.484.

Of 154 disagreements, 48 paired A's unknown rating with a definite rating from B. The other 106 had definite but different grades. Among those, B rated 83 higher and 23 lower than A. The original and supplemental sets contain different panels, so changes in κ do not isolate review quality. The two reviewers' ordering varies across disagreements.

The merged Top-3 pool contains 610 distinct panels: 266 inherited from the original reviews and 344 assessed in the supplemental review. Each reviewer retains 49 unknowns. Both assigned known grades to 513/610 panels, with linear-weighted κ=0.527. Submission completeness and known relevance are reported separately, without converting unknowns to irrelevant or adjudicating them automatically.

## Prespecified primary diagnostic endpoint

The analysis retains the original 128 queries in 120 groups, eight-stratum weights and fixed actions. The primary endpoint uses RIDGE_verified and RR@3, adding only grade 2 items to the relevant set. Extended relevance retains the known qrels positives and equivalent code texts in the pool. Each reviewer is analyzed separately, with unknown judgments enumerated by shared text variables. The table reports weighted sample identification bounds for policy RR minus original-query RR. The full-H MRR@100 results remain a separate endpoint.

|Reviewer|Compensatory pre-trial coverage|Post-retroactivity scope|This pairing is a definite query|
|---|---|---|---:|
|A|[-0.017841, +0.041899]|[+0.005843, +0.018263]|102 → 121/128|
|B|[-0.032886, +0.059318]|[+0.001279, +0.016604]|96 → 122/128|

Under the original qrels, the same RR@3 endpoint is already +0.001594. The change relative to that contrast is [+0.004250, +0.016669] for A and [-0.000315, +0.015011] for B. The interval for the B change includes 0.

|Sensitivity settings|A: RIDGE_verified Gap|B: RIDGE_verified Gap|
|---|---|---|
|Rating 1 or 2 are incorporated, original layer weighted RR@3|[+0.003297, +0.005007]|[+0.001948, +0.010069]|
|Rating only 2 Inclusion, not authorized RR@3|[+0.003906, +0.023438]|[-0.006510, +0.011719]|

Under the prespecified stratified weights, both reviewers' bounds are positive at both relevance thresholds. Under strict relevance B's unweighted bounds still span 0. The positive direction pertains to this sample and weighting rule.

## Interpretation for the study

|Fixed policy, rating 2, weighting RR@3|A Gap range|B Gap range|
|---|---|---|
|ENTR_top1|[-0.040360, -0.006583]|[-0.045258, -0.017126]|
|RIDGE_verified|[+0.005843, +0.018263]|[+0.001279, +0.016604]|
|RIDGE_risk|[-0.000177, +0.000324]|[+0.001755, +0.001962]|
|SAFE_ENTR_best3_opportunity|[+0.093181, +0.118574]|[+0.047352, +0.103379]|

1. Forced ENTR Top1 remains negative for both thresholds and reviewers in the weighted sample. Safe label-assisted best3 remains positive. These contrasts separate candidate opportunity from the outcome of a fixed automatic policy.
2. The measured utility after verification depends on the relevance set. Target-rank losses alone cannot identify changes in query meaning or user utility.
3. Positive utility after risk fallback is not established across both reviewers and both thresholds. The supplemental diagnostics do not establish a stable gain for the full policy.
4. The supplemental review adds evidence about the evaluation contract and fixed policy. The controlled B/G analyses, StaQC confirmation and retriever and weight contrasts were not rerun. The diagnostics do not apportion losses among semantic change, relevance coverage and gain prediction errors.

## Representative disagreements

The examples are the first query_id/panel_id in lexicographic order from the 0/2, u/2 and 1/2 disagreement types. They illustrate judgment boundaries and preserve both reviewers' labels without adjudication.

### Request versus response API

Query: `python get post content type` (`cosqa-train-13132`).

A: 0. The code examines response headers to decide how to decode the response. It does not read the POST request's content type.

B: 2. It identifies the type from the Content-Type header and parses it, which meets the request.

### Missing helper and visible behavior

Query: `python if type array` (`cosqa-dev-169`).

A: u. The integer dtype restriction is visible, but array detection depends on the missing `is_np_array` helper. The full behavior cannot be determined.

B: 2. It tests for a NumPy integer array, so it can determine the array type.

### String conversion versus output

Query: `python output colors from a string` (`cosqa-dev-400`).

A: 1. The code returns a string with ANSI color control codes that can be used to display color. It does not perform output or parse a color from the string contents.

B: 2. The ANSI codes wrap the string to produce color.

## Historical intake status

The audit checked the four returns, mapped 344×2 new judgments, analyzed the merged 610-panel pool and independently recomputed 48 combinations and 6,144 exact paired bounds. The intake received 688 supplemental labels. At this stage the original rule accepted B's 344 judgments and the independence confirmation for A's 344 remained pending. Files, previews, statistics and manuscript sources were retained.

The recorded next step was to confirm A's independence and update the manuscript's supplemental status, relevance sensitivity table and disagreement description. Original judgments were retained. See `RESULTS.json`, `query_bounds.jsonl` and `supplement_disagreements.jsonl` for the evidence.
