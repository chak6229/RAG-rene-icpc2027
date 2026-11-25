# From candidate opportunity to realized gain: stage accounting and independent human review

**The cached stage analysis was complete. The two independent sets of judgments were pending when this report was recorded.**

This post-hoc diagnostic uses the previously evaluated CoSQA H8530 queries in 1963 groups. It adds no confirmatory comparison.

## Complete overall reconciliation of accounts

|Retriever/stage|Slurpium MRR|Positive proceeds|Scope of loss|Opportunities Leaked|Retention of positive proceeds|Actual use|Harmful|
|---|---:|---:|---:|---:|---:|---:|---:|
|CodeBERT/ENTR_top1|-0.02110467|0.02810167|0.04920633|0.03799099|42.52%|8530|2328|
|CodeBERT/HGB_gain_control|-0.00287610|0.01785652|0.02073262|0.04823614|27.02%|5305|1277|
|CodeBERT/RIDGE_gain|+0.00007864|0.01522978|0.01515114|0.05086288|23.04%|6034|1317|
|CodeBERT/RIDGE_verified|+0.00084705|0.01609260|0.01524555|0.05000006|24.35%|5890|1280|
|CodeBERT/RIDGE_risk|-0.00017777|0.00050831|0.00068608|0.06558435|0.77%|1006|60|
|BM25/ENTR_top1|-0.01443510|0.00759145|0.02202655|0.01182173|39.10%|7955|2929|
|BM25/HGB_gain_control|-0.00271496|0.00229032|0.00500528|0.01712286|11.80%|2615|853|
|BM25/RIDGE_gain|-0.00207943|0.00109306|0.00317249|0.01832012|5.63%|3559|1072|
|BM25/RIDGE_verified|-0.00171681|0.00085910|0.00257590|0.01855408|4.43%|2926|906|
|BM25/RIDGE_risk|+0.00000000|0.00000000|0.00000000|0.01941318|0.00%|0|0|

Every query U = max (0 ENTR Pool candidate RR, D = P−H; U−D=(U−P)+H. The opportunity leakage amount is U−P, which includes loss of experience in ranking and adoption and cannot be directly interpreted as the causal share of semantic drift or learning error.

- CodeBERT safe candidate opportunity: +0.06609266. Verification was allocated to 2132 queries. Across the full denominator, including queries without verification, RR improved for 158 and worsened for 139. The verification evidence included the known target for 1470 allocated queries.
- The CodeBERT gate rejected 1090 positive, 1220 negative and 2574 zero-gain actions. It removed positive gain of 0.01558429 and avoided loss of 0.01455946. Known-target coverage is an observable diagnostic cue and does not determine whether the evidence was sufficient.
- BM25 safe candidate opportunity: +0.01941318. Verification was allocated to 2132 queries. Across the full denominator, including queries without verification, RR improved for 308 and worsened for 205. The verification evidence included the known target for 491 allocated queries.
- The BM25 gate rejected 78 positive, 906 negative and 1942 zero-gain actions. It removed positive gain of 0.00085910 and avoided loss of 0.00257590. Known-target coverage is an observable diagnostic cue and does not determine whether the evidence was sufficient.

## Manual review preparation

|Layer|Full total number of articles|Fixed sample|Each weight|
|---|---:|---:|---:|
|It's not the right thing to do.|1090|16|68.1250|
|Door refusals and negative returns|1220|16|76.2500|
|Final positive gain.|55|16|3.4375|
|Final negative gain|60|16|3.7500|
|Door denial zeroes.|2574|16|160.8750|
|There's still a good chance.|931|16|58.1875|
|The rest is zero, but the ranking changes.|849|16|53.0625|
|The remaining opportunities were not positive and had not been implemented|1751|16|109.4375|

A total of 128 Article queries/120 associated groups; Each 383 rule is rewritten differently, and 327 code judgement. Scrolls of the same size cannot be considered as a whole.

Two examiners received the reviewer A.zip and the examiner B.zip. Finish the syntax page 01 first, save full export; Imports the semantic record again on the 02 code page. Only the respective packages will be handed over to the corresponding examiner; private Catalogues, current reports and statistical results are not submitted to reviewers.

The code is restored in the official original format, and all 6267 entries correspond to the actual tokenized code identity; Artificial reading information is not the same as a model visible feature.

## So far, we've been able to answer and wait for judgment.

Phased accounting allows for the identification of actions on which candidates are not selected, as well as verification/retroactivity changes in positive gains and losses. Whether semantics are transposed, whether the target really meets the original demand, and whether the unmarked code is relevant requires two independent judgments on return. The incorrect statistical record of revenue projections still cannot be the only reason for separating training, features, models and indications.

Recycling portal: python - m experiments.rene_mechanism_review_20261020 collect - a < A Complete Phase II JSON -- b < B Complete Phase II JSON > --out New Output Directory. Incomplete/strings/repeated items were verified and the statistics rejected. All discussions were held after the independent submission, with the original reservation.

The protocol/sampling/anonymous mapping is retained; original confirmation and official five pointers are maintained.
