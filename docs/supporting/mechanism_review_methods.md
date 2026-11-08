# Candidate opportunity and automatic utility: accounting and blinded review

Supplementary design prepared on 2026-10-20 and updated after receipt on 2026-10-21. The cache-based accounting, review packets and two final human-review returns are complete. Final-label agreement and weighted diagnostics are reported in mechanism_review_return_results.md; disagreement interpretation and cases are in review_disagreement_discussion.md.

## Stage accounting

For each of the 8,530 consumed CoSQA evaluation queries, let U be the maximum nonnegative RR@100 improvement available among the frozen ENTR candidates, with the original query retained as the reference. For a policy with realized improvement D, write P=max(D,0) and H=max(-D,0). Then U-D=(U-P)+H. The first term measures unretained positive opportunity and the second measures realized ranking loss. This exact accounting describes observed decisions; its terms are not mutually exclusive semantic causes.

Independent Fraction calculations from the cached rankings reproduce 153,540 reciprocal ranks across CodeBERT and BM25 and match the existing full-sample results. With CodeBERT, U averages 0.06609266. The gain selector retains positive improvement 0.01522978 while incurring loss 0.01515114, producing net +0.00007864. Verification yields positive improvement 0.01609260 and loss 0.01524555, for net +0.00084705. Risk fallback removes positive improvement 0.01558429 and avoids loss 0.01455946, leaving net -0.00017777. These quantities identify where opportunity was not retained, without assigning a semantic cause to an individual error.

## Sampling and review

Before obtaining new human judgments, we fix a stratified sample of 128 queries from the consumed CodeBERT evaluation frame. Eight disjoint strata cover rejected positive, negative and zero-improvement actions; final positive and negative actions; remaining positive opportunities; remaining zero-improvement actions that change the ranking; and remaining queries with neither opportunity nor adoption. Sixteen queries per stratum are selected by a fixed salted hash order. The sample spans 120 connected groups. This is a post-hoc diagnostic study, not a new method-effect confirmation or external preregistration.

Two Python-proficient reviewers received separately shuffled and encoded packets designed for independent assessment. The first phase assesses query clarity and distinct-rewrite meaning (preserved, drift, malformed or uncertain) with reasons. The second assesses code relevance to the original query on a 0/1/2/uncertain scale and opens after a complete first-phase export is imported. Packets hide method names, rankings, reciprocal rank, gain, selection score, stratum and known-positive role. The present analysis concerns final returned labels.

The material contains 383 distinct rewrites and 327 code panels per reviewer. Duplicate texts within a query are judged once with their position identities retained privately. Code panels include the known target, original and rewrite first results, and a verification witness where available, with roles hidden. Original formatted source is supplied for human readability; model inputs used the corresponding tokenized code. Relevance outside these panels remains unjudged.

## Analysis and interpretation

Both submitted judgments are retained. We report exact agreement and nominal Cohen kappa; code relevance additionally reports linear weighted kappa on the mutually judged 0/1/2 subset with its coverage. Query proportions use stratum population weights. Conservative finite-frame intervals combine simultaneous stratum-wise hypergeometric bounds, treating the prespecified hash ordering as a pseudorandom sampling design. This finite-frame analysis does not require independent query outcomes and does not assert generalization to future queries.

Definite semantic issues require both reviewers to label drift or malformed text; a broader possible-issue rate includes uncertainty and disagreement. Meaning-preserved actions with RR loss and non-target displayed code judged relevant by both are diagnostic evidence. These are overlapping indicators, not causal shares or bounds on one true-error rate. Complete disagreement records and final labels remain intact. Cases use the lexicographically first query with each of three existing flags; selection is post-hoc and illustrative, with prevalence estimated only from the full stratified sample. Discussion does not overwrite labels or constitute consensus adjudication.

Evidence: results/rene_mechanism_review_20261020/{PROTOCOL.json,STAGE_RESULTS.json,STAGES.csv,PACKET_STATUS.json,DELIVERY_ACCEPTANCE.json}; review return entry: experiments.rene_mechanism_review_20261020.collect. Existing confirmation results, policies and deployment pointers are preserved.
