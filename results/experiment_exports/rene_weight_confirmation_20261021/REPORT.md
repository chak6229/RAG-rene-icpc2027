# Paired B/G confirmation with two sets of T5 weights on a new sample

Independent new queries and linked groups: 25,600. Fixed code corpus: 32,768 snippets.
Eligibility, sample size, models, the complete generation and retrieval settings, and the six-comparison family were sealed locally before new generation and effect inspection. There was no external preregistration.

## Six prespecified comparisons

|Comparison|Point estimate|Six corrections t|Six correction groups bootstrap|Pre-determined direction determination|Nonzero groups|
|---|---:|---|---|---|---:|
|local::ENTR_B|+0.002803|[+0.001565, +0.004041]|[+0.001576, +0.004024]|positive|9,734|
|local::ENTR_G|+0.003241|[+0.002409, +0.004073]|[+0.002422, +0.004091]|positive|11,173|
|t5base::ENTR_B|+0.004282|[+0.002941, +0.005624]|[+0.002946, +0.005626]|positive|9,781|
|t5base::ENTR_G|+0.003520|[+0.002636, +0.004404]|[+0.002636, +0.004427]|positive|11,485|
|base_minus_local::ENTR_B|+0.001479|[-0.000325, +0.003282]|[-0.000336, +0.003267]|unresolved|10,982|
|base_minus_local::ENTR_G|+0.000279|[-0.000909, +0.001467]|[-0.000942, +0.001477]|unresolved|12,416|

All four ENTR components satisfied the prespecified six-comparison correction and two-interval directional rule. The paired confirmation supports a positive direction for B and G under both sets of T5 weights on this new StaQC sample, with CodeBERT and the specified generation contract fixed.

The paired weight contrast subtracts local from base for each query. A resolved contrast would concern the whole weight condition. It would not isolate CQC training. The unresolved direct differences do not establish equivalence. No effect equivalence margin was set or tested for this study.

## Comparators and effect sizes

B=S(1,mean)−S(3,mean), G=S(3,mean)−S(3,best3). S is the RR@100 advantage of ENTR over exact random selection from the full position pool for the same weight condition. Positive values indicate a decrease in the relative advantage along the specified evaluation contract. They measure neither the gain of an automatic policy nor generation calls, and do not identify a unique causal contribution.

|weights|S(1,mean)|S(3,mean)|S(3,best3)|NoRewrite MRR@100|
|---|---:|---:|---:|---:|
|local|+0.005374|+0.002571|-0.000671|0.113417|
|t5base|+0.007713|+0.003431|-0.000089|0.113417|

PROB, ENTR−PROB and the full k1/2/3×mean/best-k curves were prespecified as descriptive outputs. See `COMPLETE_CURVES.csv` and `RESULTS.json`. No additional confirmatory rejection decisions were made.

## Eligibility and planning

- Historical exclusions left 84,618 single-target StaQC linked groups. Removing prior corpus and query overlap, historical generation overlap and normalized exposures left 68,233 eligible components. Fixed hashes selected one record per component and determined the query and distractor corpus samples. The full old and new corpora share no eligible components.
- Planning used the already-used matched caches from both domains and the larger local-weight results. The resolution target was .005 RR with 95% target power for each direction. At 25,600 groups, the planned joint lower bound across four directions was 0.9652. This calculation assumes transferable variances for a .005 effect and does not guarantee detection of smaller effects.
- The planned corrected half-width for each direct weight contrast was at most .005. Here .005 is a precision target. The study did not stop based on interim effects, add seeds, fit models or promote a policy.
- Accepted single-code titles and snippets define this retrieval setting. Other snippets do not have exhaustive negative human labels. The local eligibility checks do not cover unknown foundation-model pretraining exposure, semantic clones or unrecorded historical exposure.

## Relationship to existing evidence

The earlier matched study with 512 queries per domain and its 24-comparison family remain as reported, as does the earlier StaQC two-comparison family with 13,312 groups and 16,384 snippets. This 25,600-group analysis uses a separate new component sample and a new fixed corpus. Its six-comparison family does not revise the previous decisions. The new findings do not establish effects across rewriter families or all retrievers, or isolate the effect of historical full training.

## Acceptance and verification

The checks included 14 contract tests, replay of 360 spans and complete ENTR/PROB orders on historical queries under both weight conditions, a full audit of query and corpus provenance, independent groups, known prior exposure, and candidate, action and ranking coverage. Independent rational and CDF calculations reproduced all B/G results. The six t intervals and the shared 20,000-resample group bootstrap were independently replayed. Reloading the retriever for 32 queries and 32 code snippets supported the complete corpus-rank replay.

Verification files: `DESIGN.json`, `ELIGIBILITY.json`, `PRECISION_PLAN.json`, `PREOPEN_AUDIT.json`, `NUMERICAL_AUDIT.json`, `MODEL_AUDIT.json`, and `FINAL_ACCEPTANCE.json`.
