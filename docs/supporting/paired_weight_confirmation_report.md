# Paired Two-Weight Confirmation Report

The independent new query/connected-group sample contains 25,600 groups and a fixed corpus of 32,768 code snippets. Eligibility, sample size, models, complete generation/retrieval configuration and the six-comparison family were locally sealed before new generation and effect inspection. This was not external preregistration.

## All six prespecified comparisons

| Comparison | Estimate | Adjusted t interval | Adjusted group-bootstrap interval | Direction | Nonzero groups |
|---|---:|---|---|---|---:|
|local::ENTR_B|+0.002803|[+0.001565, +0.004041]|[+0.001576, +0.004024]|Positive|9,734|
|local::ENTR_G|+0.003241|[+0.002409, +0.004073]|[+0.002422, +0.004091]|Positive|11,173|
|t5base::ENTR_B|+0.004282|[+0.002941, +0.005624]|[+0.002946, +0.005626]|Positive|9,781|
|t5base::ENTR_G|+0.003520|[+0.002636, +0.004404]|[+0.002636, +0.004427]|Positive|11,485|
|base_minus_local::ENTR_B|+0.001479|[-0.000325, +0.003282]|[-0.000336, +0.003267]|Unresolved|10,982|
|base_minus_local::ENTR_G|+0.000279|[-0.000909, +0.001467]|[-0.000942, +0.001477]|Unresolved|12,416|

All four ENTR components meet the prespecified six-comparison correction and joint interval-direction rule. In this new StaQC sample, with fixed CodeBERT and the specified complete generation evaluation contract, the paired confirmation supports positive B/G directions for both T5 weights. This strengthens the second contribution's evidence across the tested weight conditions.

Paired weight differences subtract local from base for each query. A resolved difference would concern the complete weight conditions rather than isolate CQC training itself; unresolved differences do not imply equivalence. This experiment neither specifies nor tests an effect-equivalence margin.

## Comparisons and effect scale

B=S(1,mean)−S(3,mean), G=S(3,mean)−S(3,best3). S is the ENTR RR@100 advantage over exact random selection within the complete position pool for each weight. Positive values indicate a smaller relative advantage along the specified evaluation-contract steps. These quantities are neither automatic-execution gains, generation-call budgets nor unique causal attributions.

| Weights | S(1,mean) | S(3,mean) | S(3,best3) | NoRewrite MRR@100 |
|---|---:|---:|---:|---:|
|local|+0.005374|+0.002571|-0.000671|0.113417|
|t5base|+0.007713|+0.003431|-0.000089|0.113417|

PROB, ENTR−PROB and the complete k1/2/3 × mean/best-of-k results remain prespecified descriptive controls in COMPLETE_CURVES.csv and RESULTS.json. They do not add confirmatory rejection claims.

## Eligibility and planning

After excluding historical references, 84,618 single-target StaQC connected groups remained. Excluding the previous complete corpus, queries, historical generation and normalized exposure left 68,233 eligible components. Fixed hashes selected one item per component, query components and distractor corpus components. New and old complete-corpus components did not overlap.

Planning used only consumed matched caches from the two domains and the larger local-weight result. The target resolution was .005 RR with a 95% power target for each direction; at 25,600 groups, the planned lower bound for all four directions was 0.9652. This plan assumes transferable variance for a .005 effect; it does not guarantee resolution of a smaller true effect.

The planned adjusted half-width for each of the two weight differences was at most .005. Here .005 is a precision target, not an equivalence margin. There was no interim effect-based stopping, seed addition, model fitting or policy promotion.

Accepted single-code-answer titles and snippets define one fixed retrieval condition. Other snippets were not exhaustively judged irrelevant. Unknown base-model pretraining exposure, semantic clones and undocumented historical exposure are outside the local eligibility guarantee.

## Relationship to earlier evidence

The historical 512-query matched analyses in each domain retain their 24-comparison family. The earlier StaQC 13,312-group/16,384-code study retains its two-comparison family. The 25,600 groups form a separate new component sample and fixed corpus; its six-comparison family does not retrospectively change earlier conclusions. The results do not establish a finding across model families or all retrievers, or attribution to historical full training.

## Recorded validation

Validation included 14 evaluation-contract tests; replay of 360 historical-query position spans under both weights and their complete ENTR/PROB orderings; checks of all query/corpus provenance, group independence, prior exposure, candidate/action coverage and rankings; independent rational-number/CDF reconstruction of all B/G values; independent replay of six t intervals and 20,000 common-group bootstrap resamples; and actual retriever reload with full-corpus ranking replay for 32 queries and 32 code snippets.

The frozen experiment export contains DESIGN.json, ELIGIBILITY.json, PRECISION_PLAN.json, PREOPEN_AUDIT.json, NUMERICAL_AUDIT.json, MODEL_AUDIT.json and FINAL_ACCEPTANCE.json. Delivered results are in `results/experiment_exports/rene_weight_confirmation_20261021/` and `results/paired_weight_confirmation_results.json`.
