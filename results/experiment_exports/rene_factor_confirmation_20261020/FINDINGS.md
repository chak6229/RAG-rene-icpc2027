# New-sample eligibility, precision and frozen B/G confirmation

Completed at 2026-10-20T10:27:46.954001+00:00. This confirmation was frozen in advance for a new single-code-answer StaQC setting. The original H comparison and the existing study components were retained.

## Sample and precision planning

- Strict removal of documented overlaps from the original CodeSearchNet valid/test sets left 420 queries in 103 linked groups. This sample did not meet the 0.005 resolution target, so its new method effects were not evaluated.
- The StaQC source was fixed by commit. The audit identified 84905 single-code-answer records in 84618 linked groups.
- Hash selection yielded one query per linked group, for 13312 queries in 13312 groups. The fixed corpus comprised 16384 code snippets, with the additional snippets drawn from separate linked groups.
- Before opening efficacy results, an independent check found that the original power calculation used within-group variation from groups with multiple queries, whereas the new sample had one query per group. The initial calculation was withdrawn. The original seal and unlabeled candidates were retained, and the revised plan based on matched single-query variances and a sensitivity upper bound was sealed locally. This table records the final sample and power plan.
- There were two primary comparisons. Each used a two-sided alpha of .025 and a target marginal power of 90%. Under the variance-transfer assumption, the lower bound on detecting both effects was 80%. Outcomes from the new sample did not determine its size.

|Main comparison|Required (0.005)|Practical planning effectiveness|estimate|Correction cluster-t|Correction Group bootstrap|Judgement.|
|---|---:|---:|---:|---|---|---|
|ENTR_B_budget_mean_1_to_3|11842|93.241%|+0.00325690|[+0.001721, +0.004792]|[+0.001740, +0.004814]|positive|
|ENTR_G_aggregation_at3_mean_to_max|4440|99.994%|+0.00360952|[+0.002577, +0.004642]|[+0.002576, +0.004649]|positive|

## Descriptive comparisons and adoption reference

|retriever|ENTR Top1− Original sentence|ENTR best3− Original sentence|ENTR Top1− Exact Random|ENTR best3− Exact Random|
|---|---:|---:|---:|---:|
|CodeBERT|-0.00836347|+0.02147861|+0.00635826|-0.00050817|
|BM25|-0.00931813|+0.00563239|-0.00127785|-0.00083002|

Secondary estimates are descriptive. See `RESULTS.json` for the intervals for PROB, ENTR−PROB, BM25 and the direct contrasts between retrievers. The best3 calculation uses relevance labels and measures candidate opportunity.

## Interpretation and scope

The primary test measures how changing the selected count and aggregation affects the relative position-selection advantage. The decision follows the corrected intervals for the two prespecified components. The Top1 and best3 signs alone do not determine the result.
Two positive components support both directions in this setting. A single positive component confirms only that component. Intervals spanning zero leave the direction unresolved. The primary comparisons remain as specified before inspection of the effects.
StaQC pairs Stack Overflow question titles with accepted-answer code snippets. Single-target RR in this setting differs from evaluation on Bing queries and a full-function corpus. StaQC adds evidence for a distinct setting. It does not replace the frozen CoSQA confirmation, isolate a causal effect of query source, establish an entropy-specific mechanism, or validate new UQ methods.
The audit covered documented training, development and confirmation uses, as well as original and normalized queries, code, URLs, functions and repositories. Unknown pretraining exposure and semantic clones cannot be fully ruled out. Each single-code record has one known positive. Other snippets lack exhaustive human relevance judgments.
The source labels were visible during the provenance check. The new predictions and effects remained closed until the confirmation was frozen with local timestamps and hashes. `external_preregistration=false`.

## Verification

- Before opening results, all queries, source-answer mappings, linked groups, known overlaps, candidate positions, selections and exported ranking structures were checked.
- RR was independently recomputed with `Fraction`, exact random expectation was checked with an alternative CDF formula, and corrected intervals were replayed. T5 regeneration, FP64 entropy and CodeBERT vectors were spot-checked. The receipts distinguish these checks from a full model rerun and provide counts and errors.
- Previously frozen pointers, the original H comparison and the acceptance hashes for the development factor analysis and training controls were retained. `risk_certified=false`. No additional training, seeds or policy tuning were performed.
