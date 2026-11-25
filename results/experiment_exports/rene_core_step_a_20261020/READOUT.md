# Step A: generation confidence under single-selection and three-candidate contracts

Date: 2026-10-20. The specified analysis A and its independent numerical checks passed. It uses frozen caches and adds no generation, training or retrieval. H rankings and qrels remained closed, and existing components and prior confirmation were retained. This is a post-hoc mechanism diagnostic on previously used development and source data.

## Main findings

**In CoSQA/CodeBERT, the measured ENTR advantage over random selection is larger under the single-candidate contract than under the three-candidate opportunity contract. The paired difference remains positive after the four-comparison correction.** PROB follows a similar pattern, and an ENTR-specific advantage was not established. The corresponding ENTR contract differences remain unresolved for both BM25 conditions and source CodeBERT. Forced ENTR Top1 yields lower MRR than the original query in all four settings.

In this controlled diagnostic, the measured utility of the same generation-confidence signal depends on selected count, aggregation and retriever. The result does not establish a positive automatic-adoption gain or validate a new UQ method.

## Evaluation contract

- The source AdvTest bridge uses 2048 queries in 594 repositories and the full 19210-function corpus. CoSQA development D uses 1490 queries in 512 linked groups and 6267 documents.
- The same local CQC T5 and `greedy20_source` profile serve frozen CodeBERT and BM25, using positions 1…n and full-vocabulary teacher-forced entropy. The T5-base summaries were excluded.
- ENTR, PROB and RAND share candidates and rankings at every position. Within each selected count the comparison uses the same pool. Duplicate texts retain the probability mass of their positions.
- The exact single-selection expectation averages RRs over positions. The main three-candidate comparator reuses the mean best3 of the original 20 random orders. Best3 measures relevance-label-assisted opportunity without including the original query.
- Compute `a=ENTR Top1−random single selection`, `b=ENTR best3−RAND20 best3` and `j=a−b` per query and take query-weighted averages. The intervals use repository or linked-group clustering, with 10000 shared group bootstrap draws and cluster-t intervals.
- Pointwise 95% intervals are descriptive. The four ENTR j contrasts also use Bonferroni correction, with a 98.75% interval for each. An interval for b spanning zero in D leaves the direction unresolved. The existing H equivalence test remains specific to its designated contrast.

## Direct matched-sample contrasts

The following are 95% group bootstrap intervals. Each row uses the same queries for a, b and j.

| Data/researcher | a: Entropy stand-alone - random stand-alone | b: Nebest3-RAND20 | j: differential effect of two contracts | j 95% |
|---|---:|---:|---:|---|
| CoSQA / CodeBERT | +.02793583 | +.00236800 | **+.02556783** | **[+.01660135, +.03437975]** |
| CoSQA / BM25 | −.00169972 | −.00194770 | +.00024798 | [−.00443244, +.00478773] |
| Source AdvTest/ CodeBERT | +.00083866 | −.00435302 | +.00519168 | [−.00016682, +.01020699] |
| Source AdvTest/ BM25 | −.00216420 | −.00133742 | −.00082678 | [−.00371686, +.00190719] |

For CoSQA/CodeBERT, the interval for a is `[+.01922302,+.03670910]`, and for b it is `[−.00348469,+.00831373]`. The latter leaves meaningful positive differences possible. The estimate of j is the direct paired contrast.

Bonferroni intervals for the four ENTR j contrasts:

| Data/researcher | Correction Group bootstrap | Correction cluster-t |
|---|---|---|
| CoSQA / CodeBERT | **[+.01416754, +.03661756]** | **[+.01426053, +.03687514]** |
| CoSQA / BM25 | [−.00586250, +.00587435] | [−.00570029, +.00619625] |
| Source AdvTest/ CodeBERT | [−.00154743, +.01167464] | [−.00141297, +.01179634] |
| Source AdvTest/ BM25 | [−.00447948, +.00263736] | [−.00443103, +.00277747] |

The direct same-query retriever difference for CoSQA is `j_CodeBERT−j_BM25=+.02531986`, with a descriptive 95% group bootstrap interval of `[+.01588118,+.03467845]`. The corresponding source difference is `+.00601846`, with a bootstrap interval spanning 0. The retriever comparison uses these paired estimates.

## Evidence on an entropy-specific advantage

CoSQA/CodeBERT：

| Comparison | Mean | Group bootstrap95% |
|---|---:|---|
| PROB Single-random single selection | +.02582259 | [+.01733916, +.03380518] |
| PROB best3−RAND20 best3 | +.00141586 | [−.00411222, +.00715333] |
| PROB Contract difference j | +.02440674 | [+.01622558, +.03236417] |
| ENTR - PROB Select | +.00211324 | [−.00419867, +.00878305] |
| ENTR−PROB best3 | +.00095215 | [−.00303516, +.00501679] |
| ENTR j−PROB j | +.00116109 | [−.00553739, +.00804653] |

The results support discussion of generation confidence under the tested contracts. Across all four settings, the ENTR−PROB intervals for single selection and best3 span zero. Superiority, equivalence and independent benefit from a new UQ method remain unestablished.

## Net utility relative to the original query

| Data/researcher | NoRewrite MRR@100 | ENTR Top1 MRR@100 | ΔMRR | Group bootstrap95% |
|---|---:|---:|---:|---|
| CoSQA / CodeBERT | .51890040 | .50524189 | −.01365851 | [−.02269701, −.00478947] |
| CoSQA / BM25 | .22412467 | .20377371 | −.02035096 | [−.02571306, −.01522141] |
| Source AdvTest/ CodeBERT | .30874417 | .28464340 | −.02410077 | [−.03044502, −.01761638] |
| Source AdvTest/ BM25 | .21275343 | .19905240 | −.01370103 | [−.01698839, −.01049453] |

Using the shared CoSQA/CodeBERT reference, random single selection minus the original query is `−.04159434`. ENTR recovers `+.02793583` relative to random, leaving `−.01365851` relative to the original query. This arithmetic describes the measured policy and does not isolate generation quality from ranking quality causally.

## Sensitivity to the 20 random replays

An additional calculation integrates the maximum RR of three positions drawn uniformly without replacement. Sort the position RRs in ascending order. The probability that the maximum is at position i is `C(i, k−1)/C(n, k)` with i starting at 0. Use `k=min(3,n)` when the pool is smaller. Positions with duplicate text retain their probability mass.

| Data/researcher | RAND20− Accurate Random best3 | Accurate randomly placed ENTR j | j Group bootstrap95% |
|---|---:|---:|---|
| CoSQA / CodeBERT | +.00041141 | **+.02515642** | **[+.01626819, +.03382412]** |
| CoSQA / BM25 | +.00081058 | −.00056260 | [−.00522946, +.00392198] |
| Source AdvTest/ CodeBERT | −.00000347 | +.00519515 | [−.00000739, +.01017433] |
| Source AdvTest/ BM25 | −.00002053 | −.00080625 | [−.00360604, +.00185966] |

The CoSQA/CodeBERT result persists under the exact random expectation. This sensitivity leaves the original primary analysis and confirmation as reported.

## Interpretation and next study

1. Matched pools and queries, two selection contracts and the PROB control provide direct evidence of a contract difference in CoSQA/CodeBERT. The comparison quantifies what the signs of Top1 and best3 alone cannot resolve.
2. This analysis did not fit an adoption head or establish positive executable utility. Step B fixes the ENTR Top1 rewrite and measures the conditional contributions of generation confidence and the original-query reference under matched adoption counts and relative to NoRewrite.
3. The selected count and maximum aggregation both change in j. The contrast is specific to the two evaluation contracts and does not identify epistemic or aleatoric effects. Candidate coverage and saturation under maximum aggregation are also consistent with larger candidate opportunity and a smaller advantage over random selection.
4. The historical analysis called for checking the full related work on SSQR, QueryRoute and query adoption. The original H equivalence remains specific to ENTR−RAND20. No equivalence decision was made for the present D three-candidate contrast.

## Verification and delivery

- All 5 evaluation-contract tests passed, covering exact random maximum versus enumeration with ties and short pools, RR cutoffs, paired j cancellation and query-weighted group standard errors.
- All 3538 Query Original ENTR/ PROB / RAND20 Location selected for recurrence with two searchers 94216 Ranked text identity/Top100/Sortting and tie contract passed.
- The independent program did not import the master computing module to recalculate Fraction RR and compare; 7076 query xResearcher records, 122914 Second RR value, 134444 Sub-effect values and 320 The inter-area/summary fields were adopted. Maximum Effect Error `2.22e−16 `, inter-sectional/aggregated error `1.11e−16 `.
- Biggest difference in the original Bridge-by-Queary master indicator `2.22e−16 `； 32 The source code/input/protect file is passed by Hashi. Five official points, old H confirmed to be maintained; risk_certified = false, no new policy upgrade.
- This round master calculates approximately 9.21 seconds and independent audits are carried out separately; it is not the time to rerun the neurodrive.

Agreement:`PROTOCOL.json `query:`per_query.jsonl `. Full space:`RESULTS.json `Independent review:`AUDIT.json `Access:`experiments/rene_core_step_a_20261020/README.md `.
