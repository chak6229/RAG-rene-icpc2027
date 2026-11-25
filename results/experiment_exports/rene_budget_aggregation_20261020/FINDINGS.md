# Same-pool factor decomposition: results and interpretation

2026-10-20. The analysis compares aggregation at fixed selected counts and compares 1/2/3 candidates under fixed aggregation for ENTR, PROB and exact random selection. The source setting contains 2048 queries in 594 repositories, and CoSQA development contains 1490 queries in 512 linked groups. Both reuse the local CQC T5 candidates and cached CodeBERT/BM25 rankings.

**Main finding: In CoSQA/CodeBERT, increasing the selected count and using label-assisted best-of-k aggregation both reduce the advantage of ENTR over exact random selection. Both components remain positive after the specified 24-comparison correction.** This separates the two changes that occurred together in the earlier comparison A.

## 1. Complete results at matched pools and selected counts

The table gives ΔMRR relative to the original query for CoSQA/CodeBERT. The original-query MRR is 0.51890040. Mean aggregation is the expected RR from uniformly adopting one selected candidate. Best-k is the maximum RR in the same set, using the relevance labels.

|Selection rule|Aggregation|1 candidate|2 candidates|3 candidates|
|---|---|---:|---:|---:|
|ENTR|mean|−0.01365851|−0.02007243|−0.02687819|
|PROB|mean|−0.01577175|−0.01898857|−0.02639018|
|exact random selection|mean|−0.04159434|−0.04159434|−0.04159434|
|ENTR|best-k|−0.01365851|+0.03780419|+0.06037845|
|PROB|best-k|−0.01577175|+0.03656354|+0.05942630|
|exact random selection|best-k|−0.04159434|+0.02778825|+0.05759904|

For ENTR, expanding the selected prefix decreases mean RR and increases best-k RR. A larger set offers more opportunities, and the aggregation rule determines how those opportunities enter the metric. The `first` reference always executes the first candidate, so its result is invariant to the selected count. Mean aggregation averages candidate RRs. Best-k uses relevance labels, which are unavailable for automatic selection. Executable RRF is evaluated separately under its own comparison.

## 2. Path decomposition of the earlier contrast A

Let S(k,a) be the MRR advantage of ENTR over exact random selection at the same k. For CoSQA/CodeBERT:

- S(1,mean)=+0.02793583。
- S(3,mean)=+0.01471615。
- S(3,best-k)=+0.00277941。

|Prespecified component|Estimate|Group bootstrap interval, Bonferroni-corrected over 24 comparisons|Corrected cluster-t interval|
|---|---:|---|---|
|B: reduction in advantage when the selected count increases from 1 to 3 under mean aggregation|+0.01321969|[+0.00088210,+0.02481866]|[+0.00088510,+0.02555428]|
|G: reduction in advantage when aggregation changes from mean to best-k at 3 candidates|+0.01193674|[+0.00490738,+0.01893323]|[+0.00488512,+0.01898835]|

The per-query identity gives J_exact=B+G=+0.02515642. The earlier analysis A used 20 random replays and gave J20=+0.02556783. The +0.00041141 difference is the finite-repeat discrepancy between RAND20 and the exact random best3 expectation. The earlier result remains as reported.

At a fixed count of 3, changing mean to best-k increases ENTR by +0.08725664 and exact random selection by +0.09919338. The extra +0.01193674 for exact random selection equals G. The smaller best-k advantage does not rule out an informative ENTR ordering for single selection.

This paired arithmetic decomposition follows the specified evaluation path. At k=1 the aggregation rules coincide. The components do not identify unique causal contributions or shares of the effect.

## 3. Interpretation across PROB and the second retriever

- For PROB in CoSQA/CodeBERT, B=+0.01061844 has a corrected interval spanning 0, while G=+0.01337689 has a corrected interval of [+0.00640977,+0.02045887]. The direct ENTR−PROB intervals for B and G both span 0. Different individual decisions do not establish an ENTR-specific mechanism.
- For source/CodeBERT, the corrected intervals for G are positive for ENTR and PROB, with estimates +0.00638704 and +0.00611574. The B directions are unresolved for both. A resolved component does not determine the interval for the original total contrast A.
- For source/BM25 and CoSQA/BM25, all corrected B/G intervals span 0. The finding concerns the tested data, retrievers and evaluation contract.

## 4. Role in the paper

Under a matched pool and selected count, choosing the best candidate with labels changes a rule's measured advantage over random selection. Expanding the selected prefix at fixed aggregation can also change that advantage. The two components quantify the changes that the signs of forced Top1 and best3 alone cannot separate.

This is a mechanism diagnostic within the specified replication configuration. It does not establish positive net gain for a new automatic policy, a distinct advantage of entropy over PROB, disentangled epistemic and aleatoric uncertainty, or population risk certification. Controlled CQC training contrasts and stability on a new sample are separate questions and were outside this run.

## 5. Verification and delivery

- All 10 evaluation-contract tests passed. The source and unlabeled actions for 3538 queries were sealed before the new factor calculations. The earlier A results were known, so the study remains a post-hoc diagnostic on previously used data.
- The independent audit used `Fraction` to recalculate 66306 candidate RRs and 7076 original-query RRs from rankings and qrels without importing the production metrics module. It cross-checked exact random best-k using a separate CDF formula.
- The checks covered 108 factor cells, 440 contrasts, 24 specified components and 28304 per-query decomposition identities. The maximum statistical discrepancy was 1.67e−15. All 39 frozen or protected files matched.
- All results reuse cached outputs without additional T5 generation, training or retrieval. k counts the selected candidates after generation. It does not lower generation or entropy-scoring cost. Replay time does not measure online latency.

See `READOUT.md`, `FACTORIAL.csv`, `CONTRASTS.csv` and `DECOMPOSITION.csv` for all four conditions and intervals, `AUDIT.json` for the numerical checks, and `FINAL_ACCEPTANCE.json` for export and hash verification. The historical manuscript source was `paper/rene2027/budget_aggregation_results.md`. The H confirmation and study configuration were retained. `new_confirmation=false`, `risk_certified=false`.
