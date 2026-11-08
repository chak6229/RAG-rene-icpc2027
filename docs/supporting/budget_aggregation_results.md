# Same-pool selected-count and aggregation analysis

This post-hoc analysis uses the existing local CQC T5 candidates and CodeBERT/BM25 rankings on AdvTest (2,048 queries/594 groups) and CoSQA D (1,490 queries/512 groups). ENTR and PROB select nested prefixes of 1/2/3 positions; exact random selection samples original positions uniformly without replacement. Duplicate texts retain their original sampling probability. Mean aggregation is expected RR under uniform adoption within the selected set. Best-of-k aggregation uses relevance labels; first-candidate execution is a structural anchor. All generation and scoring precede the analysis, so k denotes downstream selected count.

Earlier endpoint results were already known. Inputs, finite comparisons and actions using no labels were sealed before the new factor effects were inspected. Estimates weight queries equally and use 10,000 common group-bootstrap resamples together with cluster-t intervals. The 24 prespecified components receive Bonferroni correction; other curves are descriptive.

## CoSQA and CodeBERT results

The ENTR position-selection advantage over exact random selection decreases from 0.02793583 for one candidate to 0.01471615 for three candidates under mean aggregation, then to 0.00277941 under best3. Holding mean aggregation fixed, increasing selected count gives B=0.01321969, with adjusted bootstrap interval [0.00088210,0.02481866]. Holding three candidates fixed and changing aggregation gives G=0.01193674, with adjusted interval [0.00490738,0.01893323]. The per-query components sum to J_exact=0.02515642.

With three candidates fixed, changing mean to best3 increases ENTR MRR by 0.08725664 and exact-random MRR by 0.09919338. Label-assisted selection thus gives the random set a larger increment, reducing the relative ENTR advantage. Small between-selector differences under best-of-k aggregation can coexist with a useful single-selection ordering signal.

| Selector and aggregation | k=1 ΔMRR | k=2 ΔMRR | k=3 ΔMRR |
|---|---:|---:|---:|
|ENTR mean|−0.01365851|−0.02007243|−0.02687819|
|PROB mean|−0.01577175|−0.01898857|−0.02639018|
|Exact random mean|−0.04159434|−0.04159434|−0.04159434|
|ENTR best-of-k|−0.01365851|+0.03780419|+0.06037845|
|PROB best-of-k|−0.01577175|+0.03656354|+0.05942630|
|Exact random best-of-k|−0.04159434|+0.02778825|+0.05759904|

These changes use the original query as reference, whereas S uses exact random selection. PROB has a similar aggregation component, and direct ENTR−PROB component differences are unresolved. The pattern is therefore not established as an entropy-specific uncertainty mechanism. AdvTest/CodeBERT also has a positive aggregation component, while its selected-count component is unresolved. Adjusted component intervals in both BM25 conditions cross zero.

## Relationship to earlier analyses

The earlier CoSQA/CodeBERT endpoint difference used RAND20 and was 0.02556783. Exact random selection gives 0.02515642; the difference of 0.00041141 reflects finite RAND20 replay error. The historical result and frozen H comparison retain their own definitions and values.

B/G provide arithmetic accounting along the specified reference path, rather than unique causal attribution. At k=1 the aggregation rules coincide. Best-of-k uses relevance labels and is not an automatic-execution capability; mean aggregation is not document-list fusion. This analysis does not introduce a new uncertainty algorithm, establish stable adoption gains or certify risk. Training-configuration effects, new-sample confirmation and policy utility have separate evidence.

Recorded validation covers 10 scientific tests, independent Fraction/CDF and statistical replay, 108 factor cells and 24 adjusted components. The source status remains new_confirmation=false and risk_certified=false. Exact results are in `results/experiment_exports/rene_budget_aggregation_20261020/`; the main reader table is S8.
