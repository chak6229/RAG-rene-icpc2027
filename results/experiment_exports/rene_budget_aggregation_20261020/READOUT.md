# Same-pool decomposition of selected count and aggregation

The source sample contains 2048 queries in 594 repositories. CoSQA development contains 1490 queries in 512 linked groups. Both use the frozen local CQC T5 candidates and both retrievers' caches. The selected count k is determined after generating and scoring all positions. The post-hoc paired diagnostic adds no generation, training, retrieval or H evaluation.

Mean aggregation gives the expected RR from uniformly adopting one selected candidate. `max` is the label-assisted best-of-k opportunity. `first` always executes the first candidate in the prefix. Exact random selection integrates the uniform distribution over distinct positions without replacement. Positions with identical text retain their probability mass. Mean aggregation averages RRs, and `max` depends on labels.

## Aggregation at fixed counts and selected count under fixed aggregation

The table gives the selection rule's advantage over exact random selection at the same k. Each cell lists mean / max. See `FACTORIAL.csv` for the full MRR values, differences from the original query and individual intervals. The advantage under `first` is constant across k.

|Data / Retrieval|Rule|k=1 mean / max|k=2 mean / max|k=3 mean / max|
|---|---|---:|---:|---:|
|source / CodeBERT|ENTR|+0.00083866 / +0.00083866|+0.00232108 / -0.00232957|+0.00203055 / -0.00435649|
|source / CodeBERT|PROB|+0.00304005 / +0.00304005|+0.00148662 / -0.00215254|+0.00243968 / -0.00367606|
|source / BM25|ENTR|-0.00216420 / -0.00216420|-0.00147598 / -0.00209884|-0.00031800 / -0.00135795|
|source / BM25|PROB|-0.00330705 / -0.00330705|-0.00121710 / -0.00203988|-0.00047485 / -0.00133460|
|cosqa / CodeBERT|ENTR|+0.02793583 / +0.02793583|+0.02152191 / +0.01001594|+0.01471615 / +0.00277941|
|cosqa / CodeBERT|PROB|+0.02582259 / +0.02582259|+0.02260577 / +0.00877529|+0.01520416 / +0.00182727|
|cosqa / BM25|ENTR|-0.00169972 / -0.00169972|+0.00244249 / +0.00188731|-0.00002198 / -0.00113711|
|cosqa / BM25|PROB|-0.00062378 / -0.00062378|+0.00106614 / -0.00085339|-0.00014547 / -0.00170806|

## Specified decomposition path

S is the advantage over exact random selection at the same k. B=S(1,mean)−S(3,mean), G=S(3,mean)−S(3,max), and J_exact=B+G. A positive B or G means the original advantage shrinks across the specified step. A negative value means it grows.

The specified family has 24 comparisons from four data-by-retriever settings, the ENTR and PROB rules and their direct difference ENTR−PROB, and the B and G components. The table shows Bonferroni-corrected group bootstrap intervals. The full results also contain corrected cluster-t intervals. This analysis adds no equivalence decision.

|Data / Retrieval|Rule|1/3 change in fixed-mean|24 Correction slots|G fixed 3 mean→max variation|24 Correction slots|J_exact|
|---|---|---:|---|---:|---|---:|
|source / CodeBERT|ENTR|-0.00119189|[-0.00793894, +0.00546475]|+0.00638704|[+0.00195385, +0.01072038]|+0.00519515|
|source / CodeBERT|PROB|+0.00060037|[-0.00568646, +0.00742975]|+0.00611574|[+0.00137037, +0.01093849]|+0.00671611|
|source / CodeBERT|ENTR_minus_PROB|-0.00179226|[-0.00805951, +0.00403014]|+0.00027130|[-0.00315565, +0.00374282]|-0.00152096|
|source / BM25|ENTR|-0.00184620|[-0.00532895, +0.00136783]|+0.00103996|[-0.00146605, +0.00352039]|-0.00080625|
|source / BM25|PROB|-0.00283219|[-0.00645139, +0.00082012]|+0.00085974|[-0.00168482, +0.00351185]|-0.00197245|
|source / BM25|ENTR_minus_PROB|+0.00098599|[-0.00223506, +0.00422136]|+0.00018021|[-0.00128016, +0.00147299]|+0.00116620|
|cosqa / CodeBERT|ENTR|+0.01321969|[+0.00088210, +0.02481866]|+0.01193674|[+0.00490738, +0.01893323]|+0.02515642|
|cosqa / CodeBERT|PROB|+0.01061844|[-0.00071012, +0.02122361]|+0.01337689|[+0.00640977, +0.02045887]|+0.02399533|
|cosqa / CodeBERT|ENTR_minus_PROB|+0.00260125|[-0.00806805, +0.01277867]|-0.00144016|[-0.00679460, +0.00421164]|+0.00116109|
|cosqa / BM25|ENTR|-0.00167774|[-0.00787790, +0.00396237]|+0.00111513|[-0.00249380, +0.00471802]|-0.00056260|
|cosqa / BM25|PROB|-0.00047832|[-0.00624677, +0.00497573]|+0.00156260|[-0.00213394, +0.00545720]|+0.00108428|
|cosqa / BM25|ENTR_minus_PROB|-0.00119942|[-0.00592436, +0.00301265]|-0.00044746|[-0.00277231, +0.00215915]|-0.00164688|

## RAND20 accounting in the earlier analysis A

The earlier A is retained: J20=J_exact+(RAND20_best3−RAND_exact_best3). The primary decomposition here uses exact random selection and reports the finite-repeat discrepancy separately.

|Data / Retrieval|ENTR J20|ENTR J_exact|RAND20− Accurate Random best3|
|---|---:|---:|---:|
|source / CodeBERT|+0.00519168|+0.00519515|-0.00000347|
|source / BM25|-0.00082678|-0.00080625|-0.00002053|
|cosqa / CodeBERT|+0.02556783|+0.02515642|+0.00041141|
|cosqa / BM25|+0.00024798|-0.00056260|+0.00081058|

## Interpretation and scope

- At k1 all three aggregation rules coincide. Under `first`, increasing k leaves the executed action unchanged. The change from mean to `max` adds the ability to use labels to select the candidate. This differs from executable automatic selection.
- B/G form an arithmetic decomposition along a specified reference path. The decomposition does not identify independent causal mechanisms or their shares.
- Report selection advantage over exact random selection separately from net gain over the original query. Assess ENTR versus PROB by their direct difference using paired queries.
- The bootstrap used 10000 linked-group resamples and query weighting. The development and source samples had already been used. The original H confirmation and study configuration were retained. `risk_certified=false`.

`RESULTS.json` contains all machine-readable results. `FACTORIAL.csv` contains 108 factor cells including `first` references, and 72 mean/max cells. `CONTRASTS.csv` contains counts at fixed aggregation and aggregation at fixed counts. `DECOMPOSITION.csv` contains the 24 specified components. See `AUDIT.json` for independent Fraction/CDF replay and `FINAL_ACCEPTANCE.json` for the scope of acceptance.
