# SSQR three-candidate aggregation and the public evaluator

The public JSON generation path writes multiple rewrites under the same URL. The public evaluator stores predictions as a URL-to-answers dictionary, so later rows overwrite earlier rows. When the public evaluator was run from the verified author Git blob, a synthetic example with 3 rewrites for one URL scored 0. Per-query best3 gave 1. The example demonstrates the public interface behavior and leaves the original production procedure unresolved.

Section 5.1 of the original paper describes selecting the top three and reporting the highest MRR. The public evaluation chain does not explicitly implement a per-query maximum RR or a maximum over the three position-wise MRRs. The three interpretations below are bounded diagnostics. The frozen main CoSQA comparison is retained.

The source and CoSQA D data had already been used. The diagnostics were frozen before opening new bridge effects. Queries with at least 3 words ensure that each profile has three actual positions. Shorter queries were excluded only from this supplementary subset, without padding with a zero or the original query. The main-experiment denominator is retained. See `PROTOCOL` for the subset counts.

| retriever | Data/Mainbridge queries | PVQary best 3: E-R | Three rank-wise MRR Take best: E−R | Open reader Last line caliber: E−R |
|---|---|---:|---:|---:|
| CodeBERT | source/2048 | -0.00435302 | +0.00156252 | +0.00172419 |
| BM25 | source/2048 | -0.00133742 | +0.00110986 | +0.00171944 |
| CodeBERT | cosqa/1490 | +0.00236800 | +0.02294983 | -0.00046317 |
| BM25 | cosqa/1490 | -0.00194770 | +0.00369643 | -0.00682315 |

The maximum of three rank-wise means is a non-smooth statistic. Selecting the best position using labels and treating it as a fixed policy would invalidate an ordinary paired interval. `RESULTS` gives a conservative contrast interval propagated from simultaneous Bonferroni cluster-t approximate bounds on 66 policy-by-rank means. Group intervals for the other two interpretations pair the queries. All three are diagnostics.

The reconstruction uses a per-query maximum RR as the three-candidate opportunity measure and records that the production aggregation path was not recovered. The last-row behavior in the public evaluator does not identify the original production pipeline. The synthetic diagnostic does not refute the original results.

Public evaluator blob: `a75b4944ca9d656641ed6214ce14c2709dca946e`. The source, protocol and synthetic execution evidence were retained. `RESULTS.json` provides the six profiles and PROB results.
