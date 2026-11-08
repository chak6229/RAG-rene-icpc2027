# UniXcoder extension results

This extension fixes the rewrite pools, relevance labels and code corpora from CoSQA D and the previously used StaQC sample. It applies the official zero-shot code-search configuration of UniXcoder-base: the encoder-only prefix, code256/nl128 truncation, mean pooling over non-padding tokens and unit-normalized dot products. No new training or T5 generation is performed.

The 12-comparison Bonferroni family contains B/G in the two conditions and paired same-query component differences relative to CodeBERT and BM25. Both group-bootstrap and cluster-t intervals are reported. Because these samples were used earlier, this is a model extension with specified comparisons, not new-sample confirmation.

| Condition | Component | Estimate | Adjusted cluster-t interval | Adjusted bootstrap interval |
|---|---|---:|---|---|
|CoSQA|B|+0.00552941|[-0.003520, +0.014579]|[-0.003408, +0.014635]|
|CoSQA|G|+0.00988737|[+0.004591, +0.015184]|[+0.004386, +0.015018]|
|StaQC|B|+0.00220045|[+0.000568, +0.003833]|[+0.000544, +0.003845]|
|StaQC|G|+0.00248619|[+0.001350, +0.003622]|[+0.001327, +0.003606]|

Interpret within-retriever components and direct cross-retriever differences using the full report. Best3 remains a label-assisted opportunity measure; it does not establish deployment gains, an ENTR-specific mechanism or universal performance across models. Full results, formulas, directions and validation are in `results/experiment_exports/rene_third_retriever_20261020/` and S10.
