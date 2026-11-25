# Third retriever: UniXcoder with fixed candidates

This study evaluates the selected-count component B and the aggregation component G for an additional code retriever and compares their values directly with CodeBERT and BM25.
We used the authors' public UniXcoder-base zero-shot retrieval weights and official code256/nl128, encoder-only and normalized mean-pooling settings. The study reused the T5 candidates and required no local training.

## Primary comparisons

B measures the reduction in ENTR's advantage over exact random selection from one candidate under mean aggregation to three candidates under mean aggregation. G measures the further reduction from three candidates under mean aggregation to label-assisted best3. Positive values indicate that the measured advantage shrinks. Negative values indicate that it grows.
In each of the two settings, the family includes UniXcoder's B/G and its paired B/G differences with CodeBERT and BM25. The 12 comparisons share a Bonferroni correction. Directional decisions require both corrected intervals to meet the rule.

|scene|Comparison|Size|estimate|Correction cluster-t|Correction bootstrap|Direction|
|---|---|---|---:|---|---|---|
|cosqa|UniXcoder|B|+0.00552941|[-0.003520, +0.014579]|[-0.003408, +0.014635]|unresolved|
|cosqa|UniXcoder|G|+0.00988737|[+0.004591, +0.015184]|[+0.004386, +0.015018]|positive|
|cosqa|UniXcoder_minus_CodeBERT|B|-0.00769027|[-0.020211, +0.004831]|[-0.020111, +0.004234]|unresolved|
|cosqa|UniXcoder_minus_CodeBERT|G|-0.00204936|[-0.009969, +0.005870]|[-0.010016, +0.005915]|unresolved|
|cosqa|UniXcoder_minus_BM25|B|+0.00720715|[-0.003308, +0.017723]|[-0.002952, +0.018155]|unresolved|
|cosqa|UniXcoder_minus_BM25|G|+0.00877224|[+0.002344, +0.015201]|[+0.002100, +0.014638]|positive|
|staqc|UniXcoder|B|+0.00220045|[+0.000568, +0.003833]|[+0.000544, +0.003845]|positive|
|staqc|UniXcoder|G|+0.00248619|[+0.001350, +0.003622]|[+0.001327, +0.003606]|positive|
|staqc|UniXcoder_minus_CodeBERT|B|-0.00105645|[-0.003479, +0.001366]|[-0.003549, +0.001276]|unresolved|
|staqc|UniXcoder_minus_CodeBERT|G|-0.00112333|[-0.002820, +0.000573]|[-0.002839, +0.000576]|unresolved|
|staqc|UniXcoder_minus_BM25|B|+0.00331365|[+0.001269, +0.005358]|[+0.001149, +0.005387]|positive|
|staqc|UniXcoder_minus_BM25|G|+0.00182083|[+0.000419, +0.003222]|[+0.000380, +0.003179]|positive|

## Matched-count selection advantage and original-query reference

The table gives descriptive point estimates. Individual 95% intervals are in `ALL_EFFECTS.csv` and `RESULTS.json`. The best3 estimates use relevance labels and measure candidate opportunity.

|scene|retriever|ENTR− Random: 1 mean|3 mean|3 best3|Top1− Original query|best3− Original query|
|---|---|---:|---:|---:|---:|---:|
|cosqa|UniXcoder|+0.01473342|+0.00920401|-0.00068337|-0.00108584|+0.06026182|
|cosqa|CodeBERT|+0.02793583|+0.01471615|+0.00277941|-0.01365851|+0.06037845|
|cosqa|BM25|-0.00169972|-0.00002198|-0.00113711|-0.02035096|+0.00977629|
|staqc|UniXcoder|+0.00413588|+0.00193543|-0.00055076|-0.00253959|+0.02014645|
|staqc|CodeBERT|+0.00635826|+0.00310135|-0.00050817|-0.00836347|+0.02147861|
|staqc|BM25|-0.00127785|-0.00016466|-0.00083002|-0.00931813|+0.00563239|

## Interpretation and scope

- CoSQA: Some components have a positive direction after correction. The remaining directions are unresolved, so the evidence does not establish both components in this setting.
- StaQC: B and G are both positive after correction, supporting the selected-count and aggregation decomposition for the tested UniXcoder setting.

Retriever differences are assessed using paired intervals on the same queries. Differences in individual directional decisions do not establish a paired difference. Components measure changes in selection advantage. PROB and ENTR−PROB are descriptive controls and component magnitudes do not rank overall retriever or scoring performance.
The rewrite pools, code text and labels are fixed. The complete retrieval systems vary jointly in training, pooling, normalization and official input lengths. UniXcoder uses a RoBERTa architecture, and the same rewriter family is used throughout this extension.
Both data sets had already been used. The extension was fixed before inspecting the new retriever effects, reused the existing samples and had no external preregistration. CoSQA contains 1490 queries in 512 groups and StaQC contains 13312 queries in 13312 groups. Sample sizes were fixed independently of these outcomes. Power was assessed using variance scenarios from existing retrievers, which do not guarantee precision for UniXcoder.
UniXcoder uses CodeSearchNet NL–PL pretraining. Unknown pretraining exposures and semantic clones remain possible. StaQC's single known positive differs from the Bing-query and function-corpus setting. The extension supports conclusions about the evaluated retriever and settings.

## Verification and replay

Before opening the new effects, we ran 9 contract tests, checked the official examples and independent model forward passes, and audited the complete vector-to-Top100 path. We then independently replayed RR with `Fraction`, exact random expectations using a separate CDF expression, and paired statistics. Counts and errors are in `PREOPEN_AUDIT.json` and `FINAL_ACCEPTANCE.json`.
`MODEL_PROVENANCE.json` records the upstream commit, Git/LFS hashes, license and mirror transfer. `DESIGN.json` binds the inputs, code, weights and protected files.
The full historical replay requires the bound local data and rankings and uses `python -m experiments.rene_third_retriever_20261020.audit --after`. The `run` module performs new inference. The recorded effects were opened once.
`risk_certified=false`. The original confirmation and production configuration remained as recorded.
