# StaQC confirmation of selected-count and aggregation directions

After observing the development findings, we fixed B/G comparisons for the independently sourced StaQC Python single-code-answer condition. Exact/normalized query and code overlap, source-question identity and repeated-target associations were audited. A fixed hash selected one query per group, giving 13,312 queries/13,312 groups and a fixed 16,384-snippet corpus.

The study retained the same CQC T5 and SSQR pooler CodeBERT, generated all positions once, and compared ENTR, PROB and exact random selection for k=1/2/3 with mean and best-of-k aggregation. The two primary comparisons use Bonferroni-adjusted cluster-t and group-bootstrap intervals. PROB/BM25 remain prespecified descriptive controls. Selected count k does not measure generation calls or cost. Neither aggregation adds the original query as a safety option.

A variance-matching issue was detected during label-free candidate generation. The initial plan was withdrawn and the revision recorded before new effects were inspected. The final sample was resealed using a matching single-query/marginal-variance upper bound from old development data. There was no external preregistration.

| Primary component | Estimate | Adjusted cluster-t interval | Adjusted bootstrap interval | Decision |
|---|---:|---|---|---|
|ENTR_B_budget_mean_1_to_3|+0.00325690|[+0.001721, +0.004792]|[+0.001740, +0.004814]|Positive|
|ENTR_G_aggregation_at3_mean_to_max|+0.00360952|[+0.002577, +0.004642]|[+0.002576, +0.004649]|Positive|

Both adjusted interval methods support positive B and G. These directions confirm a reduction in relative position-selection advantage after changing selected count or aggregation in this condition.

| Retriever | S(1,mean) | S(3,mean) | S(3,best3) | Top1 versus original | best3 versus original |
|---|---:|---:|---:|---:|---:|
|CodeBERT|+0.00635826|+0.00310135|-0.00050817|-0.00836347|+0.02147861|
|BM25|-0.00127785|-0.00016466|-0.00083002|-0.00931813|+0.00563239|

These reference values are descriptive. The extension concerns mined Stack Overflow titles and code snippets, rather than a second independent CoSQA confirmation. Best-of-k remains a label-assisted opportunity measure; it does not establish positive automatic-adoption gains or a new uncertainty method.

Validation covered upstream Git objects and eligible groups, old-data power planning, complete Fraction RR and CDF random expectations, adjusted intervals, and actual T5 FP64 entropy/regeneration and CodeBERT-vector spot checks. Evidence is in `results/experiment_exports/rene_factor_confirmation_20261020/` and S9. CoSQA development and StaQC confirmation retain their own comparison families; separate interval signs do not establish a between-source effect difference.
