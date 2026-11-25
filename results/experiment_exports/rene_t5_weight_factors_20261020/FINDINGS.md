# Matched B/G contrasts for two sets of T5 weights

ENTR B and G have positive point estimates under both weight conditions on CoSQA. The source-domain B estimate changes sign between the two conditions. After joint correction over 24 comparisons, all eight within-weight ENTR components and all eight ENTR/PROB direct weight contrasts are unresolved. The full decomposition adds evidence about weight sensitivity. Independence of B/G from the weights is not established.

With base T5, PROB has positive G in both domains after correction. The remaining 22 primary comparisons are unresolved, and no component has a negative direction after correction. ENTR−PROB B/G and their weight contrasts are secondary descriptive comparisons. Their 95% intervals span zero, leaving an entropy-specific component unestablished.

## Matching and estimands

- AdvTest and CoSQA each contain 512 queries in 512 repositories or linked groups. Within each domain both weights share queries, original-query RR, corpus, labels, retriever and position pool size. The study reuses the matched 512-query samples, not the full 2,048/1,490 samples.
- `local` uses the existing CQC T5 weights and `t5base` uses the existing base T5 weights. Both belong to the same T5 family. The paired contrast compares complete weight conditions and does not isolate CQC training.
- The original position pools contain 18,680 candidate RRs. Duplicate text retains its probability mass by position. ENTR and PROB use their original generation scores and break ties by position.
- k=1/2/3 is the selected count from the existing pool. Mean aggregation is the expected RR from uniform adoption within the selected set. Best-k is the candidate opportunity when choosing with labels. The comparison does not measure generation budget, automatic adoption gain or fusion gain.
- S(k,a)=rule RR−exact uniform random RR from the same pool; B=S(1,mean)−S(3,mean), G=S(3,mean)−S(3,best3), and J=B+G. These are arithmetic contrasts along the stated path.

## Full component and direct weight contrasts

The table shows two-sided cluster-t intervals with a Bonferroni correction for the full family of 24 comparisons. Machine-readable results also contain corrected independent-group bootstrap intervals. A direction requires agreement between the two interval methods and at least 20 nonzero groups.

| Data | weights | Rule | B (correction interval) | G (correction interval) |
|---|---|---|---|---|
| AdvTest | local | ENTR | -0.005239 [-0.019681, +0.009203] | +0.003479 [-0.006850, +0.013808] |
| AdvTest | local | PROB | -0.002036 [-0.016130, +0.012057] | +0.005028 [-0.005560, +0.015615] |
| AdvTest | t5base | ENTR | +0.001237 [-0.011825, +0.014298] | +0.005483 [-0.003986, +0.014952] |
| AdvTest | t5base | PROB | +0.007238 [-0.005819, +0.020295] | +0.009344 [+0.000480, +0.018208] |
| CoSQA | local | ENTR | +0.005815 [-0.013606, +0.025237] | +0.009869 [-0.001979, +0.021718] |
| CoSQA | local | PROB | +0.011552 [-0.007936, +0.031039] | +0.011625 [-0.000438, +0.023688] |
| CoSQA | t5base | ENTR | +0.018967 [-0.003537, +0.041472] | +0.012087 [-0.001941, +0.026114] |
| CoSQA | t5base | PROB | +0.014316 [-0.006975, +0.035607] | +0.018360 [+0.003595, +0.033124] |

| Data | Rule | Δ B: base−local (correction interval) | Δ G: base−local (correction interval) |
|---|---|---|---|
| AdvTest | ENTR | +0.006476 [-0.014390, +0.027341] | +0.002004 [-0.011352, +0.015360] |
| AdvTest | PROB | +0.009274 [-0.009496, +0.028045] | +0.004316 [-0.009061, +0.017693] |
| CoSQA | ENTR | +0.013152 [-0.016157, +0.042461] | +0.002217 [-0.016541, +0.020975] |
| CoSQA | PROB | +0.002764 [-0.026959, +0.032488] | +0.006734 [-0.012763, +0.026232] |

The direct weight intervals span zero and extend beyond ±0.005. Equivalence, invariance and small training effects are not established. CoSQA ENTR ΔB is +0.013152, with corrected interval [−0.016157,+0.042461]. The interval leaves meaningful weight differences unresolved.

## Same-pool ENTR advantage along the evaluation path

| Data | weights | S(1,mean) | S(3,mean) | S(3,best3) |
|---|---|---:|---:|---:|
| AdvTest | local | -0.005820 | -0.000581 | -0.004061 |
| AdvTest | t5base | +0.009123 | +0.007886 | +0.002403 |
| CoSQA | local | +0.021066 | +0.015251 | +0.005381 |
| CoSQA | t5base | +0.034753 | +0.015786 | +0.003699 |

In CoSQA, point estimates under both weights show a smaller ENTR advantage at larger selected counts and with best3 aggregation. Descriptive 95% intervals are positive for G under both weights and for B under base T5. The decisions for the family of 24 use the corrected intervals. The local CQC weight B estimate is negative in AdvTest, so the count component is reported by setting and weight.

## Role in the study

The matched analysis extends the earlier two-weight Top1/best3 envelope to separate count and aggregation components and provides paired direct weight contrasts. For CoSQA both sets of point estimates have the same direction, and the corresponding ENTR comparisons remain unresolved under correction. The frozen StaQC confirmation and the third-retriever extension answer their own specified comparisons.

These are post-hoc sensitivity analyses on previously used samples. The protocol was sealed locally before the new decomposition. Earlier Top1/best3 and the full-sample local-weight results were known. The reconstruction has not recovered the original authors' checkpoint, training CSV or production aggregation. The comparison does not establish effects across rewriter families, identify the effect of historical full training, or show that the complete adoption policy transfers.

## Verification and delivery

- All 9 contract tests passed. Replay reproduced 16,384 per-query historical references for Top1, best3, PROB, random selection and the original query with maximum difference 0.
- Independent `Fraction` RR calculations from accepted Top100 ranks and qrels verified 18,680 candidates and 2,048 original queries. An independent CDF calculation and checks of 131,072 factor fields, 65,536 paired fields, 384 summary means and 24 primary comparisons passed. Maximum factor and paired discrepancy: 2.37e−16.
- The analysis pairs weights before drawing 20,000 shared whole-group bootstrap resamples. It required no new training, generation, model passes or retrieval. Historical pointers and delivery hashes were retained.
- Results: `RESULTS.json`, `per_query.jsonl` and `paired_per_query.jsonl`. Tables: `factorial_cells.csv` (72 cells), `primary_contrasts.csv` (24 comparisons) and `all_summaries.csv` (384 summaries). Figures: `selection_curves` and `factor_contrasts` (PNG/PDF).
- The historical manuscript supplement was `paper/rene2027/t5_weight_factor_results.md`. See `REPLAY.md` for replay instructions.
