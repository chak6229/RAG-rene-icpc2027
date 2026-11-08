# Integrated replication and controlled-extension results

Generated from sealed machine results. Primary confirmation, predefined descriptive estimates, and post-hoc diagnostics retain their original roles. Retrieval effects are MRR@100 differences; review rates use their stated denominators. Populations are not pooled. Updated 2026-10-21.

## RQ1: external replication of position selection

|Population|Comparison|ΔMRR|Group-bootstrap 95% CI|Evidence role|
|---|---|---:|---|---|
|7718 queries / 1963 groups|CodeBERT ENTR-RAND20 best3|-0.000837|[-0.003255, +0.001541]|Single frozen comparison; both 95% intervals inside prespecified ±0.005|

The result concerns the declared per-query best-of-three opportunity contract in the reconstructed implementation. It does not establish identity with unavailable author weights or a causal effect of query provenance.

## RQ1: separating selected count and aggregation

Let S(k,a) be the selector's mean RR@100 minus exact uniform selection over the same insertion positions. The selected-count component is B=S(1,mean)-S(3,mean); the aggregation component is G=S(3,mean)-S(3,best3). Their sum J follows this specified arithmetic path. Positive values mean the selector's relative advantage decreases along that step. The pool is held fixed: k counts selected candidates, not generation calls. Mean is expected RR under uniform adoption within the selected set; best-k uses relevance labels. Exact random sampling is without replacement over positions and retains duplicate-text probability mass.

The development analysis covers ENTR, PROB and exact random, k=1/2/3 and mean/best-k. The 24-comparison family includes both selectors' B/G and their direct differences across four domain/retriever cells. The table shows ENTR; T8 contains all selectors and contrasts.

|Dataset / retriever|Component|Estimate|Adjusted cluster-t CI (24)|Decision|
|---|---|---|---|---|
|source / CodeBERT|B|-0.001192|[-0.007717, +0.005333]|unresolved|
|source / CodeBERT|G|+0.006387|[+0.001930, +0.010845]|positive|
|source / BM25|B|-0.001846|[-0.005161, +0.001469]|unresolved|
|source / BM25|G|+0.001040|[-0.001416, +0.003496]|unresolved|
|cosqa / CodeBERT|B|+0.013220|[+0.000885, +0.025554]|positive|
|cosqa / CodeBERT|G|+0.011937|[+0.004885, +0.018988]|positive|
|cosqa / BM25|B|-0.001678|[-0.007614, +0.004258]|unresolved|
|cosqa / BM25|G|+0.001115|[-0.002463, +0.004693]|unresolved|

CoSQA/CodeBERT resolves both components: B=+0.013220 and G=+0.011937, J_exact=+0.025156. Source/CodeBERT resolves G but not B; the original BM25 cells remain unresolved. ENTR-PROB component differences are unresolved. This refines the earlier combined contrast (+0.025568, using RAND20 at the best3 endpoint); that different estimand is retained in T2. The path decomposition is not a unique causal attribution.

## RQ1: frozen confirmation in a new setting

StaQC Python provides 13,312 queries in 13,312 associated groups and 16,384 snippets after source/license, exposure and overlap checks. It represents mined Stack Overflow titles paired with single accepted snippets. The locally sealed design fixes two CodeBERT ENTR primary comparisons; both adjusted cluster-t and bootstrap intervals must be positive. PROB and BM25 remain prespecified descriptive controls.

|Component|Estimate|Adjusted cluster-t CI (2)|Adjusted bootstrap CI (2)|
|---|---|---|---|
|B|+0.003257|[+0.001721, +0.004792]|[+0.001740, +0.004814]|
|G|+0.003610|[+0.002577, +0.004642]|[+0.002576, +0.004649]|

Both components confirm the directional finding in this new setting. Their magnitude is approximately 0.0033 and 0.0036 MRR, not an effect at least 0.005 or an executable deployment gain. The earlier CoSQA equivalence margin belongs to a different comparison. BM25 gives B=-0.001113 and G=+0.000665 with pointwise descriptive intervals respectively below and above zero. PROB behaves similarly to ENTR. The frozen design was amended before new effect access and is locally documented, not externally preregistered. StaQC is not a fresh CoSQA confirmation.

## RQ3: third-retriever extension

The public UniXcoder-base zero-shot configuration reuses consumed CoSQA and StaQC pools. Twelve comparisons jointly cover its B/G and paired differences from CodeBERT and BM25. These are model-extension estimates on existing samples.

|Dataset|Comparison|Component|Estimate|Adjusted cluster-t CI (12)|Decision|
|---|---|---|---|---|---|
|cosqa|UniXcoder|B|+0.005529|[-0.003520, +0.014579]|unresolved|
|cosqa|UniXcoder|G|+0.009887|[+0.004591, +0.015184]|positive|
|cosqa|UniXcoder - CodeBERT|B|-0.007690|[-0.020211, +0.004831]|unresolved|
|cosqa|UniXcoder - CodeBERT|G|-0.002049|[-0.009969, +0.005870]|unresolved|
|cosqa|UniXcoder - BM25|B|+0.007207|[-0.003308, +0.017723]|unresolved|
|cosqa|UniXcoder - BM25|G|+0.008772|[+0.002344, +0.015201]|positive|
|staqc|UniXcoder|B|+0.002200|[+0.000568, +0.003833]|positive|
|staqc|UniXcoder|G|+0.002486|[+0.001350, +0.003622]|positive|
|staqc|UniXcoder - CodeBERT|B|-0.001056|[-0.003479, +0.001366]|unresolved|
|staqc|UniXcoder - CodeBERT|G|-0.001123|[-0.002820, +0.000573]|unresolved|
|staqc|UniXcoder - BM25|B|+0.003314|[+0.001269, +0.005358]|positive|
|staqc|UniXcoder - BM25|G|+0.001821|[+0.000419, +0.003222]|positive|

UniXcoder extends the aggregation result to both settings and the selected-count result to StaQC. Three paired differences from BM25 are positive after correction; all four differences from CodeBERT are unresolved. Unresolved differences do not establish equivalence. Retriever dependence is assessed with direct paired contrasts, not inferred from one significant and one nonsignificant result.

## RQ2: fixed-action adoption with matched supervision

M0: common evidence; MT: M0 plus generation confidence; MR: M0 plus original-query reference; MRT: both. Fixed20 adopts the top 20% of eligible actions within each frozen OOF fold. The denominator remains all 1,490 development queries.

|Retriever|Head|ΔMRR|Group-bootstrap 95% CI|Harmful / adopted|
|---|---|---:|---|---|
|CodeBERT|M0|+0.001341|[-0.001885, +0.004670]|53/296|
|CodeBERT|MT|-0.000667|[-0.003665, +0.002503]|55/296|
|CodeBERT|MR|+0.001080|[-0.002494, +0.004705]|75/296|
|CodeBERT|MRT|+0.001363|[-0.002071, +0.004994]|81/296|
|BM25|M0|-0.001049|[-0.001636, -0.000510]|94/274|
|BM25|MT|-0.001179|[-0.001825, -0.000570]|93/274|
|BM25|MR|-0.001906|[-0.003140, -0.000942]|92/274|
|BM25|MRT|-0.002253|[-0.003542, -0.001223]|96/274|

At fixed adoption count, CodeBERT MRT-M0 is +0.00002276, a joint-feature contrast. Generation-confidence increments are assessed through MT-M0 and MRT-MR; these do not establish extra adoption utility. Original-query-reference increments use MR-M0 and MRT-MT. Position ranking and fixed-action adoption are different decision tasks; their significance patterns do not constitute a tested stage interaction.

## RQ2 extension: executable multi-candidate fusion

All rows use the same native-ENTR candidate pool. RRF3 merges the three candidate lists; RRF4 also includes the original query. RRF constant 60, equal unique-input weights, Top100 inputs and exact rational ordering were fixed before these fusion effects. Best3 uses relevance labels and is not executable or a bound on fused rankings.

|Retriever|Policy|ΔMRR|Group-bootstrap 95% CI|Harmful queries / 1490|
|---|---|---:|---|---:|
|CodeBERT|ENTRTop1|-0.013659|[-0.022697, -0.004789]|388|
|CodeBERT|RRF3|-0.010100|[-0.019718, -0.000395]|352|
|CodeBERT|RRF4|-0.006308|[-0.013563, +0.000681]|284|
|CodeBERT|ENTRBest3|+0.060378|[+0.051462, +0.069387]|125|
|BM25|ENTRTop1|-0.020351|[-0.025713, -0.015221]|555|
|BM25|RRF3|-0.003570|[-0.006057, -0.001095]|411|
|BM25|RRF4|-0.002024|[-0.003760, -0.000412]|349|
|BM25|ENTRBest3|+0.009776|[+0.005181, +0.014688]|319|

The four fusion-versus-original effects do not establish positive utility; after four-effect correction, only BM25 RRF3 remains negative. BM25 fusion improves over forced Top1 in paired secondary comparisons. Including the original query has a positive point trend over RRF3, with intervals crossing zero for both retrievers. Unique retrieval inputs per query are 1 for NoRewrite, 1 for standalone Top1, 2.9799 for RRF3, and 3.9799 for RRF4; these counts are not measured latency.

## Frozen full-H selection, verification and fallback

These predefined descriptive stages use all 8,530 H queries / 1,963 groups. They are not the 7,718-query primary sample and are not interchangeable with development results. No H outcomes were reopened to build these tables.

|Retriever|Stage|ΔMRR|Group-bootstrap 95% CI|Harmful / adopted|Positive-gain retention|CE pairs|
|---|---|---:|---|---|---:|---:|
|CodeBERT|ENTR_top1|-0.021105|[-0.025139, -0.017047]|2328/8530|42.52%|0|
|CodeBERT|HGB_gain_control|-0.002876|[-0.005663, -0.000055]|1277/5305|27.02%|0|
|CodeBERT|RIDGE_gain|+0.000079|[-0.002245, +0.002410]|1317/6034|23.04%|0|
|CodeBERT|RIDGE_verified|+0.000847|[-0.001586, +0.003265]|1280/5890|24.35%|17056|
|CodeBERT|RIDGE_risk|-0.000178|[-0.000551, +0.000195]|60/1006|0.77%|17056|
|BM25|ENTR_top1|-0.014435|[-0.016920, -0.012049]|2929/7955|39.10%|0|
|BM25|HGB_gain_control|-0.002715|[-0.003930, -0.001543]|853/2615|11.80%|0|
|BM25|RIDGE_gain|-0.002079|[-0.002771, -0.001403]|1072/3559|5.63%|0|
|BM25|RIDGE_verified|-0.001717|[-0.002281, -0.001154]|906/2926|4.43%|17056|
|BM25|RIDGE_risk|+0.000000|[+0.000000, +0.000000]|0/0|0.00%|17056|

Positive-gain retention is defined against the same candidate pool with an original-query safe arm. It is not a fraction of net mean gain, and it is not used for the new RRF rankings. Risk-stage empirical harm and cost accounting do not issue a population risk certificate.

## RQ3: matched T5-weight sensitivity

Each domain uses its historical one-query-per-group 512-query sample, identical insertion-position and retrieval contracts. The additional base-weight rankings have now been persisted and independently audited. Both weights belong to T5-base; these are not independent model families or fresh confirmations.

|Dataset|T5 weights|Top1−original|95% CI|Best3−original|95% CI|
|---|---|---:|---|---:|---|
|source|local|-0.030252|[-0.045216, -0.016203]|+0.029989|[+0.018285, +0.042346]|
|source|t5base|-0.013366|[-0.024340, -0.002626]|+0.038525|[+0.025869, +0.051803]|
|cosqa|local|-0.020197|[-0.036746, -0.004346]|+0.061308|[+0.046028, +0.076993]|
|cosqa|t5base|-0.042703|[-0.062963, -0.022992]|+0.056700|[+0.039201, +0.074852]|

All four best3 gains remain positive under the eight-effect Bonferroni sensitivity. Two of four Top1 intervals cross zero after that correction: source/t5-base and CoSQA/local. CoSQA/t5-base single-candidate ENTR outperforms exact uniform selection by +0.03475321, 95% [0.01882089,0.05044681], while remaining below the original query. Therefore below-original utility cannot by itself diagnose an uninformative entropy ranking.

The complete matched B/G design covers both selectors under both weights, plus paired base-minus-local differences on the same 512 queries per domain. T11 exports all 24 primary comparisons; the following ENTR rows retain the family correction.

|Dataset|Weight / difference|Component|Estimate|Adjusted cluster-t CI (24)|Decision|
|---|---|---|---|---|---|
|source|local|B|-0.005239|[-0.019681, +0.009203]|unresolved|
|source|local|G|+0.003479|[-0.006850, +0.013808]|unresolved|
|source|t5base|B|+0.001237|[-0.011825, +0.014298]|unresolved|
|source|t5base|G|+0.005483|[-0.003986, +0.014952]|unresolved|
|source|base_minus_local|B|+0.006476|[-0.014390, +0.027341]|unresolved|
|source|base_minus_local|G|+0.002004|[-0.011352, +0.015360]|unresolved|
|cosqa|local|B|+0.005815|[-0.013606, +0.025237]|unresolved|
|cosqa|local|G|+0.009869|[-0.001979, +0.021718]|unresolved|
|cosqa|t5base|B|+0.018967|[-0.003537, +0.041472]|unresolved|
|cosqa|t5base|G|+0.012087|[-0.001941, +0.026114]|unresolved|
|cosqa|base_minus_local|B|+0.013152|[-0.016157, +0.042461]|unresolved|
|cosqa|base_minus_local|G|+0.002217|[-0.016541, +0.020975]|unresolved|

All eight within-weight ENTR component estimates and all eight ENTR/PROB paired weight differences are unresolved after correction. PROB G under T5-base is positive in both domains; the full family has two positive and 22 unresolved comparisons. On CoSQA both weights have positive ENTR B/G point estimates, while source B changes point-estimate sign. Complete coverage characterizes precision rather than establishing weight invariance or equivalence. Both conditions use T5-base; this comparison does not isolate the causal effect of CQC training.

## RQ3: separate new-sample paired T5-weight confirmation

A new, locally sealed confirmation uses 25,600 previously unused StaQC query/connection groups and a fixed 32,768-snippet corpus. This is separate from the preceding 512-query-per-domain sensitivity and from the 13,312-group StaQC confirmation above. Both T5 weights share queries, targets, corpus and frozen CodeBERT retrieval; the six-comparison family covers each weight's ENTR B/G and paired t5base-minus-local B/G differences.

|Weight condition|Component|Estimate|Six-comparison adjusted cluster-t interval|Decision|
|---|---|---:|---|---|
|local|B|+0.002803|[+0.001565, +0.004041]|positive|
|local|G|+0.003241|[+0.002409, +0.004073]|positive|
|t5base|B|+0.004282|[+0.002941, +0.005624]|positive|
|t5base|G|+0.003520|[+0.002636, +0.004404]|positive|
|t5base−local|B|+0.001479|[−0.000325, +0.003282]|unresolved|
|t5base−local|G|+0.000279|[−0.000909, +0.001467]|unresolved|

All four within-weight directions also pass the shared 20,000-draw group-bootstrap rule. The direct weight differences remain unresolved under both interval methods; this confirms direction across the two tested T5 weights in this new StaQC setting, not equivalence or invariance. The 0.005 difference-precision planning target is not an equivalence margin. Full paired intervals, source checks and independent replays are in `results/rene_weight_confirmation_20261021/`.

## RQ2 diagnostic: review agreement and weighted events

Two Python-proficient reviewers returned assessments for 128 queries (120 groups), with 383 unique rewrites and 327 code panels each. Agreement describes final submitted labels. Original eight-stratum design weights estimate query events in the finite H8530 frame. Displayed-item agreement and frame-weighted rates use different units.

|Assessment|Items|Exact agreement|Kappa|Scope|
|---|---|---|---|---|
|semantics|383|76.5%|0.525|nominal|
|relevance_including_unknown|327|53.8%|0.371|nominal|
|relevance_ordinal_judged_subset|260|66.9%|0.569|79.5% coverage; linear weighted|
|clarity|128|72.7%|0.029|nominal|

|Weighted query event|Estimate|Conservative finite-frame 95% CI|
|---|---|---|
|any_definite_semantic_issue|35.01%|[9.86%, 69.60%]|
|any_possible_semantic_issue|75.46%|[39.60%, 95.38%]|
|any_uncertain_semantics|26.87%|[5.64%, 62.58%]|
|verified_action_definite_drift|4.16%|[0.09%, 36.57%]|
|verified_harm_with_preserved_meaning|12.91%|[7.62%, 40.32%]|
|missed_preserved_positive_opportunity|14.28%|[4.28%, 45.90%]|
|displayed_alternative_relevant|11.99%|[1.29%, 46.18%]|
|known_target_both_incomplete|45.09%|[15.66%, 77.56%]|
|rater_A_any_semantic_issue|75.42%|[39.55%, 95.37%]|
|rater_B_any_semantic_issue|37.56%|[10.91%, 72.04%]|

There are 106 rewrite disagreements (90 semantic-class and 16 additional type disagreements), 151 code disagreements and 35 clarity disagreements. All remain recorded with both judgments. Unknown code labels number two for A and 67 for B; the mutually judged subset's kappa is reported with its coverage. Query-event indicators overlap; finite-frame intervals omit label uncertainty and do not identify unique causal shares. Evidence concerns displayed panels rather than a complete new relevance assessment. Detailed disagreement analysis and examples are in mechanism_review_return_results.md and review_disagreement_discussion.md.

## RQ2 additional diagnostic: completed relevance sensitivity

Both reviewers independently return 344 supplementary code judgments across 126 of the original 128 queries. The pooled Top-3 set has 610 query/code-text panels: 266 inherit original labels and 344 receive new labels. Supplementary agreement is 55.23% (nominal kappa 0.321); jointly judged coverage is 296/344 (86.05%) with linear weighted kappa 0.484. All 154 supplementary disagreements remain available. Each reviewer retains 49 unknown judgments in the merged pool; these are not recoded as irrelevant.

The primary diagnostic specified before supplementary judgments is RIDGE_verified minus the original query at RR@3, using grade-2 additions and original eight-stratum weights.

|Reviewer|Expanded-relevance bounds before supplement|Bounds after supplement|Identified query contrasts|
|---|---|---|---:|
|A|[-0.017841, +0.041899]|[+0.005843, +0.018263]|102 → 121/128|
|B|[-0.032886, +0.059318]|[+0.001279, +0.016604]|96 → 122/128|

These are sample identification bounds over shared unknown relevance, not confidence intervals; sampling and annotation uncertainty are omitted. The original-qrels value at this same RR@3 endpoint is +0.001594. The change from that contrast is A [+0.004250, +0.016669] and B [-0.000315, +0.015011]; B's change includes zero.

|RIDGE_verified RR@3 sensitivity|A|B|
|---|---|---|
|Grades 1/2, original stratum weights|[+0.003297, +0.005007]|[+0.001948, +0.010069]|
|Grade 2, unweighted|[+0.003906, +0.023438]|[-0.006510, +0.011719]|

ENTR Top1 retains negative weighted bounds for both reviewers at both thresholds, whereas safe label-assisted best3 opportunity remains positive. The risk stage does not have positive bounds across both reviewers and thresholds. This diagnostic quantifies relevance-coverage sensitivity for fixed policies; it does not replace full-H MRR@100, B/G results or establish exclusive causal shares.

Complete 48-combination results, 6,144 exact query records, labels, source returns and acceptance are in results/rene_review_completion_20261021/. Tables are separately mapped in DIAGNOSTIC_TABLES.json. Full definitions, the unchanged RR@1 analysis and functional-boundary cases are in relevance_sensitivity_results.md and review_disagreement_discussion.md. The original 15-table bundle retains its source bindings.

### Sampling uncertainty for the relevance diagnostic

The positive identification ranges above concern unknown relevance in the reviewed sample. A separate post-hoc analysis uses the original eight strata, 16 sampled queries per stratum and original frame weights to estimate sampling envelopes for the fixed H8530 endpoint means. It treats each reviewer's final relevance rule as fixed and interprets salted-hash sampling as within-stratum sampling without replacement.

|Grade-2 RIDGE_verified RR@3|Sample identification range|Conservative finite-frame 95% envelope|Stratified-t approximate 95% envelope|
|---|---|---|---|
|A|[+0.005843, +0.018263]|[-0.400821, +0.417409]|[-0.008645, +0.033373]|
|B|[+0.001279, +0.016604]|[-0.401856, +0.420320]|[-0.009318, +0.033184]|

Both methods span zero for both reviewers and both relevance thresholds; the additional conservative envelope simultaneous across the four reviewer/threshold combinations also spans zero. The exact construction is conservative over the full RR support; the t construction is approximate. Neither models annotation error or adjusts for choosing the diagnostic rule after seeing original outcomes and labels. These results retain the sample's coverage-sensitivity finding without establishing positive finite-frame utility. Full methods and all 48 combinations are in relevance_sampling_uncertainty.md and results/rene_relevance_sampling_uncertainty_20261021/RESULTS.json. The B/G, confirmation and original full-H estimates are unchanged.

## Reconstruction sensitivity

Eight training configurations share 32,768 rows and 1,024 updates; T15 retains all 28 adjusted primary changes. Length settings produce identical weights and candidates in this bank. Most training contrasts remain unresolved. Author weights, original CSV bytes and production aggregation remain unavailable: these experiments do not identify a percentage of the historical replication discrepancy. Full configurations and results are in implementation_fidelity.md and cqc_training_fidelity_results.md.

## Supplementary: earlier decision-specific uncertainty comparisons

|Population / target|Comparison|ΔMRR|97.5% simultaneous interval|
|---|---|---:|---|
|1536 downstream repositories; full pipeline|uq_increment|+0.003952|[+0.001330, +0.006985]|
|1024 downstream repositories; fixed upstream action|full_minus_common|-0.000080|[-0.003612, +0.003453]|
|1024 downstream repositories; fixed upstream action|full_minus_support|+0.001347|[-0.001215, +0.004050]|

The 1,536-query comparison supports a positive increment in that frozen full pipeline. The 1,024-query fixed-action full−common comparison meets its prespecified ±0.005 equivalence rule; full−support remains unresolved because its event guard is not met. These datasets, upstream actions and estimands differ; the later comparison is not a causal decomposition or retraction of the earlier positive result.

These comparisons are retained in supplementary_uncertainty.md and T7. They remain outside the three main research questions; their different populations and actions do not provide a direct decision-stage interaction test.

## Knowledge increment and reporting implications

The external reconstruction establishes the outcome of a specified SSQR comparison. The same-pool analysis measures how selected count and aggregation change advantage against exact random, with directional confirmation on StaQC and measured retriever conditions. Executable controls measure positive-gain retention, ranking loss and work against the original query for specified policies. Their diagnostic contribution does not depend on establishing positive deployment gain. Human review distinguishes semantic and relevance interpretations. B/G and executable utility retain different references and do not constitute a causal decomposition of policy loss.

Report selector advantage relative to exact random separately from net utility relative to the original query. Vary selected count with aggregation held fixed, and aggregation with count held fixed. Evaluate the executable policy's positive gain, ranking loss, harmful-adoption frequency and paid cost. Best-k remains a label-assisted opportunity statistic. Detailed contribution mapping is in contribution_evidence.md.

Current machine source: results/rene_manuscript_integration_20261021/TABLES.json (15 tables, 176 rows). It preserves seven historical tables and adds corrected contrasts and review diagnostics, with no new model-effect estimation. Each experiment retains its correction family and population; results are not pooled.

Completed supplementary-review diagnostics and their accepted numerical version are supplied by results/rene_review_completion_20261021/{DIAGNOSTIC_TABLES.json,RESULTS.json,FINAL_ACCEPTANCE.json}. Current editorial integration, the sampling-uncertainty extension and independent pre-review materials are tracked in results/rene_contribution_strength_20261021/.
