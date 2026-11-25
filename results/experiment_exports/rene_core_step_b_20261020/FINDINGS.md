# Step B: fixed-action adoption results

**The incremental adoption utility of generation confidence T was not established.** Adding original-query evidence R gives a positive point estimate for the CodeBERT zero-threshold policy, with an interval spanning zero. The corresponding BM25 contrast is negative. The comparison does not validate a new adoption algorithm.

See `READOUT.md` and `RESULTS.json` for the experiment and full tables. The study reuses 1490 development queries in 512 linked groups and the same 5 group-separated OOF folds. It fixes the existing ENTR Top1 action for each query and compares four equally supervised Ridge heads (alpha=1). This is a development diagnostic.

## Same as the quantity result

M0 = common text/candidate search feature, MT = M0+ generation of confidence, MR = M0+ original query reference, MRT = both. 20%, which fixes the action available for each discount, to take the whole number down: CodeBERT Article 296 per head, BM25 Article 274 per head. The following Δ MRR for all 1490 Query is the denominator with NoRewrite as the baseline.

|Model|CodeBERT ΔMRR|CodeBERT Harmful/used|BM25 ΔMRR|BM25 Harmful/used|
|---|---:|---:|---:|---:|
|M0|+0.00134060|53/296|-0.00104922|94/274|
|MT|-0.00066742|55/296|-0.00117883|93/274|
|MR|+0.00107971|75/296|-0.00190590|92/274|
|MRT|+0.00136336|81/296|-0.00225271|96/274|

CodeBERT Key constituents:

- Add R: MRT−MT = +0.00203078, the descriptive group bootstrap95% is [-0.00089554, +0.00500454.
- Add T: MRT−MR = +0.00028365, 95% to [-0.00150208, +0.00219259.
- MRT and M0 point only +0.00002276; can't judge the whole module to be useful because it's a little large.
- M0 → MRT: Positive yield/Query 0.00526630 0.00632645, Loss/Query 0.00392570 0.00496309, net difference almost offset. Harmful number 53→ 81, average harmful loss approximately 0.11036→ 0.09130. The objective of the proceeds can accept more minor damage and does not automatically constitute a strategy for protection against harm.

## Zero threshold result

The zero-threshold rule adopts when predicted ΔRR>0. CodeBERT M0/MT/MR/MRT yield -0.00031563/-0.00147678/+0.00280718/+0.00265119. All four descriptive 95% intervals relative to NoRewrite span zero. MR made 673 adoptions with 158 harmful outcomes, and MRT made 665 adoptions with 157 harmful outcomes.

CodeBERT MRT-MT=+0.00412798, 95% [-0.00011372, +0.00842415], eight family Bonferroni district[-0.00200575, +0.01025486]. MRT−MR=-0.00015599, 95%[-0.00234377,+0.00206902].

All four BM25 point estimates are negative. MRT−MT=-0.00199406, with 95% interval [-0.00361088,-0.00070823] and eight-comparison corrected interval [-0.00443257,-0.00034870]. This is a negative diagnostic for adding R in the tested configuration. Some BM25 fixed-budget pointwise intervals are negative, while all the family-corrected intervals span zero.

## Forecast quality and main line meaning

CodeBERT MR Minimum MSE 0.03204865; Add T after MRT is 0.03257074 MAE Also from 0.09491220 Increase to 0.09682515. BM25 Corresponding to MSE 0.00982960 → 0.00991348. The generation of confidence does not show additional forecast quality advantages during the current round; This is the result of specifying a single time teacher-forcing signal with Ridge head, not covering all UQ algorithms.

Step A supports the use of entropy to order candidates in the tested CoSQA/CodeBERT pool. Step B did not establish conditional gains from the same generation-confidence signal for adoption. The studies estimate different quantities. Their individual decisions do not establish a statistical interaction between stages. The results describe the tested tasks and retrievers.

Step B did not apply a risk gate, so the observed adoption results cannot be attributed to fallback. The common-feature baseline remains a separate comparator, and extra gain from the full head is unestablished. Planned work at the time called for comparison with related studies. Additional threshold searches or a different learner were outside this result. A query-source explanation would require the separate C comparison, and confirming a new policy would require eligible new data and a frozen primary comparison.

The input block contains fractional differences and intra-pool statistics, Ridge also changes the expression and validity rule when adding these columns, so that the result is to assign a recalcitrant comparison of the input/leaving combination rather than " original search evidence" as the pure causal effect of abstract information.

## Audit and recovery

- 6 Contract tests passed; 40 Recompositation of separate linear algebras for models, with maximum projected deviations 2.73e-14; 40 Distortion label input difference 0.
- Independent reconstruction 2980 individual 71 (a) the dimensional characteristic vector, 2980 Group Fraction original/candidacy/disparity RR, 23840 On a case-by-case basis 34 Effects/ 220 Fields, all through.
- The freezing of input/source code/protection files and 43 model/decision storage documents for Division 44 are consistent. Formal guidance, old H confirmation and maintenance of the first step results; risk_certified=false.
- The first audit found that the parallel calibre of the auxiliary Spearman was different: production reduced by floating point RR, accuracy Fraction restored mathematically parallels. Retain original freeze `audit.py `And all the results, save `audit_v2.py `and `AUDIT_REPAIR.json `(b) Independent re-aligning of report calibres while preserving the sensitivity of precise scores; Max. Spearman 0.00009954, do not change strategy, damage or damage MRR...
- Final acceptance and inspection `FINAL_ACCEPTANCE.json `For PASS. Independent audit portal `experiments.rene_core_step_b_20261020.audit_v2 `; the original entry is kept as a freeze history. Full output cannot overwrite runback.
- (a) Re-sampling of the descriptive groups on all sectors with fixed output OOF, without each retrainment and budget reordering; Could not close temporary folder: %s
