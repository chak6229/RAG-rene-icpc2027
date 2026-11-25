# CQC Impact of limited training configuration: results and judgement

Updated training in data/such as 8 group, two-dimensional perspective assessment and three-tiered independent audit have been completed. The conclusions are:**The length cap had a strict zero effect during the current round; Optimizers/mobilization/training seeds will significantly change candidates and choices, but the open superparameter combination does not create corrections for local references that are improving, nor does it eliminate the current round's “best3 opportunity, Top1 net yield points estimated to be negative”.**

## Common control

All start with the same original t5-base entry, fixed 32,768 CODEnn training line, continuous 15% mask and sequence, 1,024 update, batch8 × 4, lr starting point.001, BF16 calculate/FP32 weights, last steps checkpoint, no earlier. Validate NLL not for weights. Training totals approximately 97.7 minutes, excluding generation/retrievation/audit. The configuration is detailed in METHODS.md and frozen DESIGN.md.

Source and CoSQA each use 512 queries in 512 original groups, the complete 19210/6267-document corpora and the same frozen CodeBERT. These are the historical bridge queries. Each arm has 9340 position candidates. Across both views and eight arms, there are 149440 candidate RRs and 16384 query records.

L=AdamW/constant/512/wd.01/seed101； O Only Adafactor; S Open only literally linear100000; N Length only 128; W Only wd0; P=Adafactor/linear100000/128/wd0/seed101； PN Only scale the schedule of P horizon by historical progress as 1093; PS Only P training seed will be modified 42. Adafactor relative_step and scale_parameter both False.

## 1...what changes have been made to the search indicators by training factors?

S3 = ENTR best3− Accurate random best3; Top1 Net income = ENTR Top1−NoRewrite. The table below sets out the changes in Configure-L directly with the query and is not less than the historical full volume model.

|Configure - L|source: S3 Change|CoSQA: S3 Change|source: Top1 Changes in net income|CoSQA: Top1 Changes in net income|
|---|---:|---:|---:|---:|
|O Exchange only Adafactor|-0.00361666|+0.00386055|-0.01083141|-0.00507920|
|S Exchange only linear100000|-0.00279246|+0.00134978|-0.01095998|+0.00245741|
|N Length only 128|+0.00000000|+0.00000000|+0.00000000|+0.00000000|
|W Change only weight decay to 0|+0.00149885|+0.00308944|-0.00890648|-0.00733077|
|P Open ultra-parameter combination|-0.00114332|-0.00396651|-0.01179079|+0.00148675|
|PN Group + progress scaling|+0.00507610|+0.00279995|-0.00693821|+0.00555303|
|PS Open Portfolio + Training seed42|+0.00240475|+0.00655420|-0.01611474|-0.01427124|

**Of the 28 prespecified comparisons, 0 are positive after correction, 1 is negative after correction, 4 are structural zeros under the length setting, and 23 have intervals spanning zero.** The single corrected negative result is for source/PS−L Top1 net utility: −.01611474, with bootstrap interval [−.03271290,−.00059109] and t interval [−.03221445,−.00001503]. PS−L changes both the setting bundle and seed, so the contrast does not isolate the seed effect.

The remaining intervals that span zero establish neither a null effect nor equivalence. For example, the public setting bundle changes source Top1 by −.01179079, and the corrected interval leaves the magnitude and direction uncertain. Both intervals for all 28 comparisons are in `READOUT.md` and `PRIMARY.csv`.

### Open the direct answer to a combination of superparameters

|Domain|Indicator change P−L|Matching estimate|28 Item correction bootstrap|28 Item correction cluster-t|
|---|---|---:|---|---|
|source|S3|-0.00114332|[-0.01347887, +0.01066236]|[-0.01365588, +0.01136924]|
|source|top1_gain|-0.01179079|[-0.02949673, +0.00379373]|[-0.02865390, +0.00507232]|
|cosqa|S3|-0.00396651|[-0.02318952, +0.01510376]|[-0.02305141, +0.01511840]|
|cosqa|top1_gain|+0.00148675|[-0.01792189, +0.02302523]|[-0.01935724, +0.02233074]|

Under the same limited budget, P relative to L does not establish a corrected ENTR advantage over exact random selection or an improvement in Top1 net gain. Equivalence of the optimization and scheduling settings is not established.

## 2. Maximum length: hard to locate zero effect

Maximum input for shared training bank 78, maximum label 26; Validate 55 / 14, no 128 cut. L/N 1024 steps loss and gradients, real weight documents of Hashi; Two Domains 9340 Newly generated candidates are identical bytes, fixed pool scores, selections, RR and all effects are the same. The training and evaluation effect of the current length difference is therefore accurate as 0.

(a) The full millions of auxiliary scans under the source order/seed101, each fixed mask with 22 entry > 128 (0.0022%), label 0 Article > 128; max306/54. This is not a full dynamic mask for the three rounds of history, and it cannot be extrapolated from the zero effect of this round to the zero effect of all historical training.

## 3. Candidate generation and pool rating will be affected by configuration

Open combination of relative L changes source 4413/5995 = 73.61%, CoSQA 1898/ 3345 = 56.74%. ENTR Top1 was changed to 195/512 and 186/512, respectively, while Top3 pool was changed 308/512 and 253/512. The result comes from an unlabelled action audit ACTION_DIAGNOSTIC.json.

Only changes in pairs after re-scoring historical candidates:

|Domain|Change in S3 for the fixed candidate P-L|Change in net income of Top1 fixed candidate P−L|
|---|---:|---:|
|source|-0.00222529|-0.00505630|
|cosqa|+0.00253087|-0.00671564|

The four fixed-pool P−L intervals span zero. The fixed-pool and newly generated views answer different questions. Their difference does not identify a unique generation effect. See `DESCRIPTIVE.csv` for the complete fixed-pool, PROB and exact random intervals.

## 4. Schedule progress and training seeds

Open source hard-coded horizon100000; Cannot be called after a direct decline to zero in 1024 step budget. PN Visible use 1093, P still used 100000. PN Last Step Validate NLL 1.14316240, P 1.21728411; Improvements in training objectives need to be viewed separately for retrieval purposes.

|Perspective|Domain|PN−P Indicators|Matching estimate|Descriptive 95% Group bootstrap|
|---|---|---|---:|---|
|own|source|S3|+0.00621942|[-0.00303360, +0.01634504]|
|own|source|top1_gain|+0.00485258|[-0.00702385, +0.01687512]|
|fixed|source|S3|+0.00862999|[+0.00086759, +0.01711667]|
|fixed|source|top1_gain|+0.01296765|[+0.00297239, +0.02398273]|
|own|cosqa|S3|+0.00676645|[-0.00429905, +0.01770037]|
|own|cosqa|top1_gain|+0.00406629|[-0.00987643, +0.01748745]|
|fixed|cosqa|S3|-0.00219973|[-0.01025333, +0.00573577]|
|fixed|cosqa|top1_gain|-0.00686437|[-0.02113743, +0.00700184]|

For PN−P on newly generated pools, both primary contrasts span zero in both domains. For the historical source pool, S3 and Top1 differences are positive with pointwise 95% intervals above zero. Improvements in CoSQA are not established. These 8 prespecified descriptive comparisons are outside the primary family of 28. They did not select PN as the study model.

Change training random seeds (PS−P) separately, data, mask, sequence fixed:

|Domain|PS−P Indicators|Matching estimate|Descriptive 95% Group bootstrap|
|---|---|---:|---|
|source|S3|+0.00354807|[-0.00416788, +0.01158168]|
|source|top1_gain|-0.00432394|[-0.01459495, +0.00609914]|
|cosqa|S3|+0.01052071|[-0.00040889, +0.02160555]|
|cosqa|top1_gain|-0.01575799|[-0.03027118, -0.00147269]|

The CoSQA Top1 description is negative, suggesting sensitivity of the current single training to adoption utility. The check is only for 101/42 pairs, does not estimate the overall variance of the training seed and does not select the best weight from one seed.

The protocol fixed the nonadditive residual as the joint effect minus the sum of the four single-setting effects before opening outcomes. All 4 intervals for this residual in the own-pool view span zero. An individual interaction remains unestablished. `EXTRA_CONTRASTS.json` has the 16 comparisons. The independent delivery check recomputed them with signed coefficients.

## 5. Effect on the main line of the paper

|Domain|Branch|ENTR Top1 Zenhara query|ENTR best3 Zenhara query|ENTR best3− Exact Random|
|---|---|---:|---:|---:|
|source|L|-0.00262201|+0.02997696|-0.00000791|
|source|O|-0.01345342|+0.02621607|-0.00362457|
|source|S|-0.01358199|+0.03151493|-0.00280037|
|source|N|-0.00262201|+0.02997696|-0.00000791|
|source|W|-0.01152849|+0.02931658|+0.00149094|
|source|P|-0.01441280|+0.03159208|-0.00115123|
|source|PN|-0.00956022|+0.03982129|+0.00506819|
|source|PS|-0.01873675|+0.02977441|+0.00239684|
|cosqa|L|-0.01369856|+0.05321337|+0.00000726|
|cosqa|O|-0.01877776|+0.05963442|+0.00386781|
|cosqa|S|-0.01124115|+0.05362517|+0.00135704|
|cosqa|N|-0.01369856|+0.05321337|+0.00000726|
|cosqa|W|-0.02102933|+0.05409982|+0.00309670|
|cosqa|P|-0.01221181|+0.04908240|-0.00395925|
|cosqa|PN|-0.00814553|+0.05630480|+0.00280721|
|cosqa|PS|-0.02796981|+0.05792475|+0.00656146|

Across the eight settings and two domains, Top1 net-gain point estimates are negative and best3 opportunity estimates are positive. These 16 cells are sensitivities within one design. Some Top1 95% intervals span zero, so a negative direction is not established for all 16 cells. Best3 uses relevance labels and measures candidate opportunity.

This round is RENE to add "training settings controlled sensitivity" evidence and close the limited configuration against this test defect. The most explicit exclusion is the interpretation of the length of this round; The impact of other configurations is quantified by pair. There is currently no evidence to suggest that an automatic single selection can be converted into a positive gain by switching to an excess parameter in the public table, nor can the recurrence discrepancy be attributed to CoSQA sources of queries.

**All differences between historical 93750 step training and authors not available checkpoint have still not been identified in this round.**P Shared retention of local training data, sheet span mask, batch/accuracy and end-step selection, which cannot be referred to as full public display; This round 1024 steps do not cover long-range training dynamics. The paper is presented in the form of a “set-down impact under a fixed limited budget” and does not explain percentages of historical differences.

## 6. Independent review and delivery

- 8 Scientific tests passed; Training audits re-engineer 32768 training cases, 8192 updates and step by step lr, 8 weights on Hashi, L/N negative control.
- Independent Fraction/CDF checks recomputed 149440 candidate RRs, 311296 query metric fields, 2432 summary statistics and 648 paired fields. The maximum discrepancy was 4.997e−16. Ranking checks covered 45305 texts.
- A new article 34764 search text with 544003702 lasting scores corresponding to Top100 line-by-line review; It's not that independent reset all points.
- The T5 audit checked 192 cases, including independent greedy regeneration for 96. Maximum FP64 discrepancies were 9.618e−7 for entropy and 8.843e−7 for log probability. Recomputing 64 CodeBERT vectors in 4 batches gave maximum discrepancy 0.
- Third-tier audit TRAIN_AUDIT/ EFFECT_AUDIT MODEL_AUDIT Adopted; Additional contrast, CSV fields, all protecting Hashi and final delivery see FINAL_ACCEPTANCE.json.

Outcome documents: RESULTS.json, PRIMARY.csv, DESCRIPTIVE.csv, per_query.jsonl, candidate_RR.jsonl. Figure: cqc_configuration_effects.png/pdf. New weight and dense score: the local model/cache directory (not included in this review export).

Supplementary paper: paper/rene2027/cqc_training_fidelity_results.md. Sources, history T5/CodeBERT, official five pointers and originals H confirmed to be maintained; new_confirmation=false, risk_certified=false.
