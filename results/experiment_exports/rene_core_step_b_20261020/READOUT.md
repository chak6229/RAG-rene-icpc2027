# Step B: generation confidence and the original-query reference for a fixed rewrite

The experiment fixes the ENTR Top1 rewrite and examines the additional decision utility of original-query retrieval evidence R and generation confidence T when deciding whether to replace the original query.

The analysis uses 1490 previously used CoSQA development queries in 512 linked groups, both retrievers and the same 5 group-separated out-of-fold splits. The candidate, labels, learner and folds were fixed. It fits 40 StandardScaler + Ridge (alpha=1, SVD) models without tuning. Scaling is fitted only on the training folds, and a decision switches between the complete original and rewritten Top100 lists.

M0 has 27 shared text, candidate retrieval and lexical-coverage features. MT adds 10 generation-confidence features, MR adds 34 original-query reference features, and MRT uses 71 features. Token length is shared. T comprises single-pass teacher-forcing probabilities and entropies with ENTR3 pool statistics. It does not separate epistemic and aleatoric uncertainty. R uses no qrels, target IDs or known target ranks.

All heads share the observable requirements of a changed text and a changed Top100 ranking. M0/MT apply the same eligibility rule while their predictors do not receive original-query retrieval features. Fixed20 adopts the top 20% of eligible actions in each evaluation fold using floor(eligible actions/5), breaking ties by query ID and allowing negative predicted gain. It is a fixed-batch diagnostic. The zero-threshold rule adopts only when predicted ΔRR>0. This comparison did not include risk certification.

## CodeBERT

Eligible actions: 1490/1490. Identical texts: 0. Identical full ranks: 0. NoRewrite MRR=0.51890040. Forced ENTR Top1 ΔMRR=-0.01365851.

|Rule|Model|Number used|Coverage|ΔMRR|Descriptive group bootstrap95%|Harmful/used|Losses per query|Retention of positive proceeds|
|---|---|---:|---:|---:|---|---:|---:|---:|
|positive|M0|515|34.56%|-0.00031563|[-0.00468573, +0.00398034]|101/515|0.00841560|26.89%|
|positive|MT|526|35.30%|-0.00147678|[-0.00584448, +0.00286956]|111/526|0.00990507|27.98%|
|positive|MR|673|45.17%|+0.00280718|[-0.00204468, +0.00785279]|158/673|0.01066756|44.74%|
|positive|MRT|665|44.63%|+0.00265119|[-0.00237067, +0.00779101]|157/665|0.01107565|45.58%|
|fixed20|M0|296|19.87%|+0.00134060|[-0.00188537, +0.00467032]|53/296|0.00392570|17.49%|
|fixed20|MT|296|19.87%|-0.00066742|[-0.00366518, +0.00250298]|55/296|0.00465025|13.22%|
|fixed20|MR|296|19.87%|+0.00107971|[-0.00249419, +0.00470514]|75/296|0.00522456|20.93%|
|fixed20|MRT|296|19.87%|+0.00136336|[-0.00207056, +0.00499405]|81/296|0.00496309|21.01%|

The primary conditional increments are MRT−MT for the original-query reference and MRT−MR for generation confidence. The family has eight contrasts across two retrievers and two adoption rules.

|Rule|Comparison|Δ MRR|Points bootstrap95%|Eight items Bonferroni Family 95%|
|---|---|---:|---|---|
|positive|MRT-MT|+0.00412798|[-0.00011372, +0.00842415]|[-0.00200575, +0.01025486]|
|positive|MRT-MR|-0.00015599|[-0.00234377, +0.00206902]|[-0.00315262, +0.00291544]|
|fixed20|MRT-MT|+0.00203078|[-0.00089554, +0.00500454]|[-0.00193256, +0.00631901]|
|fixed20|MRT-MR|+0.00028365|[-0.00150208, +0.00219259]|[-0.00214964, +0.00299282]|

|Quality of projections|M0|MT|MR|MRT|
|---|---:|---:|---:|---:|
|mse|0.03298345|0.03344376|0.03204865|0.03257074|
|mae|0.09002880|0.09185681|0.09491220|0.09682515|
|prediction_spearman|0.06179201|0.03275488|0.07752509|0.05941994|
|bias|0.00064597|0.00090310|0.00092527|0.00123468|
|mean_absolute_quintile_calibration_gap|0.01372439|0.01958198|0.02065428|0.02421493|

`RESULTS.json` contains fold-specific gains, observed harm rates and loss magnitudes, quintile calibration and secondary MR−M0 and MT−M0 contrasts. `per_query.jsonl` records fixed actions, predictions, decisions and observed RRs.

## BM25

Eligible actions: 1382/1490. Identical texts: 0. Identical full ranks: 108. NoRewrite MRR=0.22412467. Forced ENTR Top1 ΔMRR=-0.02035096.

|Rule|Model|Number used|Coverage|ΔMRR|Descriptive group bootstrap95%|Harmful/used|Losses per query|Retention of positive proceeds|
|---|---|---:|---:|---:|---|---:|---:|---:|
|positive|M0|241|16.17%|-0.00082091|[-0.00139271, -0.00029974]|77/241|0.00101927|4.47%|
|positive|MT|253|16.98%|-0.00103303|[-0.00167126, -0.00044556]|84/253|0.00124170|4.70%|
|positive|MR|343|23.02%|-0.00295111|[-0.00459446, -0.00164039]|123/343|0.00308412|3.00%|
|positive|MRT|344|23.09%|-0.00302710|[-0.00467936, -0.00168200]|122/344|0.00323638|4.71%|
|fixed20|M0|274|18.39%|-0.00104922|[-0.00163569, -0.00050988]|94/274|0.00124758|4.47%|
|fixed20|MT|274|18.39%|-0.00117883|[-0.00182518, -0.00056966]|93/274|0.00138750|4.70%|
|fixed20|MR|274|18.39%|-0.00190590|[-0.00314008, -0.00094161]|92/274|0.00200333|2.19%|
|fixed20|MRT|274|18.39%|-0.00225271|[-0.00354231, -0.00122280]|96/274|0.00235014|2.19%|

The primary conditional increments are MRT−MT for the original-query reference and MRT−MR for generation confidence. The family has eight contrasts across two retrievers and two adoption rules.

|Rule|Comparison|Δ MRR|Points bootstrap95%|Eight items Bonferroni Family 95%|
|---|---|---:|---|---|
|positive|MRT-MT|-0.00199406|[-0.00361088, -0.00070823]|[-0.00443257, -0.00034870]|
|positive|MRT-MR|-0.00007599|[-0.00042029, +0.00028668]|[-0.00055360, +0.00047407]|
|fixed20|MRT-MT|-0.00107388|[-0.00230078, -0.00010401]|[-0.00291996, +0.00018168]|
|fixed20|MRT-MR|-0.00034681|[-0.00073597, -0.00002147]|[-0.00090410, +0.00008002]|

|Quality of projections|M0|MT|MR|MRT|
|---|---:|---:|---:|---:|
|mse|0.00984201|0.00992259|0.00982960|0.00991348|
|mae|0.04256433|0.04317866|0.04373810|0.04454257|
|prediction_spearman|0.18697805|0.17356136|0.15906749|0.15332007|
|bias|-0.00024255|-0.00011845|-0.00013622|-0.00005253|
|mean_absolute_quintile_calibration_gap|0.00779409|0.00886196|0.01243896|0.01442166|

`RESULTS.json` contains fold-specific gains, observed harm rates and loss magnitudes, quintile calibration and secondary MR−M0 and MT−M0 contrasts. `per_query.jsonl` records fixed actions, predictions, decisions and observed RRs.

## Interpretation and acceptance

A positive MRT−MT increment means that adding the observable original-query reference improved the measured adoption results for these fixed-action Ridge heads. It does not establish a distinct algorithm or an effect across all retrievers. An MRT−MR interval spanning zero leaves the incremental utility of T unresolved and does not establish that UQ is ineffective. Prediction error and adoption gains are assessed separately. Improvement over forced harmful Top1 may still be below NoRewrite. The development comparison has no equivalence margin.

The intervals resample paired linked groups with query weighting using the existing out-of-fold predictions. Each resample retains the fixed models and quotas. This yields descriptive precision conditional on the OOF outputs. The development set had already been used for design, and the present comparison was fixed before the new fits.

See `AUDIT.json` and `FINAL_ACCEPTANCE.json` for the independent Fraction RR, linear-algebra refits, held-out-label perturbation, independent feature calculation, quota and interval replay, and hash checks. The original H confirmation, study configuration and NoRewrite policy were retained. `risk_certified=false`.
