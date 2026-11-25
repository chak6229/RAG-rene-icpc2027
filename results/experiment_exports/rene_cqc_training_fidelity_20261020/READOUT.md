# CQC Limited training configuration comparison: complete results

Fixed original t5-base initialization, 32768 Article shared CODEnn sample/fixed mask and sequence, 1024 Update, Valid batch32, lr Starting point.001, fixed the end step weight. All branches have the same training data and updated budgets.
The source512/512 repository and the CoSQA development 512 / 512 associated group follow the same freeze CodeBERT as the used bridge sample, complete 19210 / 6267. This round is not new; Unread H.

L Local; O Only Adafactor; S Only in writing linear100000; N Length only 128; W Only weight_decay0; P Grouping in the public table; PN Scaled according to historical progress linear1093; PS for Psee 42 Sensitivity.
All branches maintain the same local continuity 15% mask; P does not restore the author CSV or complete trainer. PN Explain it to the public on paper.

## Negative training versus length

|Branch|Fixed Initial Verification NLL|1024 Step validation NLL|Length of training (seconds, this process)|
|---|---:|---:|---:|
|L|4.59859956|1.21743914|742.0|
|O|4.59859956|1.22686025|749.2|
|S|4.59859956|1.20602935|720.9|
|N|4.59859956|1.21743914|713.8|
|W|4.59859956|1.21905978|718.5|
|P|4.59859956|1.21728411|737.3|
|PN|4.59859956|1.14316240|738.1|
|PS|4.59859956|1.21234993|744.1|

128 / 512 training input did not trigger the cut-off; L is fully consistent with the 1024 step of N loss / gradient trajectory and end-step weight bytes, which are independently negative.

## Under the same budget, each configuration generates its own candidate

S3 = ENTR best3− Accurate random best3; S1 = ENTR Top1− Accurate random Top1. best3 Optimal use of relevant labels is an opportunity indicator.

|Domain|Branch|NoRewrite MRR|ENTR Top1 Zenhara query|ENTR best3 Zenhara query|S1|S3|PROB best3− Exact Random|
|---|---|---:|---:|---:|---:|---:|---:|
|source|L|0.29908527|-0.00262201|+0.02997696|+0.01615357|-0.00000791|+0.00070047|
|source|O|0.29908527|-0.01345342|+0.02621607|+0.00313181|-0.00362457|-0.00677464|
|source|S|0.29908527|-0.01358199|+0.03151493|+0.00484210|-0.00280037|-0.00278996|
|source|N|0.29908527|-0.00262201|+0.02997696|+0.01615357|-0.00000791|+0.00070047|
|source|W|0.29908527|-0.01152849|+0.02931658|+0.00767213|+0.00149094|+0.00453509|
|source|P|0.29908527|-0.01441280|+0.03159208|+0.00212594|-0.00115123|+0.00080431|
|source|PN|0.29908527|-0.00956022|+0.03982129|+0.00421427|+0.00506819|-0.00008980|
|source|PS|0.29908527|-0.01873675|+0.02977441|+0.00081027|+0.00239684|-0.00094141|
|cosqa|L|0.51875256|-0.01369856|+0.05321337|+0.01456601|+0.00000726|+0.00104362|
|cosqa|O|0.51875256|-0.01877776|+0.05963442|+0.01270384|+0.00386781|-0.00234652|
|cosqa|S|0.51875256|-0.01124115|+0.05362517|+0.01923015|+0.00135704|+0.00258824|
|cosqa|N|0.51875256|-0.01369856|+0.05321337|+0.01456601|+0.00000726|+0.00104362|
|cosqa|W|0.51875256|-0.02102933|+0.05409982|+0.00830180|+0.00309670|-0.00370203|
|cosqa|P|0.51875256|-0.01221181|+0.04908240|+0.01740839|-0.00395925|-0.00262808|
|cosqa|PN|0.51875256|-0.00814553|+0.05630480|+0.02100812|+0.00280721|-0.00125990|
|cosqa|PS|0.51875256|-0.02796981|+0.05792475|+0.00606060|+0.00656146|-0.00180321|

## Direct pair changes relative to budget matching L

28 Pre-sets = net proceeds of 7 branches of L branch x 2 domain x S3/Top1. The following table has been corrected by Bonferroni while retaining groups bootstrap and cluster-t. The positive value increases the corresponding indicator for this configuration L; Not a pairing effect with the historical 93750 step model.

|Domain|Branch - L|Change in indicators|estimate|28 Item correction bootstrap|28 Item correction t|
|---|---|---|---:|---|---|
|source|O−L|S3|-0.00361666|[-0.01775081, +0.00920340]|[-0.01753811, +0.01030480]|
|source|O−L|top1_gain|-0.01083141|[-0.03099507, +0.00696240]|[-0.02977581, +0.00811299]|
|source|S−L|S3|-0.00279246|[-0.01879184, +0.01271154]|[-0.01831817, +0.01273326]|
|source|S−L|top1_gain|-0.01095998|[-0.02821080, +0.00472164]|[-0.02773445, +0.00581449]|
|source|N−L|S3|+0.00000000|[+0.00000000, +0.00000000]|[+0.00000000, +0.00000000]|
|source|N−L|top1_gain|+0.00000000|[+0.00000000, +0.00000000]|[+0.00000000, +0.00000000]|
|source|W−L|S3|+0.00149885|[-0.01173091, +0.01451837]|[-0.01151804, +0.01451573]|
|source|W−L|top1_gain|-0.00890648|[-0.02675286, +0.00892952]|[-0.02723410, +0.00942114]|
|source|P−L|S3|-0.00114332|[-0.01347887, +0.01066236]|[-0.01365588, +0.01136924]|
|source|P−L|top1_gain|-0.01179079|[-0.02949673, +0.00379373]|[-0.02865390, +0.00507232]|
|source|PN−L|S3|+0.00507610|[-0.00944618, +0.02073715]|[-0.00998714, +0.02013934]|
|source|PN−L|top1_gain|-0.00693821|[-0.02340819, +0.00919458]|[-0.02303554, +0.00915911]|
|source|PS−L|S3|+0.00240475|[-0.01035257, +0.01576811]|[-0.01040835, +0.01521784]|
|source|PS−L|top1_gain|-0.01611474|[-0.03271290, -0.00059109]|[-0.03221445, -0.00001503]|
|cosqa|O−L|S3|+0.00386055|[-0.01701880, +0.02440301]|[-0.01656198, +0.02428308]|
|cosqa|O−L|top1_gain|-0.00507920|[-0.02663010, +0.01518443]|[-0.02632109, +0.01616268]|
|cosqa|S−L|S3|+0.00134978|[-0.01738840, +0.01994906]|[-0.01711086, +0.01981042]|
|cosqa|S−L|top1_gain|+0.00245741|[-0.01975316, +0.02413766]|[-0.02047427, +0.02538909]|
|cosqa|N−L|S3|+0.00000000|[+0.00000000, +0.00000000]|[+0.00000000, +0.00000000]|
|cosqa|N−L|top1_gain|+0.00000000|[+0.00000000, +0.00000000]|[+0.00000000, +0.00000000]|
|cosqa|W−L|S3|+0.00308944|[-0.01489043, +0.02226375]|[-0.01558002, +0.02175889]|
|cosqa|W−L|top1_gain|-0.00733077|[-0.02768186, +0.01242557]|[-0.02741488, +0.01275335]|
|cosqa|P−L|S3|-0.00396651|[-0.02318952, +0.01510376]|[-0.02305141, +0.01511840]|
|cosqa|P−L|top1_gain|+0.00148675|[-0.01792189, +0.02302523]|[-0.01935724, +0.02233074]|
|cosqa|PN−L|S3|+0.00279995|[-0.01364006, +0.02132589]|[-0.01496912, +0.02056901]|
|cosqa|PN−L|top1_gain|+0.00555303|[-0.01384100, +0.02485300]|[-0.01541460, +0.02652066]|
|cosqa|PS−L|S3|+0.00655420|[-0.01205323, +0.02486696]|[-0.01161185, +0.02472025]|
|cosqa|PS−L|top1_gain|-0.01427124|[-0.03628325, +0.00488926]|[-0.03513717, +0.00659468]|

## Fixed historical candidates, changing training weights only.

The perspective is fully reworded to the same rewrite/rank. Changes relative to L are used for diagnostic choice changes; The difference in perspective is not interpreted as the only causal breakdown. The table below is a descriptive point estimate. See full area DESCRIPTIVE.csv.

|Domain|Branch - L|S3 Change|Top1 Change in net income|
|---|---|---:|---:|
|source|O−L|-0.00019486|+0.00320838|
|source|S−L|+0.00208092|-0.00299221|
|source|N−L|+0.00000000|+0.00000000|
|source|W−L|+0.00417612|-0.00815829|
|source|P−L|-0.00222529|-0.00505630|
|source|PN−L|+0.00640470|+0.00791135|
|source|PS−L|+0.00354554|-0.00204629|
|cosqa|O−L|+0.00337320|-0.01000757|
|cosqa|S−L|+0.00008304|-0.00393094|
|cosqa|N−L|+0.00000000|+0.00000000|
|cosqa|W−L|+0.00364561|-0.00923236|
|cosqa|P−L|+0.00253087|-0.00671564|
|cosqa|PN−L|+0.00033114|-0.01358002|
|cosqa|PS−L|+0.00942028|-0.01259211|

## Interpretation and review

- The present round quantifys all the differences between historical 93750 step models and unauthorised models because of their configuration under a fixed limited training budget.
- Literally open linear100000 at 1024 step only slight decay; PN Individually check the scaling of training, which cannot be combined.
- Inter-territory reflecting uncertainty in reference/relationship groups on regular training examples; PS Provides only one training of random sensitivity, not a substitute for multiple seed training overall differences.
- ENTR/PROB / Accurate random pooling of budgets; Relative random advantages are separated from relative NoRewrite net gains. NLL Change is not a substitute for retrieving gains.
- Original H Recognition, formal five pointers, historical weights and results maintained; Not to replace the official model with this round, risk_certified = false.

Independent audit: TRAIN_AUDIT.json Check common data/updates/lr/negative comparisons; EFFECT_AUDIT json Re-place and independently calculate ranges from the original ranking by Fraction and another CDF formula; MODEL_AUDIT.json Independent physical forward, FP64 entropy and greedy regenerated.
Full result RESULTS.json; Main change PRIMARY.csv; The remaining pairs DESCRIPTIVE.csv; Each query and each candidate is per_query.jsonl, candidate_RR.jsonl.
