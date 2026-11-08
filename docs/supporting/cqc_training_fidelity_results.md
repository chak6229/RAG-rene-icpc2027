# Bounded CQC training-configuration sensitivity

Supplementary manuscript material, 2026-10-20. The reconstruction is sensitive at the candidate and selection levels, while changing the tested optimization settings does not establish a positive family-adjusted improvement over the budget-matched local configuration. The length limit is an exact negative control in this experiment.

## Controlled design

We train eight CQC variants from the same original T5-base initialization on the same 32,768 CODEnn rows, one fixed contiguous 15% corruption per row, and the same order. Each variant receives 1,024 updates with microbatch 8 and accumulation 4, initial learning rate 1e-3, gradient clipping at 1, BF16 computation and FP32 parameters. The final update is always evaluated; validation NLL is descriptive and neither early stopping nor outcome-based checkpoint selection is used. The 32,768 training rows contain 26,155 distinct texts.

|Arm|Optimizer|Schedule horizon|Length cap|Weight decay|Training seed|
|---|---|---|---:|---:|---:|
|L|AdamW|constant|512|0.01|101|
|O|Adafactor|constant|512|0.01|101|
|S|AdamW|linear100000|512|0.01|101|
|N|AdamW|constant|128|0.01|101|
|W|AdamW|constant|512|0|101|
|P|Adafactor|linear100000|128|0|101|
|PN|Adafactor|linear1093|128|0|101|
|PS|Adafactor|linear100000|128|0|42|

Adafactor uses explicit learning rates with relative_step=False and scale_parameter=False. Public code hard-codes a 100,000-step linear horizon and overrides the configured seed 101 with set_seed(42). We preserve the literal horizon in S/P/PS. PN is a separately named progress-scaled sensitivity, using ceil(1024*100000/93750)=1093. It does not stand for the literal public schedule. P aligns the tabulated hyperparameters while retaining the shared local corpus, single-span corruption, batch budget and precision.

We retain the existing bridge samples: 512 source queries in 512 repositories and 512 consumed CoSQA development queries in 512 connected groups, searching the full 19,210 and 6,267-document corpora with the same frozen CodeBERT pooler. For each weight, we evaluate both its own generated candidates and rescoring the same historical local candidates. Inference uses the same positions, greedy 19-new-token decoding, and FP32 full-vocabulary teacher-forcing entropy. Probability scoring and exact uniform position sampling remain controls. Repeated texts at different positions retain their probability mass.

The main paired outcomes are the change relative to L in S3 (ENTR best3 minus exact RAND best3) and in ENTR Top1 minus NoRewrite. Seven non-L arms, two domains and two outcomes form 28 predefined comparisons. We report 10,000 original-group bootstrap and cluster-t intervals with Bonferroni adjustment. Intervals condition on the trained weights, not a population of training seeds. Fixed-pool, probability, direct seed and progress-scaling contrasts are descriptive. No fresh confirmation is claimed.

## Main results

|Arm minus L|Source: change in S3|CoSQA: change in S3|Source: change in Top1 gain|CoSQA: change in Top1 gain|
|---|---:|---:|---:|---:|
|O|-0.00361666|+0.00386055|-0.01083141|-0.00507920|
|S|-0.00279246|+0.00134978|-0.01095998|+0.00245741|
|N|+0.00000000|+0.00000000|+0.00000000|+0.00000000|
|W|+0.00149885|+0.00308944|-0.00890648|-0.00733077|
|P|-0.00114332|-0.00396651|-0.01179079|+0.00148675|
|PN|+0.00507610|+0.00279995|-0.00693821|+0.00555303|
|PS|+0.00240475|+0.00655420|-0.01611474|-0.01427124|

For the public hyperparameter bundle P minus L, source S3 changes by -0.00114332 (family-adjusted bootstrap interval [-0.01347887, +0.01066236]) and CoSQA S3 by -0.00396651 ([-0.02318952, +0.01510376]). Top1 gain changes by -0.01179079 on source ([-0.02949673, +0.00379373]) and +0.00148675 on CoSQA ([-0.01792189, +0.02302523]). These intervals do not establish equivalence or negligible configuration effects.

No primary contrast has positive adjusted intervals in both statistical methods. One contrast, source PS minus L in Top1 gain, is negative: -0.01611474, adjusted bootstrap [-0.03271290, -0.00059109] and cluster-t [-0.03221445, -0.00001503]. Four length-control contrasts are exactly zero; 23 other intervals cross zero. PS minus L changes the public bundle and seed together and cannot identify a seed-only effect. In the direct descriptive PS minus P comparison, CoSQA Top1 changes by -0.01575799 (pointwise bootstrap [-0.03027118, -0.00147269]); this single seed pair does not estimate seed-population variance.

## Candidate and selection sensitivity

The shared training inputs and labels never exceed 78 and 26 tokens. Caps 128 and 512 therefore produce identical tensors: L and N have identical loss/gradient trajectories, weight-file hashes, generated candidates, scores, choices and retrieval outcomes. The separate one-mask-per-row scan of one million comments finds 22 inputs above 128 and no labels above 128; it is not an audit of every historical epoch mask.

The bundle P changes 73.61% of generated source position texts and 56.74% of CoSQA texts relative to L. Even with historical candidate texts and rankings held fixed, its ENTR Top1 position changes on 195/512 source and 186/512 CoSQA queries. Fixed-pool P minus L changes S3 by -0.00222529/+0.00253087 and Top1 gain by -0.00505630/-0.00671564 (source/CoSQA); all four descriptive intervals cross zero. Differences between own-pool and fixed-pool contrasts are not a unique causal decomposition of generation and selection.

Progress scaling improves CQC validation NLL from 1.21728411 to 1.14316240, but the direct PN minus P own-pool retrieval contrasts all cross zero. Source/CoSQA S3 changes are +0.00621942/+0.00676645 and Top1 changes are +0.00485258/+0.00406629. Rescoring the fixed source pool yields positive pointwise intervals for both outcomes; the corresponding CoSQA changes are unresolved. These descriptive comparisons are not used to choose a replacement model. The predefined nonadditivity contrasts likewise do not resolve an individual interaction.

Across the eight configurations and two domains, best3-minus-original point estimates remain positive and Top1-minus-original point estimates remain negative. These are 16 related sensitivity cells, not 16 independent confirmations; several Top1 intervals cross zero. Best3 uses relevance labels and measures candidate opportunity rather than a deployable policy.

## Interpretation and reproducibility

This controlled sensitivity study quantifies optimization differences as a concrete reconstruction factor and excludes length truncation for its shared bank. None of the tested changes establishes a family-adjusted positive improvement over L or removes the observed candidate-to-Top1 separation at the point-estimate level. The study does not assign a percentage of the original replication discrepancy to training: 1,024 updates cannot identify all differences between the historical 93,750-update model and an unavailable author checkpoint. Corpus bytes, corruption conventions and the full historical trainer are not restored by the hyperparameter bundle. No query-origin causal conclusion follows.

Training records were independently rederived for all 32,768 examples and 8,192 updates. Independent Fraction/CDF calculations reproduce 149,440 candidate reciprocal ranks and all summary/paired statistics (maximum statistical discrepancy below 5e-16). The audit checks 45,305 text rankings and recomputes Top100 ordering from 544,003,702 stored new retrieval scores; it does not independently repeat every numerical dot product. Actual-model checks cover 192 T5 score cases, including 96 greedy regenerations, and 64 CodeBERT query vectors. Maximum independent FP64 entropy discrepancy is below 9.62e-7.

Evidence and complete adjusted intervals: results/rene_cqc_training_fidelity_20261020/{RESULTS.json,READOUT.md,PRIMARY.csv,DESCRIPTIVE.csv,EXTRA_CONTRASTS.json,FINAL_ACCEPTANCE.json}. Models and dense scores: external_inputs/cqc_training_model_store/. Figure: cqc_configuration_effects.pdf in the result folder. The original confirmation, historical weights, existing manuscript sources and formal deployment pointers are preserved.
