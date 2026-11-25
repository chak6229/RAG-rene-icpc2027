# CQC training configurations: controlled comparison

2026-10-20. This document records the execution method. See `RESULTS` and `FINDINGS` for retrieval results after independent verification.

## Research question

With CQC data, order, masks, initialization and update budget held fixed, how do the local and public-source optimization settings affect SSQR position selection and automatic single-selection utility?

Source bridging, six generation configurations and entropy and aggregation audits provide prior evidence. This experiment varies training settings to quantify reconstruction fidelity.

## Shared conditions

- Initialize every arm from local original t5-base weights, without resuming the historical adapted checkpoint.
- Draw a fixed 32,768 rows from the original one-million-row CODEnn set and apply a shared contiguous 15% single-span mask once. Row order matches at each step. The 32,768 training rows contain 26,155 distinct texts.
- Each arm uses 1,024 updates, microbatch8×gradient accumulation4, effective batch32 and one complete pass. The initial learning rate is .001 with gradient clipping at 1, BF16 computation and FP32 parameters.
- Use the fixed final-step weights without early stopping or checkpoint selection on validation or retrieval. The 1,024 validation examples supply token NLL only.
- Use training seed 101, except PS uses 42 with the same data, masks and order.
- Set CuBLAS to `:4096:8` before initialization. Partial launches without this environment setting were archived before effects were opened. All eight official arms started from scratch.

## Configurations and contrasts

|ID|Optimizer|Total movement control steps|Maximum length|weight decay|Seed|
|---|---|---|---:|---:|---:|
|L|AdamW|constant|512|.01|101|
|O|Adafactor|constant|512|.01|101|
|S|AdamW|linear100000|512|.01|101|
|N|AdamW|constant|128|.01|101|
|W|AdamW|constant|512|0|101|
|P|Adafactor|linear100000|128|0|101|
|PN|Adafactor|linear1093|128|0|101|
|PS|Adafactor|linear100000|128|0|42|

Adafactor explicitly uses `scale_parameter=False, relative_step=False`. The public source hard-codes 100000 total steps. Its literal schedule is still at .00098977 at update 1024. PN uses 1093 steps from ceil(1024×100000/93750) to assess training-progress scaling, with a final learning rate of .00006404. PN is a sensitivity arm.

P aligns only the listed hyperparameters. It retains the local data and training pipeline instead of substituting the authors' unavailable CSV, the public preprocessing's multiple masks or the full trainer. The experiment does not recover the original checkpoint or replace the historical 93750-step training.

## Two domains and two evaluation views

Reuse the existing source512 queries in 512 repositories and CoSQA development512 queries in 512 linked groups, with complete corpora of 19210/6267. Use the same frozen CodeBERT pooler without retraining.

1. Generate and retrieve each arm's own candidates to measure end-to-end changes across training settings.
2. Hold historical local candidate text and ranks fixed and rescore with the CQC weights for each arm to measure selection changes on the same candidates.

All arms use positions 1…n, greedy19 new tokens, batch8, FP32, TF32 disabled and dropout disabled. ENTR uses full-vocabulary teacher-forcing entropy, and PROB uses mean label log probability. RAND is the exact uniform expectation over original positions without replacement. Duplicate texts retain their positions' probability mass. RAND20 is reported separately for the historical comparison.

S3=ENTR best3−RAND_exact best3. Top1 net utility=ENTR Top1−NoRewrite. Best3 uses relevance labels to measure candidate opportunity. Report advantage over random separately from net gain over the original query.

## Statistical analysis and verification

Before opening effects, the primary family was fixed at 7 non-L arms×2 domains×{change in S3, change in Top1 net utility}=28 contrasts. Pair queries and use 10000 original-group bootstrap resamples and cluster-t intervals with Bonferroni correction. Fixed-candidate, PROB, PS−P and PN−P results are descriptive. Effects did not determine the family or production configuration.

Intervals condition on the actual training runs and do not cover seed-to-seed variation. An interval spanning zero leaves direction unresolved and does not establish equivalence. Cross-domain differences require direct contrasts.

Independent checks reconstructed the training bank, updates, learning rates and weights, recalculated RR, exact random selection and group intervals from ranks with Fraction/CDF, and spot-checked T5 regeneration, FP64 entropy and CodeBERT forward passes. Every Top100 order was independently checked against the newly retained dense scores.

This training-fidelity diagnostic reused previously used data and left H closed. It supplies neither a new confirmation nor risk certification. The five production pointers, historical T5/CodeBERT and earlier confirmation and results were retained.
