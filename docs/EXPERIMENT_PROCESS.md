# Experiment process and design

This guide describes the logical dependencies among the archived experiments. Study design/protocol files preserve their original status; packaging does not convert post-hoc diagnostics into new confirmations. Dates in directory names identify experiment bundles, not human-review process duration.

1. **Establish the reconstruction contract.** Recover the SSQR insertion/scoring procedure, inspect teacher-forced full-vocabulary ENTR/PROB, verify CodeBERT pooling/dot products, check query/data roles and overlap. Preserve the distinction between recoverable public implementation and unavailable author checkpoint/training CSV/production aggregation.
2. **Bridge and evaluate the external setting.** Apply the frozen reconstruction to AdvTest and CoSQA with their own corpora/qrels. Preserve the original RAND20 CoSQA confirmation and its group structure. Raw differences across datasets do not isolate query wording.
3. **Separate candidate count and aggregation.** Reuse each condition's identical candidate pool. Compare ENTR/PROB with exact uniform random sampling without replacement at k=1/2/3. Compute S, B and G from query-level paired values; repeated candidate texts keep their original position probability mass. Best-k uses labels and is an opportunity measure.
4. **Measure executable actions.** Compare fixed ENTR Top1 adoption using four grouped OOF Ridge heads and matched Fixed20 allocations; compare RRF3/RRF4 fusion. On frozen full H, track gain selection, actual cross-encoder verification, risk fallback, positive-opportunity retention, ranking loss and work already charged.
5. **Check conditions.** Carry the same definitions into StaQC, a third retriever (UniXcoder), matched T5 weights, and the new 25,600-group paired-weight confirmation. Keep original comparison families and descriptive controls separate.
6. **Diagnose with human review.** Use the fixed eight-stratum, 16-query-per-stratum sample from CodeBERT H8530. Two reviewers evaluate query clarity, rewrite meaning and displayed code relevance. Then add 344 supplemental code judgments per reviewer on the same 128 queries. Retain A/B unknowns/disagreements. Compute weighted event/type rates, expanded-relevance sample bounds and separate finite-frame sampling uncertainty.
7. **Bound implementation sensitivity.** Six generation profiles and eight budget-matched 1,024-update training arms test stated reconstruction choices. These do not restore all effects of historical 93,750-update training.

The experiment map links these steps to code, inputs, results and replay coverage. `results/TABLES.json` retains original cell-to-source pointers and hashes. `provenance/FILE_LINEAGE.json` connects archived source files to their delivered copies. Machine-local intake paths are normalized for portability; label values, reasons, formulas and frozen numerical inputs are preserved.

## Formula and stage semantics

S(k,a) is the mean, across the designated queries, of selector RR@100 minus matched exact-random RR@100. B = S(1,mean) − S(3,mean); G = S(3,mean) − S(3,best3). Their sum is the change along that stated comparison path, not a unique causal attribution. A positive B/G means the relative selector advantage decreases along the corresponding step.

Mean aggregation is expected RR from adopting one selected candidate uniformly. Best3 is relevance-label-assisted selection. RRF creates a new ranking; candidate best3 is not an upper bound on a fused ranking. Full-H gain is actual policy RR change from the original query. Verification work is retained even if a later gate falls back.

All conditions keep their own sample roles, denominators, grouping, correction families and interval rules. The two corrected interval methods and nonzero-group checks remain as originally fixed. Unresolved differences are not equivalence. Displayed code panels are not a full-corpus relabeling.
