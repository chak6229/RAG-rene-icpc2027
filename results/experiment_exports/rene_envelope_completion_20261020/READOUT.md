# Two sets of T5 weighting evidence completed and independently reviewed

Date: 2026 - 10 - 20. The known evidence of the historical four-arm summary is not confirmed by new independence. Reuse original to generate subsets for candidate and original 512 query/ group matching 512, without retracing or regeneration T5. The new calculation is a freeze CodeBERT for base candidate by-election ranking and full score.

## Key findings

The following is relative to original query MRR@ 100 Discrepancies, item-by-line sampling by associated group 95% District. Two weights for each domain using the same 512 query against the original query; Not with the 2048 source or 1490 development of a cluster denominator.

|Data|T5 Weight|Top1 Δ|95% Range|best3 Δ|95% Range|
|---|---|---:|---|---:|---|
|source|local|-0.03025188|[-0.04521606, -0.01620265]|+0.02998920|[+0.01828534, +0.04234607]|
|source|t5base|-0.01336644|[-0.02433999, -0.00262637]|+0.03852459|[+0.02586949, +0.05180257]|
|cosqa|local|-0.02019729|[-0.03674618, -0.00434649]|+0.06130778|[+0.04602760, +0.07699339]|
|cosqa|t5base|-0.04270304|[-0.06296258, -0.02299205]|+0.05669980|[+0.03920094, +0.07485249]|

After correction across eight comparisons, all four best3-minus-original-query intervals are positive. For Top1, the source/base and CoSQA/local intervals span 0, while the other two are negative. All four Top1 point estimates have the same sign, and two directions remain unresolved under correction. The direct best3 differences between weights span 0, leaving the ordering of the weight conditions unresolved.

CoSQA/base ENTR Single-precise flat single to +.03475321, 95% [+.01882089,+.05044681]. So Top1 lower than original query does not mean that entropy is sorted without information. The "sign failure" or "generator-free" interpretation in the old ENVELOPE.md is no longer a conclusion of this mechanism; The old document is kept for tracking.

## Closed chain of evidence executed

- source/base 5995 candidate positions, 6283 unique search input, 19210 language documents; CoSQA/base is 3345, 3855, 6267.
- Original query and all base candidate recode: Freeze pooler_output, Ununified, FP32, TF32 Closed, maximum length 256,batch16; Code vector reuse and check the original document ID order.
- Save hologram score with Top100: Break parallel to original document ID ascending in descending order of FP32 points. source About 87.46 seconds and 50.19 seconds with CoSQA retrieval steps; Without original T5 generation/training and independent audit, not as a full-link online delay.
- Independent recalculation of 10138 full text Top100, 144855715 full language score, FP32 error 0; 18680 Candidates RR, 16384 Query indicators and inter-group matching.
- Independent model forward checks for 16 queries and 16 corpus items gave maximum error 0. Across 16 FP64 dot-product checks, the maximum absolute difference was 5.955e−5, within the 1e−4 check tolerance. This check does not establish identical full FP64 and FP32 rankings.
- The four historical summaries matched with maximum difference 3.33e−16. All 6 evaluation-contract tests passed. Hashes matched for 59 frozen and protected files and 10 newly sealed outputs.

## The role of papers

It closes the gap “only base aggregate, no ranking and query evidence” by providing a robust diagnosis of two sets of T5-base weights in the same sample. The two weights belong to the same family and the sample is not unknown; It is not referred to as cross-model family extension or four independent confirmations. The core contribution remains assigned SSQR reproduction and A direct against the pool.

## Recount & File

- `PROTOCOL.json `、`RANKINGS_SEALED.json `: Freezing input and ranking calibre.
- `rankings/*/{Top100.jsonl,texts.json,query_vectors.npy,scores.npy}`: text, vector, full score and final ranking.
- `per_query.jsonl `、`candidate_RR.jsonl `、`RESULTS.json `_: a complete itemized result.
- `AUDIT.json `、`TEST_RESULTS.json `、`FINAL_ACCEPTANCE.json `: Actual acceptance and inspection.
- Independent entrance:`experiments/rene_envelope_completion_20261020/audit.py --root <workspace>`. Complete mode requires original model/ vector/freezing input; compact incremental package use `--cached `, it's only ranked to RR/zone.

No re-evaluation of H labels/ranks, no change in original confirmation, CURRENT / STABLE or risk configuration; risk_certified=false.
