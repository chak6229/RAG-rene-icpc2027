# Human review results and scope

Two independent reviewers are identified as A and B. Each originally assessed 128 queries, 383 rewrites and 327 code panels. Each then assessed 344 supplemental panels within the same query sample, producing 688 additional judgments. A's independent supplemental completion was confirmed separately. Unknowns, disagreements and item-level rationales remain intact, without automatic adjudication.

Supplemental exact agreement is 55.23%, with nominal Cohen κ=.321. Jointly judged coverage is 296/344 (86.05%), with linear-weighted κ=.484. Among 154 disagreements, 48 involve unknown and 106 differ between known grades. Each reviewer retains 49 unknown judgments in the merged 610-panel pool.

Original semantic agreement is 76.50% (κ=.525). Code-relevance agreement including unknown is 53.82% (κ=.371); jointly judged coverage is 79.51%, with linear-weighted κ=.569.

## Relevance diagnostic

The prespecified endpoint is RIDGE_verified, relevance threshold 2 and RR@3 under the original eight-stratum weights. Values compare the fixed policy with the original query. These are sample identification bounds over unknown relevance, not confidence intervals; they exclude sampling and annotation error.

| Reviewer | Before supplemental judgments | After supplemental judgments | Identified queries |
|---|---|---|---:|
|A|[-0.017841, +0.041899]|[+0.005843, +0.018263]|102 → 121/128|
|B|[-0.032886, +0.059318]|[+0.001279, +0.016604]|96 → 122/128|

The original-qrels endpoint is +0.001594. Changes relative to the original qrels are A [+0.004250, +0.016669] and B [-0.000315, +0.015011]; B's range includes 0. Including grades 1/2 gives weighted RR@3 ranges [+0.003297,+0.005007] and [+0.001948,+0.010069]. At grade 2 without weighting, the ranges are [+0.003906,+0.023438] and [-0.006510,+0.011719].

The corresponding finite-frame sampling envelopes cross zero. The positive weighted sample ranges therefore do not establish positive utility across H8530. See the English [Sampling Uncertainty Analysis](../docs/supporting/relevance_sampling_uncertainty.md).

## Role in the paper

Supplemental judgments strengthen the diagnostic account of the specified evaluation contract and fixed-policy utility. RIDGE_verified has positive weighted sample bounds for both reviewers at both thresholds; direct ENTR Top1 remains negative, while safe best3 opportunity remains positive. Positive bounds after risk fallback do not hold across both reviewers and both thresholds. This does not confirm positive full-H MRR@100 utility, a stable deployment benefit or mutually exclusive causal shares. The second contribution remains supported by same-pool B/G comparisons, new-sample confirmation and retriever-condition evidence.

Original stimuli, reviewer identities and process times are not included with anonymous labels. Opaque references support pairing within this dataset. The `note` fields contain English translations of the archived rationales. The reviewers’ category labels and the resulting agreement and relevance analyses are preserved.
