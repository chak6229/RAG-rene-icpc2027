# Study Design

## Reconstruction and evaluation populations

We reconstruct SSQR's insertion-position procedure using independently trained CQC T5 weights. The rewriter enumerates insertion positions and scores generated spans by full-vocabulary teacher-forced predictive entropy (ENTR) or generation probability (PROB). CodeBERT uses the verified pooler/raw-dot retrieval contract. A BM25 implementation provides a lexical comparator; UniXcoder-base uses its public zero-shot encoder-only retrieval configuration \cite{guo2022unixcoder}. Exact checkpoint identities, tokenization, training, generation and ranking choices are recorded in the artifact and implementation-fidelity table.

The author checkpoint could not be supplied. Original training CSV bytes and the unpublished production aggregation are unavailable. Our target is therefore the specified method under an explicit executable reconstruction. Source-data bridges, six generation profiles, aggregation audits and eight budget-matched training configurations assess concrete reconstruction choices. The training sensitivity holds 32,768 rows and 1,024 updates fixed; it does not reconstruct all effects of the historical 93,750-update run.

|Population|Queries / groups|Use|
|---|---|---|
|CoSQA D|1,490 / 512|Development, factorial decomposition, adoption and fusion controls|
|CoSQA capped H|7,718 / 1,963|Frozen ENTR-RAND20 best3 comparison|
|CoSQA full H|8,530 / 1,963|Predefined descriptive execution-stage accounting|
|Original-data bridge|2,048 / 594|Same-implementation source comparison and factorial diagnostic|
|StaQC Python|13,312 / 13,312|New-setting frozen B/G confirmation; 16,384-snippet corpus|
|Matched T5 subsets|512 / 512 per domain|Paired local/base weight sensitivity|
|New matched T5-weight confirmation|25,600 / 25,600|Previously unused StaQC components; fixed 32,768-snippet corpus|
|Human diagnostic sample|128 / 120|Eight strata from the full-H CodeBERT frame|

CoSQA contains web queries paired with code targets. Source inspection distinguishes its query field from the target docstring and audits code, query and connected-group overlap with training and earlier evaluations. It is an external query source, not a guarantee of unseen code or repositories. StaQC contains mined Stack Overflow titles paired with accepted single snippets \cite{yao2018staqc}; source, licensing, known exposure and query/code/associated-group overlap were checked before effect access. One query per associated group is selected. Corpus, target judgments and task difficulty differ between settings, so their comparison does not isolate query wording as a cause. All samples retain their distinct roles and denominators.

## Candidate-count and aggregation contrasts

All within-condition comparisons share the same already generated insertion-position pool, target judgments and corpus. ENTR and PROB select their first k ranked positions for k=1,2,3, with position-based tie breaking and k capped by pool size. Exact random draws k positions uniformly without replacement. Duplicate texts retain their original position probability. The CoSQA replication retains the original frozen RAND20 comparator; subsequent factorial analyses use the exact random expectation.

For query q, mean aggregation averages the selected candidates' RR@100, equivalent to expected RR under uniform adoption within that selected set. Best-k takes their largest RR and therefore uses relevance labels. It excludes an extra original-query arm. For selector advantage S(k,a), averaged over the declared queries, define

```text
S(k,a) = RR(selector,k,a) - RR(exact random,k,a)
B = S(1,mean) - S(3,mean)
G = S(3,mean) - S(3,best3)
J_exact = B + G
```

B changes selected count while holding mean aggregation fixed; G changes aggregation while holding count at three. Positive components indicate a reduction in the selector's advantage relative to random. They follow a specified arithmetic path, rather than identifying unique causal shares. Here k is selected count from a fixed pool, not generation calls or measured cost. The earlier combined contract contrast uses a RAND20 endpoint and remains separately identified.

## Executable policies and stage accounting

Fixed-action adoption controls use five frozen OOF folds, the same ENTR Top1 action, and four Ridge heads: common evidence (M0), generation confidence added (MT), original-query reference added (MR), and both (MRT). Fixed20 adopts the highest-scored 20% of eligible actions within each fold, with all development queries retained as the evaluation denominator. These intervals condition on the fitted OOF models. Zero-threshold and matched-count comparisons retain distinct roles.

RRF3 fuses the same ENTR three lists; RRF4 also includes the original-query list. The fixed reciprocal-rank constant is 60 with equal weights over unique input texts and Top100 lists. Fusion creates a new ranking, so candidate best3 is neither executable nor its upper bound. Input counts quantify work, not elapsed online latency.

The frozen full-H pipeline evaluates gain selection, actual cross-encoder verification and risk fallback. Net gain is the mean change in labeled-target RR. Harm frequency counts negative RR changes, which need not imply semantic error. Positive-gain retention uses the total nonnegative gain available from the same candidates with an original-query safe arm. If U is this available gain and D the realized change, then U-D=(U-max(D,0))+max(-D,0). We separately track unretained positive opportunity, realized ranking loss and paid verification work. A later rejection does not refund earlier verification cost. Empirical harm counts do not issue a population risk certificate.

## Statistical roles

The CoSQA primary comparison was frozen before H effects: CodeBERT ENTR-RAND20 best3, capped at 16 hash-selected queries per connected group. Both two-sided 95% cluster-t and 10,000-group-bootstrap intervals must lie within ±0.005 for the declared equivalence conclusion. The 0.005 margin concerns this comparison and is not a universal significance or user-value threshold.

The same-pool development factorial analysis is post-hoc and adjusts 24 comparisons: B/G for ENTR and PROB plus their direct differences in four domain/retriever cells. The StaQC design fixes two CodeBERT ENTR B/G comparisons before new effects and uses Bonferroni-adjusted cluster-t and group-bootstrap intervals. Its precision amendment occurred before effect access and is retained; the protocol is locally sealed, not externally preregistered. PROB and BM25 are prespecified descriptive controls.

UniXcoder's 12-comparison family contains B/G and paired differences from CodeBERT and BM25 in two consumed settings. The matched two-weight family has 24 comparisons: two domains, two selectors, B/G and local/base/base-minus-local. Directions follow each study's fixed rule requiring both adjusted interval methods and its nonzero-group guard. Unresolved differences do not establish equivalence. Each study retains its original comparison family; unified presentation does not pool populations or create a new confirmation.

A separate outcome-blind paired confirmation sampled 25,600 previously unused StaQC query/connection groups and fixed a 32,768-snippet corpus. The local CQC T5 and public T5-base share queries, targets, corpus and the frozen CodeBERT retriever; each retains its complete generated position pool. The six-comparison family comprises within-weight ENTR B/G for each weight and paired t5base-minus-local B/G differences. The local protocol was sealed before new generation and effect access but was not externally preregistered. A directional result requires Bonferroni-adjusted cluster-t and shared group-bootstrap intervals to exclude zero on the same side, plus at least 20 nonzero groups. The 0.005 weight-difference planning target is a precision goal, not an equivalence margin; this sample, its family and its results remain separate from the historical 512-query sensitivity and prior StaQC confirmation.

## Human diagnostic review

Before collecting labels, we fixed eight disjoint strata in the consumed CodeBERT H8530 frame, covering final positive/negative decisions, rejected positive/negative/zero decisions and remaining opportunity/action conditions. Sixteen queries per stratum were selected by salted hash order. Each of two Python-proficient reviewers assessed 383 distinct rewrites and 327 code panels across 128 queries. Packets hid method names, scores, ranking outcomes, strata and known-target roles, and separated semantic assessment from code relevance. Code panels included the known target and selected displayed retrieval witnesses, not a complete new corpus judgment.

We analyze the final submitted labels. Semantic exact agreement and nominal Cohen kappa use the rewrite unit. Code relevance includes unknown as a nominal category; linear weighted kappa additionally uses the jointly judged 0/1/2 subset with coverage reported. Query events and issue types use the original N_h/n_h stratum weights. For each indicator, simultaneous stratum-wise hypergeometric bounds yield a conservative fixed-frame 95% interval under the pseudorandom hash-sampling interpretation. These intervals describe sampling uncertainty in H8530, not future populations or annotation error. Both judgments and disagreements are preserved; interpretation of examples does not alter labels or constitute reviewer adjudication.

A separate additive-relevance diagnostic was fixed after the original outcomes and labels were visible and before supplementary judgments were collected. The union of the original and three fixed candidate Top-3 lists contains 610 query/code-text panels: 266 inherit the original judgments, and both reviewers independently assess the remaining 344 across 126 of the same queries. Supplementary packets hide old labels, method identities and retrieval outcomes. For each reviewer, the expanded relevant set retains original positives and adds code graded 2; including grades 1/2 is a separate sensitivity. The primary diagnostic is RIDGE_verified minus the original query at RR@3 under the original eight-stratum weights. Shared text-level unknown assignments yield exact paired sample identification bounds. These bounds omit sampling and annotation uncertainty and are distinct from both confidence intervals for query-event rates and the full-H MRR@100 estimates.

A subsequent post-hoc uncertainty analysis retains the original eight-stratum design and constructs sampling envelopes for the H8530 means of both identification endpoints. With the final relevance rule treated as fixed, one-sided hypergeometric CDF inversion on the full RR support and Bonferroni tail control yield conservative finite-frame 95% envelopes; Neyman variance with finite-population correction and Welch–Satterthwaite t intervals supplies a separately labeled approximation. We report reviewer/threshold combinations separately and add a conservative family envelope for the four RIDGE_verified RR@3 reviewer-by-threshold combinations. No simultaneous coverage is claimed for all 48 diagnostic combinations. The inference uses the pseudorandom within-stratum sampling interpretation, includes no annotation-error model, and does not adjust for data-informed choice of the diagnostic rule. Full methods are in relevance_sampling_uncertainty.md.
