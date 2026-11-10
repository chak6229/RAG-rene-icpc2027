# Results and evidence evaluation contract

This English guide records the evaluation rules used to interpret the current artifact. Obsolete editorial and duplicate documentation files are kept in the author-side archive; the packaging changes are recorded in `provenance/SCOPE_CLEANUP_20261110.json`. Table labels S1–S15 correspond to historical T1–T15; their 176 result records remain unchanged. The manuscript is supplied separately.

## Outcomes and references

The primary outcome is MRR@100. NoRewrite, ENTR Top1 and label-assisted best3 are reported separately. Best3 does not add NoRewrite; safe candidate opportunity explicitly adds the original query. The designated best3 evaluation contract takes the maximum RR within each query before averaging. Rank-wise aggregate-MRR maxima and the public duplicate-URL reader behavior are separate sensitivity analyses; the original production aggregation was not recovered.

Selection methods share the generated position pool and ranking cache. Exact random selection samples positions uniformly without replacement, retaining probability mass for duplicate texts. RAND20 denotes 20 random repeats and does not increase query or group sample size. The selected count is capped at pool size. Per-query changes use the declared full-query denominator. Missing method rows, repeated query/position rows, duplicate ranked documents and wrong ranking depth are rejected rather than silently removed.

Realized ΔRR is used for evaluation, not as an unobserved deployment feature. Adoption is determined from an actual ranking change, rather than a nonzero gain. Risk fallback restores the complete original Top100 list. Historical harm accounting uses ΔRR<−1e−10 for the labeled target; it is not a human semantic-error label. Unlabeled documents are not thereby judged irrelevant.

Positive-gain retention uses the total positive gain available from the same candidate set with the original-query safety option. A ratio of post-gate to pre-gate net changes would not measure retention when net gains approach zero. Cross-encoder pairs and tokens record work already incurred; a later rejection does not recover that work. Matched-supervision controls compare no-CE and actual-CE updates on the same verification allocation.

## Design and inferential roles

The frozen CoSQA primary comparison is CodeBERT ENTR−RAND20 under a maximum of 16 fixed-hash queries per group: 7,718 queries and 1,963 groups. The revised single-comparison design replaced an earlier four-comparison plan before effects were opened. At δ=0.005 and two-sided α=0.05, the plan required 1,836 groups and used 1,963; the recorded 82.58% planning calculation assumes transferable D variance and independent groups. It is not an 80% power guarantee for practical equivalence.

Practical equivalence requires both the cluster-t interval and the 10,000-resample group-bootstrap 95% interval to lie within ±0.005, together with the recorded changed-group requirement. This frozen rule is more conservative than conventional 90%-interval TOST. An interval crossing zero alone establishes neither no effect nor equivalence.

PROB, BM25, full-H sensitivity, stage differences and matched-supervision CE increments retain their descriptive roles. The source bridge and six generation profiles are post-confirmation diagnostics whose new-effect designs were fixed in advance; they are not unseen confirmations. Direct effect differences preserve covariance from shared groups. H8530 has been consumed, so subsequent policy development belongs on D.

| Evidence | Sample and role | Reader table or report |
|---|---|---|
| Primary external comparison | CoSQA 7,718/1,963; specified practical-equivalence target | S1; Replication Results |
| Original-data bridge and endpoint contrasts | AdvTest 2,048/594 and CoSQA D 1,490/512; post-hoc diagnostics | S2; Replication Results |
| Fixed-action adoption | D; four Ridge heads, five group folds; intervals conditional on fitted OOF scores | S3 and continuations |
| RRF3/RRF4 | D; constant 60, equal weights for unique text; four adjusted versus-original comparisons | S4 and continuations |
| Complete H stages | 8,530/1,963; frozen selection, verification and fallback | S5 |
| Earlier matched-weight endpoint analysis | 512/512 per domain; eight adjusted effects | S6 |
| Historical UQ targets | Separate 1,536- and 1,024-repository targets and intervals | S7 |
| Same-pool B/G | AdvTest and CoSQA D; ENTR/PROB and direct component differences; 24 comparisons | S8 |
| StaQC confirmation | 13,312 groups, 16,384 codes; two CodeBERT ENTR primary comparisons | S9 |
| UniXcoder extension | Previously used CoSQA D and StaQC; 12 comparisons | S10 |
| Complete matched-weight B/G | 512/512 per domain; 24 comparisons | S11 |
| New paired-weight confirmation | 25,600 new groups, 32,768 codes; six comparisons | Main Table II; paired-weight report |
| Independent human review | 128 queries/120 groups; 383 rewrites and 327 code panels per reviewer | S12–S14 |
| Training sensitivity | Eight arms, 32,768 rows, 1,024 updates; 28 comparisons | S15 |

Each family retains its original correction and role. Direction requires both adjusted interval methods to satisfy the declared condition and the applicable nonzero-group threshold. The new six-comparison weight confirmation uses 20,000 common-group bootstrap resamples and at least 20 nonzero groups. Four within-weight ENTR B/G directions are positive; the direct weight differences remain unresolved. Its .005 difference-precision target is not an equivalence margin. The older 2/12/24/28-comparison families are not merged or reselected for presentation.

## Selected count and aggregation

S(k,a)=RR(selector,k,a)−RR(exact random,k,a), averaged over the designated queries. B=S(1,mean)−S(3,mean), G=S(3,mean)−S(3,best3), and J_exact=B+G. The selected count k concerns a fixed generated pool, not generation calls. Mean aggregation is expected RR under uniform adoption; best-of-k is label-assisted opportunity. The decomposition describes a specified arithmetic path, not unique causal shares.

The historical endpoint difference uses RAND20 best3 and differs from J_exact: CoSQA/CodeBERT +0.02556783 versus +0.02515642. Their random references must not be interchanged. RRF generates new rankings, so candidate best3 is not its upper bound. Mean unique RRF3/RRF4 inputs of 2.9799/3.9799 measure counts rather than latency. Relative-to-random position-selection advantage and relative-to-original executable utility retain distinct references.

## Independent review and relevance uncertainty

Initial human review preserves final independent A/B labels, unknowns and disagreements. Linear-weighted κ on 260 jointly judged panels is reported with 79.5% coverage. Query-event rates use eight-stratum N_h/n_h weights and simultaneous hypergeometric stratum bounds for a conservative finite-frame 95% interval. Events overlap and do not represent mutually exclusive causal shares. The inference concerns the fixed H8530 frame and excludes annotation error.

The post-hoc relevance diagnostic was specified after original effects and labels were visible but before supplemental judgments. It keeps the original 128 queries/120 groups, eight-stratum weights and fixed policies. Expanded relevance retains original positives and same-text matches, then adds grade-2 code; grades 1/2 provide a separate sensitivity condition. Policy and original-query RR use the same RR@1 or RR@3 cutoff, distinct from full-H MRR@100.

The combined Top3 pool contains 610 text panels: 266 inherit labels and 344 receive new judgments from each reviewer, covering 126 original queries. Each reviewer retains 49 unknowns. Shared-text unknown assignments yield 6,144 exact paired query bounds across 48 combinations. The primary diagnostic is RIDGE_verified at grade 2 with weighted RR@3: A [+0.005843,+0.018263], B [+0.001279,+0.016604]. Original-qrels utility is already +0.001594. B's change from original qrels and strict unweighted result still cross zero.

These are sample identification bounds, not confidence intervals. Separate finite-frame inference interprets salted-hash selection as within-stratum simple random sampling without replacement. Full-support hypergeometric CDF inversion and Bonferroni correction produce conservative 95% envelopes; Neyman finite-population variance with a t reference gives approximate envelopes. All primary individual and four-combination simultaneous envelopes cross zero. Coverage is conditional on treating the final diagnostic rule as fixed and is not adjusted for its data-informed choice. It does not establish full-H positive utility, equivalence to zero or future-population effects.

Supplemental agreement is 55.23%, nominal κ0.321; jointly judged coverage is 296/344 with weighted κ0.484. The 154 disagreements remain. Different-panel κ values do not identify a change in review quality. Case interpretation does not adjudicate the original labels; any later consensus would be an additional version.

## Verification scope

The artifact separates cached-rank/label statistical replay, small actual-model checks recorded historically, and full model training/data acquisition. Passing the first two does not establish the third. Historical matched-weight validation included 144,855,715 corpus FP32 scores and 10,138 Top100 lists, with model spot checks of 16 queries and 16 code snippets. Those 32 examples are not a claim of complete independent re-encoding; the four old summaries differed by at most 3.33e−16. FP64 spot checks do not replace the FP32 tie-ordering evaluation contract.

Dependencies, hashes, acquisition routes, rights and table-specific replay coverage are documented in the current README and EXPECTED_RESULTS.md. Current release validation does not claim new retraining, online latency measurements or new effect confirmation.
