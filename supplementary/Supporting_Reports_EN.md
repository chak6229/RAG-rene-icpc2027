# Supporting reports for the external replication of SSQR

Release 2026.11.08. These reports match the names used in the current English manuscript. Numerical evidence and original human labels remain unchanged. Use ../docs/APPENDIX_INDEX.md for individual files and ../docs/RESULT_PATH_MAP.md for historical source identifiers.

## Implementation Fidelity Table

Manuscript material, 2026-10-19. The implementation table describes the completed reconstruction. Bridge results are supplied by the separately frozen supplementary experiment; they do not replace the CoSQA confirmation.

### Reconstruction contract

We reconstruct SSQR's position-selection experiment using a T5-base CQC rewriter and a CodeBERT retriever. ENTR, PROB, and RAND operate on exactly the same cached position-level rewrites. Each policy selects three positions, or all available positions when fewer than three exist. We evaluate the best reciprocal rank among those selected rewrites at depth 100. The original query is evaluated separately and is not added to best-of-three. Twenty fixed RAND permutations reduce Monte Carlo noise in random position selection; they do not increase the number of independent queries.

The author checkpoint is unavailable: the authors explicitly confirmed that it could no longer be supplied. Our weights are therefore independently trained artifacts, identified by their full hashes. We distinguish the published method, choices recovered from public code, and operational choices required for an executable reconstruction. This distinction matters because the article and code specify different insertion-position, decoding, and training details.

| Component | Published article / public implementation | Executed reconstruction and evidence |
|---|---|---|
| Evaluation query | CodeXGLUE: joined `docstring_tokens`; CoSQA web-query text is `doc` | Canonical CodeXGLUE token sequence in the bridge; CoSQA `doc` copied verbatim. Target docstrings are never substituted for CoSQA queries. |
| CodeBERT objective | Public model uses the pooler, raw dot product, and full query-to-code in-batch cross-entropy | Same representation and objective; 251,820 training pairs, batch 32, two complete epochs / 15,740 steps. Actual forward vectors and loss were independently checked. |
| CodeBERT optimization | Article: constant learning rate 5e-5. Public training path: linear schedule | Public-source linear schedule with 10% warmup, seed 123456, BF16 training; FP32 retrieval with TF32 disabled. The training contract records runtime and optimizer settings. |
| CQC training examples | Article: CODEnn first 1M comments and one continuous 15% mask. Public preprocessing/trainer: CSV cleanup and multiple length-three spans | Local HDF5-derived first 1M; a single continuous `round(0.15*n)` span. Original author CSV bytes were not recovered. |
| T5 optimization | Article: three epochs, learning rate 1e-3, batch 32. Public trainer: Adafactor / linear schedule, length 128, a hard-coded seed 42 despite a seed-101 configuration | Local AdamW / constant schedule, length 512, seed 101; completed 93,750 steps, source-validation best at step 92,000. Both training procedure and checkpoint identity are explicit. |
| Position set | Article prose: n+1 positions. Algorithm 1 and public JSON generation loop: positions 1 through n | Confirmation uses 1 through n. The bridge compares 1…n with 0…n on identical common-position candidates. |
| Generation | Article describes sampling and m=10. Public JSON code calls bare `model.generate(input_ids)` without preserving the historic generation configuration | Confirmation uses explicit greedy decoding, decoder length at most 20 including start. Sensitivity includes a decoder-length-10 operationalization and one fixed sampled output per position; temperature 1, no top-k/top-p truncation. |
| Entropy | Public JSON path scores raw teacher-forced full-vocabulary probabilities for `sentinel0 + decoded span + sentinel1` | Same label wrapper and all non-padding positions, including EOS; independent FP64 entropy checks. Lexical-token-only entropy is a separately labeled sensitivity. |
| PROB | Article reports a probability baseline; no corresponding public implementation was recovered | Declared mean teacher-forced label log probability. It is a probability control, with no assertion that the original unpublished normalization was recovered. |
| Three-candidate aggregation | Article §5.1 describes reporting the best of three reformulations. The public evaluator stores predictions in a URL-keyed dictionary and retains the last row when reformulations share a URL; it does not explicitly implement either per-query maximum RR or maximum of the three rank-wise aggregate MRRs | The confirmation explicitly defines per-query maximum RR. A separately frozen, cache-only diagnostic compares that definition, the maximum rank-wise MRR, and last-selected-row behavior. The public reader's behavior does not establish which unpublished evaluation path produced the article tables. |
| Source search pool | Article: all 19,210 source-test code segments. Public retrieval path builds code vectors from its supplied test file, including repeated rows when reformulations are concatenated | Bridge searches the 19,210 unique original test rows in author-split order. Identical code rows from different original identities remain distinct. The 45,244-row auxiliary source file and historical 4,096 pool are not used as the primary bridge corpus. |
| Ranking ties | Public code uses reverse NumPy quicksort; reformulation retrieval also strides repeated code rows | Our deterministic order is descending score then ascending original document ID. Cached confirmation rankings are preserved. Exact historical production tie order is not claimed. |
| Short queries | Public JSON generation skips queries with fewer than k words; original source-test queries have at least three words | New-query evaluation retains every frozen query, uses `min(3,n)`, and inserts no identity padding. Candidate-count distributions are reported. |
| Second retriever | Original article uses Java Lucene and StandardAnalyzer | Local BM25 with a fixed lexical tokenizer, k1=1.2 and b=0.75; a second retrieval mechanism, not an exact Lucene replication. |

The local CQC training differences remain part of the reconstructed model's identity. Correct query fields, entropy arithmetic, and the CodeBERT objective have been independently verified; those corrections alone do not establish identity with an unavailable author checkpoint.

### Source-data bridge and finite implementation sensitivity

The main CoSQA comparison was frozen and evaluated before this supplementary study. We retain that result unchanged. To examine whether the reconstruction itself explains a missing ENTR advantage, we apply the same fixed rewriter, retriever, position policy, ranking metric, and RAND seeds to a deterministic 2,048-query sample of the original CodeXGLUE AdvTest dataset. All 19,210 original source-test functions form the search corpus. The sample covers 594 source repositories. The corresponding CoSQA bridge arm reuses all 1,490 consumed development queries in 512 connected groups, over the fixed 6,267-function corpus.

The source sample is selected by a fixed hash without inspecting retrieval outcomes. All additional CoSQA generation uses the development role. Historical source tests and CoSQA development data have already informed earlier work, so this bridge provides supplementary diagnostic evidence rather than another independent confirmation.

We predefine six generation profiles: greedy decoding with total decoder limits 20 and 10, each at positions 1…n and 0…n, followed by sampling at limit 10 under both position sets. Here the limit includes the decoder start token: 10 corresponds to nine newly generated tokens. This is an explicit operationalization of the article's m=10 because its exact historical token-count convention was not recoverable. Each profile is run on the same 512 queries per dataset, with one deterministically selected query per group. We also compare full-label and lexical-span entropy masks.

Position comparisons reuse identical common-position generations. For sampling, each query-position pair has a deterministic RNG seed derived from seed 101 and its identity. Adding position zero cannot consume random draws belonging to other positions. This is one generated sequence per position, not a seed ensemble. Candidate scoring, selection, and retrieval are sealed before supplementary effects are opened. No profile is chosen to replace the already evaluated confirmation policy.

An additional source-interface inspection identified the public evaluator's duplicate-URL overwrite behavior. Before any new bridge effects were opened, we froze an aggregation diagnostic using the same caches: per-query maximum RR, maximum of three rank-wise aggregate MRRs, and the last selected candidate. The latter reproduces the public reader's handling of repeated URLs, not necessarily the authors' actual production evaluation. Every main and sensitivity query has at least three whitespace words, so these comparisons use three real positions without changing the query denominator. For the non-smooth maximum of aggregate means, simultaneous cluster-t bounds on all 66 policy-by-rank means are propagated conservatively; an outcome-selected winning rank is not treated as a prespecified deployment policy. This supplement is documented in `results/rene_ssqr_aggregation_audit_20261019/` and does not change the frozen confirmation's estimand.

The target contrast is ENTR best-of-three minus the mean RAND best-of-three across twenty repetitions. We report this contrast within each dataset and directly estimate its source-minus-CoSQA difference. Connected groups are linked across datasets through recovered repository identities, retaining covariance when repositories are shared. Query-weighted cluster-t and 10,000 group-bootstrap intervals accompany the estimates. Finite profile changes use paired differences on the same queries. These supplementary intervals describe uncertainty; they are not added to the original confirmatory hypothesis family.

Changing datasets also changes the corpus, query lengths, difficulty, and relevance judgments. The bridge tests whether the fixed reconstruction's effect depends on the evaluation environment. It does not identify query wording alone as a causal mechanism. Likewise, a positive source estimate and an interval crossing zero on CoSQA are not, by themselves, evidence that the two effects differ: the directly estimated effect difference is the relevant comparison.

### Reproducibility record

T5 weights: `7acd4cb8379a1f5297bb978a748bc002b708ef5fd60a8b4737a6f8c7d2ebcaa5`.

CodeBERT weights: `0d533c04602916c22d1f13d282056f081b6bd04f9866ccc1ded46a0516e13e3d`.

Supplementary protocol: `results/rene_ssqr_bridge_20261019/PROTOCOL.json`, SHA-256 `7d3d4c0728aa413a21dfc748d4a01dc4f29d5d59a0b4a44591eb48793dc3d95e`.

Executable package: `experiments/rene_ssqr_bridge_20261019/`. The artifact distinguishes cached numerical replay from actual model inference and from retraining. Earlier CoSQA confirmation, source snapshots, and formal deployment configuration are preserved.


## Replication Targets and Result Mapping

The original study is SSQR. Original numerical values below come from the retained public arXiv HTML version; they do not establish item-by-item identity with the formally published version. The method citation and implementation conventions are recorded separately. The original authors' checkpoint, original training CSV and unpublished production aggregation were not recovered.

| Original claim or use scenario | Public source evidence | Corresponding condition and result in this study | Interpretation |
|---|---|---|---|
| ENTR has an advantage over random position selection | Original Table 4 reports CodeBERT ENTR 0.2221 and RAND 0.1886 | Under the documented reconstruction, the 2,048-query AdvTest bridge gives ENTR−RAND20 best3 −0.00435302, group-bootstrap 95% interval [−0.00822558,−0.00033732] | The bridge does not show the original positive difference; model and aggregation identities differ, so this is not exact numerical recovery of the original production system |
| Position-selection advantage extends to an external query source | Extension of the original result to another query source | Capped CoSQA H: 7,718 queries, 1,963 groups; ENTR−RAND20 best3 −0.00083652, group-bootstrap 95% interval [−0.00325479,+0.00154146]; both prespecified intervals are within ±0.005 | Practical equivalence under the designated reconstruction and per-query best3 evaluation contract; the margin is not a universal threshold |
| Single- and three-candidate performance differs | Original Table 5 reports 0.1726 for one position and 0.2221 for three positions | Same-pool exact random selection separately varies selected count and mean/best-of-k aggregation; CoSQA/CodeBERT S moves from +0.027936 through +0.014716 to +0.002779 | B/G are components along the designated path; their direction, size and new-sample confirmation are measured, not unique causal shares |
| Several rewrites are recommended for user selection | The original paper describes users choosing a rewrite consistent with their intent | This study adds automatic adoption, verification updates and risk fallback, with fixed-action, fusion and full-H stage analyses | Automatic execution is an extension; failure to establish positive net utility does not directly refute the user-selection scenario |

The direct original-data-bridge minus CoSQA D contrast is −0.00672102, with group-bootstrap 95% interval [−0.01364477,+0.00045645]. Covariance from 34 shared cross-domain connected groups is retained. This direct difference is unresolved; separate within-domain signs do not establish a domain difference. Corpus, relevance judgments and difficulty also differ, so query wording is not isolated as a causal factor.

The implementation-fidelity table, six generation configurations, aggregation audit and finite-budget training comparisons document reconstruction choices. The 1,024-update sensitivity study does not recover all effects of historical 93,750-update training. It neither changes the frozen CoSQA comparison nor establishes complete attribution to historical training differences.


## Replication Results

Supplementary source-data and sensitivity results; the primary CoSQA confirmation remains unchanged.

Under the same frozen greedy reconstruction, ENTR minus RAND20 best-of-three is -0.00435302 on the 2,048-query original-data bridge (group-bootstrap 95% [-0.00822558, -0.00033732]), and +0.00236800 on the 1,490-query CoSQA development arm ([-0.00354366, +0.00796936]). The directly estimated source-minus-CoSQA contrast is -0.00672102 ([-0.01364477, +0.00045645]), with 34 shared repository-connected groups retained in the covariance estimate.

These are post-confirmation supplementary diagnostics, selected before their new effects were opened. The estimand remains the frozen per-query maximum-RR candidate opportunity. A source-interface audit additionally found duplicate-URL overwrite in the public evaluator; a cache-only supplement compares that behavior with per-query and rank-wise aggregation, without asserting which unpublished path produced the original article's results. The published author's checkpoint was unavailable; differences in local CQC training remain documented, and the bridge changes corpus, judgments, query length, and difficulty as well as query origin. It therefore does not identify query wording as an isolated cause.

The finite sensitivity study uses the same 512 queries per dataset across six predefined position, length, and decoding profiles. Common-position generations are identical within position comparisons, and sampled generations use position-local RNG. Full results, paired profile changes, lexical-entropy controls, and independent artifact checks are linked in the supplementary readout. No sensitivity profile replaces the already frozen CoSQA confirmatory policy.

Aggregation sensitivity is material. On the same CoSQA development candidates, ENTR−RAND20 is +0.00236800 for the mean of per-query maxima, compared with +0.02294983 for the maximum of the three rank-wise aggregate MRRs. The latter has a prespecified simultaneous-bound 95% interval of [-0.09028707, +0.13570056] and is not a confirmed advantage. These estimands cannot substitute for each other; neither the external equivalence finding nor the public duplicate-URL behavior establishes the original production table's aggregation.

The confirmatory CoSQA estimate is −0.00083652 MRR@100 for ENTR−RAND20 on the capped-16 design (7,718 queries; 1,963 groups). Both prespecified 95% intervals fit within ±0.005. As an interpretable scale, 0.005 is about 14.9% of the original Table-4 gap of 0.0335, or the average RR change produced if 1% of queries move from rank 2 to rank 1. This is a metric-scale justification, not a claim to have measured user satisfaction or a universal application threshold.

Candidate opportunity remains distinct from automatic realized utility: on all 8,530 CoSQA confirmation queries, the ENTR best-of-three gain, without adding NoRewrite, is +0.05655770 for CodeBERT and +0.01358236 for BM25. The fixed downstream-stage decomposition quantifies what survives selection, verification, and fallback. The implementation-fidelity evidence supports interpreting these results as a controlled external reconstruction with explicit choices, rather than as an exact numerical recovery of unavailable author artifacts.


## Provenance and Meaning of the Frozen Risk Working Point

The full-H pipeline inherits the fixed probability threshold of 0.075 from the historical development protocol. During selected-action risk development, raw-tight and the additional risk heads applied the same rule: predicted harm probability must not exceed 0.075. The subsequent cleaned three-class anchor-risk pipeline retained this operating point. The full-H design sealed it again; H labels were not used for fitting, threshold selection or a threshold sweep.

This provenance establishes inheritance of a fixed operating point. It does not establish that 0.075 is optimal or derive a certified calibration formula from 0.1.

| Quantity | Role | Interpretation |
|---|---|---|
| threshold = 0.075 | Threshold on predicted harm probability for the proposed action | Exceeding it restores the complete original-query Top100 list without selecting another action; incurred verification work remains |
| target_harm = 0.1 | Historical target for the harm rate among adopted actions | Distinct from the prediction threshold; it does not certify population risk for full H |
| Observed harm rate or count | Empirical result under a particular sample and policy | Reported separately from positive-gain retention, net change and the adoption denominator |

Threshold provenance, source hashes and the current frozen-design pointer are recorded in `provenance/review_fixes/risk_working_point_provenance.json`; historical model definitions are excluded. The full-H statuses `risk_certified=false` and `deployment_policy_promoted=false` remain unchanged. Stage results describe the gain–loss trade-off at this frozen operating point; they do not establish an optimal gate or a result for every threshold.


## Review Cases and Disagreement Analysis

This discussion analyzes the original human-review returns and the completed supplementary code-relevance assessments. It preserves both judgments and interprets their relationship to the cached retrieval outcomes. It does not create new reviewer labels or a consensus adjudication. Original agreement and weighted rates are reported in mechanism_review_return_results.md, with source labels in results/rene_review_returns_20261020/. Completed supplementary labels and paired relevance diagnostics are in results/rene_review_completion_20261021/ and relevance_sensitivity_results.md.

### Where the judgments differ

There are 90 semantic-class disagreements among 383 rewrites. Sixteen additional rewrites have matching semantic classes but different issue types, yielding 106 rewrite disagreements. Among 327 code panels, 151 relevance labels differ; 35 of the 128 query-clarity labels differ. At least one difference occurs in 122 sampled queries. These counts refer to displayed items and are not frame-weighted prevalence estimates.

|A / B semantic class|Count|Interpretation for discussion|
|---|---:|---|
|drift / preserved|47|Whether additional wording constrains the task or expresses assumed context|
|preserved / drift|7|The reverse boundary; semantic disagreement is not one-directional|
|drift / uncertain|14|A definite alternative reading versus insufficiently determined meaning|
|malformed / uncertain|16|Incoherent text versus an unresolved intended request|
|malformed / preserved|2|Whether a readable original task can still be recovered|
|drift / malformed|2|Changed task versus inability to state a coherent task|
|preserved / uncertain|2|A recoverable original task versus residual ambiguity|

These are complete class-pair counts. The interpretations name discussion issues, not a new exhaustive thematic coding of the 90 texts. For example, adding `http` to `set python request headers` is drift for A because it restricts the protocol, but preserved for B because it expresses the usual request-header context. Adding `value` raises a similar boundary. The factual question is whether the added wording changes acceptable answers under the original request, not simply whether a new token appears. By contrast, both reviewers call `python get linux from clipboard` drift, but A marks an unsupported constraint and B an operation change. This illustrates type disagreement within an agreed semantic problem; both original type labels are retained.

Code disagreement has two distinct numerical components. Sixty-five panels have a numeric label from A and unknown from B (29 at A=2, 32 at A=1 and four at A=0). The remaining 86 differences are between numeric grades: B is lower on 82 and higher on four. B uses unknown on 67 panels in total, A on two. Missing helper implementations, surrounding classes and query context recur in their written reasons; full versus partial functionality is also a substantive issue. The jointly judged subset therefore cannot replace the full-panel result: its linear-weighted kappa 0.569 is reported with 260/327 coverage, alongside full-panel nominal kappa 0.371.

Query clarity differs in 28 A-clear/B-ambiguous cases, four A-clear/B-insufficient cases, two A-ambiguous/B-clear cases and one A-ambiguous/B-insufficient case. A labels 124/128 queries clear, whereas B labels 94 clear. The resulting 72.7% exact agreement and kappa 0.029 reflect strongly imbalanced marginal labels; neither alone certifies query clarity. These judgments help explain why a short query can admit more than one relevance interpretation.

### Example selection and provenance

We use the lexicographically first query ID with each of three already computed query flags: any definite semantic issue, meaning-preserved verified-stage loss, and displayed non-target relevance. This is a transparent post-hoc illustrative rule, not a representative or independently sampled case series. Rates and uncertainty come from all 128 queries under the original eight-stratum design, not from these examples. The query IDs retain upstream `dev`/`train` names; study membership follows the frozen H manifest rather than these strings.

Full original notes, code panels, candidate fractions, sample weights and decisions are exported in results/rene_manuscript_integration_20261021/REVIEW_CASES.json and REVIEW_CASES.md, with input hashes and label-file line references. The three cases below concern the CodeBERT H frame.

### Case 1: an unsupported constraint need not change target RR

**cosqa-dev-400.** Original: `python output colors from a string`. Candidate position 5 is `python output colors from a json string`. Both reviewers label drift/unsupported constraint: the original did not require a JSON input. Its target RR stays at 1/5, so its delta is exactly zero. The best candidate in the same pool, `python output the colors from a string`, is marked preserved by both and improves target RR from 1/5 to 1/3. The other preserved candidate reaches rank four.

The gain, verified and risk stages retain the original query in this case; no cross-encoder work is allocated. Thus the existence of a drifted candidate is not proof that it caused the policy's loss, and semantic drift can be invisible to this single-target RR outcome. The case also contains a preserved positive opportunity that the policy does not adopt. Its separate indicators may overlap at the query level.

### Case 2: a preserved request can lose labeled-target rank

**cosqa-dev-106.** Original: `get index of an item in list python`. The adopted position-6 rewrite is `get index of an item in the list python`. Both reviewers mark preserved/none, describing `the` as an article that leaves the index-finding request intact. Target RR changes from 1/16 to 1/18, delta -1/144. Another both-preserved candidate, `get index value of an item in list python`, reaches rank 14, delta +1/112. The policy's cached gain score orders the adopted action above this positive alternative, providing a concrete observed prediction/action mismatch.

The RIDGE_verified stage makes the same action and allocates zero cross-encoder pairs here. The stage name must not be interpreted as evidence that the cross-encoder inspected this query and made an erroneous judgment. The risk policy retains the action. This case supports the claim that the observed RR loss is not automatically a semantic-drift label; it does not identify which retriever, label-coverage or learning intervention would remove the loss.

The displayed code also contains a substantive relevance disagreement. Document 456 returns `index_or_name` unchanged if it is an integer, and otherwise calls `lst.index(index_or_name)`. A grades it fully relevant (2); B grades it partially relevant (1), noting that an integer value is treated as an index rather than searched as an element. This distinction follows an actual code branch. For document 6214, which searches through `self.iter()`, A assigns 2 while B uses unknown because the iteration behavior is not shown. The known target uses bisection; both grade it partial because the original request does not ensure sorted input. These three different judgments should not be collapsed into a single rater-style explanation.

### Case 3: relevant displayed alternatives outside the known target

**cosqa-dev-198.** Original: `set python request headers`. The selected rewrite, `set the python request headers`, is marked preserved by both. The known target moves from rank one to rank two, delta -1/2. The top displayed alternative, document 3330, constructs a `requests.Session` and calls `self.req_session.headers.update(self.headers)`. Both reviewers grade it fully relevant (2), although it is outside the known-target labels. This is direct displayed-panel evidence that the target-only RR loss can coexist with a result assessed as useful for the request.

The gain and verified stages adopt the position-1 rewrite without cross-encoder allocation; the risk stage restores the original. The known-target panel calls an unseen `self.with_header` helper: A grades it 2, B uses unknown. The same query's `http` rewrite supplies the added-context semantic disagreement discussed above. These observations flag relevance coverage and context boundaries; they do not estimate full-corpus missing-label prevalence or prove that relabeling would eliminate the aggregate policy gap.

### Supplementary code-relevance disagreements

The same two reviewers independently assess 344 additional panels across 126 of the original queries. These are new code judgments, not additional queries, rewrite judgments or reviewers. The original 106 rewrite, 151 code and 35 clarity disagreements remain unchanged.

|Supplementary relevance grade|A|B|
|---|---:|---:|
|2: fully relevant|21|58|
|1: partially relevant|103|146|
|0: not relevant|172|140|
|u: unknown/context insufficient|48|0|

Supplementary exact agreement is 55.23%, with nominal kappa 0.321 including unknown. The jointly judged subset covers 296/344 panels (86.05%); its linear weighted kappa is 0.484. Of 154 disagreements, 48 involve A's unknown and B's numeric judgment. The remaining 106 differ between numeric grades: B is higher on 83 and lower on 23. This direction differs from the original panels' numeric-grade pattern and does not establish that either reviewer is invariably stricter. The panels also differ between batches, so their kappas cannot measure a change in assessment quality.

In the merged 610-panel Top-3 pool, 266 panels inherit original labels and 344 receive supplementary labels. Each reviewer retains 49 unknown judgments. Jointly judged coverage is 513/610 (84.10%), with linear weighted kappa 0.527; nominal kappa including unknown is 0.340. These coverage figures describe the merged pool, not all 327 original code panels plus the supplement.

#### Additional illustrative cases

For each of the 0/2, u/2 and 1/2 disagreement categories, we take the lexicographically first query_id/panel_id pair after receiving the labels. This transparent post-hoc choice illustrates functional boundaries; it is not a prespecified representative sample or an exhaustive thematic coding of the 154 disagreements. Full code and both original notes are in results/rene_review_completion_20261021/REPRESENTATIVE_DISAGREEMENTS.json.

**Request versus response: cosqa-train-13132, `python get post content type`.** The `json_or_text(response)` implementation reads the response's Content-Type and decodes its body. A assigns 0 because it does not read the POST request's content type; B assigns 2 because Content-Type processing and parsing are usable. The disagreement concerns which functional object must satisfy the request.

**Missing helper versus visible function: cosqa-dev-169, `python if type array`.** The `is_integer_array` function calls an unseen `is_np_array` helper before testing an integer dtype. A uses unknown because the array-type check cannot be confirmed from the displayed context; B assigns 2 based on the visible NumPy integer-array check. The different labels expose how missing context enters relevance assessment.

**Transformation versus completed output: cosqa-dev-400, `python output colors from a string`.** The `highlight` function returns a string wrapped in ANSI control codes. A assigns 1 because it supplies a display-ready transformation without an output step or parsing colors from the string; B assigns 2 because the ANSI-wrapped string can display color. The disagreement concerns what completion of the short request requires.

These accounts preserve the observed 0/2, u/2 and 1/2 pairs without choosing a winning interpretation. They supplement the three original semantic/ranking cases. Their relationship to the aggregate utility reading is tested through separate reviewer-specific relevance sets and exact unknown-relevance bounds, not by merging labels or assigning causal percentages.

### What is resolved by this discussion

The final-label agreement, complete class-pair counts, weighted issue/event rates, completed supplementary coverage and traceable examples now form a coherent diagnostic account. It distinguishes semantic changes, meaning-preserved ranking effects, displayed alternative relevance, functional boundaries and missing context. Both original ratings and all unresolved judgments remain available. No semantic class is reassigned and no new consensus rate is substituted for the reported agreement.

The remaining scientific boundary is causal attribution: generation, retrieval, relevance coverage and decision learning have not been intervened on independently. The paper uses the present evidence to explain what observed outcomes can mean, rather than assigning exclusive shares of the unrealized gain. Any later reviewer consensus should be recorded as an additional version, with pre-consensus labels retained.


## Relevance Sensitivity Results

### Question and fixed analysis

How sensitive are the sampled policies' measured utilities to adding relevant code outside the original target labels? This post-hoc diagnostic uses the same 128 queries in 120 groups, the original eight-stratum weights and fixed CodeBERT actions. Prior method outcomes and original human labels were visible before the diagnostic was designed. Its protocol, supplementary packets and primary endpoint were locally fixed before supplementary judgments; it is not an independent method-effect confirmation.

The reference remains the original query. Original qrels are preserved. For each reviewer separately, an expanded relevant set includes original positive targets, identical-text counterparts within the pooled lists, and code that reviewer grades fully relevant (2). Including partial relevance (grades 1/2) is a separate sensitivity. These additive rules test dependence on observed relevance coverage; neither reviewer's expanded labels replace the original ground truth.

We evaluate RR@1 and RR@3 with the same cutoff for original and rewritten lists. A pooled panel is the union of the original and fixed three candidate Top-3 lists, merged by exact code text. Unknown judgments remain unknown. Shared text-level assignments yield exact paired identification bounds. These are descriptive sample ranges, not confidence intervals; they omit sampling and annotation uncertainty. The separate sampling analysis below and in relevance_sampling_uncertainty.md adds design-based uncertainty without changing those ranges. The original full-H MRR@100 estimates retain their original scope.

### Completed collection and agreement

Both reviewers independently complete all 344 supplementary code judgments across 126 of the original 128 queries. The complete pooled Top-3 set contains 610 distinct query/code-text panels: 266 inherit original judgments and 344 receive new judgments. Original assessments of 383 rewrites and 327 code panels per reviewer remain unchanged; resubmission does not add queries or reviewers.

|Supplementary relevance grade|A|B|
|---|---:|---:|
|2: fully relevant|21|58|
|1: partially relevant|103|146|
|0: not relevant|172|140|
|u: unknown/context insufficient|48|0|

Supplementary exact agreement is 55.23% and nominal Cohen kappa is 0.321. The jointly judged subset covers 296/344 panels (86.05%), with linear weighted kappa 0.484. Of 154 disagreements, 48 pair A's unknown with B's numeric grade and 106 differ between numeric grades; B is higher in 83 of those 106 and lower in 23. These are displayed-panel counts, separate from the original frame-weighted query-event rates.

In the 610-panel pool, each reviewer retains 49 unknown judgments. Jointly judged coverage is 513/610 (84.10%), with linear weighted kappa 0.527; full-pool nominal kappa including unknown is 0.340. Completion concerns returned assessments, not the elimination of unknown relevance or disagreement. Original and supplementary panels differ, so their kappas do not measure a change in reviewer quality. Detailed label pairs and three additional functional-boundary examples are in review_disagreement_discussion.md.

### Primary diagnostic: weighted RIDGE_verified RR@3

The primary endpoint specified before supplementary judgments is RIDGE_verified minus the original query at RR@3, using grade-2 additions and the original eight-stratum weights. The original-qrels value for this same sampled endpoint is +0.001594. Supplementation narrows the expanded-relevance bounds as follows.

|Reviewer|Before supplementary judgments|After supplementary judgments|Identified query contrasts|
|---|---|---|---:|
|A|[-0.017841, +0.041899]|[+0.005843, +0.018263]|102 → 121/128|
|B|[-0.032886, +0.059318]|[+0.001279, +0.016604]|96 → 122/128|

Both completed ranges are positive under this rule. This is range narrowing within the sampled endpoint; its original-qrels value was already positive. The change relative to that original-qrels contrast is A [+0.004250, +0.016669]; B [-0.000315, +0.015011]. Reviewer B's change range includes zero even though B's expanded contrast is positive.

|RR@3 sensitivity for RIDGE_verified|A: policy minus original|B: policy minus original|
|---|---|---|
|Grades 1/2, original stratum weights|[+0.003297, +0.005007]|[+0.001948, +0.010069]|
|Grade 2, unweighted|[+0.003906, +0.023438]|[-0.006510, +0.011719]|

Including grades 1/2 gives positive weighted bounds for both reviewers, identifying 126 query contrasts for A and 124 for B. Under strict grade-2 relevance without the original stratum weights, B's range still includes zero. The positive direction therefore concerns the declared weighted diagnostic and does not hold across all weighting choices.

### Sampling uncertainty around the primary ranges

We retain the original eight-stratum, 16-queries-per-stratum design and frame weights, treating each reviewer's final relevance rule and fixed actions as given. Under the pseudorandom hash-as-SRSWOR interpretation, one-sided hypergeometric CDF count limits on the full RR support yield conservative 95% finite-frame envelopes. A stratified Neyman variance estimate with finite-population correction and Welch–Satterthwaite t critical values supplies an approximate precision reference.

|Grade-2 RIDGE_verified RR@3|Conservative finite-frame 95% envelope|Stratified-t approximate 95% envelope|
|---|---|---|
|A|[-0.400821, +0.417409]|[-0.008645, +0.033373]|
|B|[-0.401856, +0.420320]|[-0.009318, +0.033184]|

Both methods span zero for A and B at both relevance thresholds; the additional conservative envelopes simultaneous across all four reviewer/threshold combinations also span zero. The exact construction is wide because it covers both endpoints, all eight strata and all full-support CDF thresholds with 16 observations per stratum. The t reference is approximate. Neither method includes reviewer error or a correction for choosing the diagnostic rule after original outcomes and labels were visible. Positive sample identification ranges therefore do not establish positive utility across the H8530 frame. The complete methods, threshold-1 and simultaneous results are in relevance_sampling_uncertainty.md; all 48 combinations are in results/rene_relevance_sampling_uncertainty_20261021/RESULTS.json.

### Fixed-policy comparison under expanded relevance

|Grade 2, original stratum weights, RR@3|A: policy minus original|B: policy minus original|
|---|---|---|
|ENTR_top1|[-0.040360, -0.006583]|[-0.045258, -0.017126]|
|RIDGE_verified|[+0.005843, +0.018263]|[+0.001279, +0.016604]|
|RIDGE_risk|[-0.000177, +0.000324]|[+0.001755, +0.001962]|
|SAFE_ENTR_best3_opportunity|[+0.093181, +0.118574]|[+0.047352, +0.103379]|

ENTR Top1 has negative weighted bounds for both reviewers at both relevance thresholds; label-assisted safe best3 opportunity has positive bounds in all four cases. This preserves the distinction between available candidate opportunity and direct adoption. The verified-policy diagnostic shows how relevance coverage can change the utility reading of a fixed action. Risk fallback does not have positive bounds across both reviewers and both thresholds. These sample results do not establish a positive full-pipeline deployment effect or allocate exclusive causal shares to semantic drift, relevance coverage and decision-learning error.

### Earlier RR@1 result, unchanged by supplementation

The original Top-1 panels were already covered, with unknown labels retained. For RIDGE_verified at threshold 2, the weighted 128-query original-qrels RR@1 difference is -0.018318. Reviewer A's expansion gives +0.007547, a change of +0.025865. Reviewer B gives [-0.000440,+0.007547], with change [+0.017878,+0.025865]. The paired contrasts are identified for all 128 queries under A and 127 under B. Supplementation adds no Top-1 labels, so these values remain unchanged. Their cutoff and relevance definition differ from the original full-H MRR@100 result.

### Artifacts and reporting role

The supplementary packets show only the original query and code in independent anonymous orders, hiding old labels, method names, rewrite texts, ranks, known-target roles, strata and the other reviewer's judgments. Original labels, reasons, unknowns and disagreements remain intact.

The fixed design is experiments/rene_relevance_sensitivity_20261021/DESIGN.md. Completed evidence is in results/rene_review_completion_20261021/: RESULTS.json contains all 48 reviewer/threshold/cutoff/policy combinations; query_bounds.jsonl contains 6,144 exact paired query records; supplement_labels.jsonl and supplement_disagreements.jsonl preserve all new labels and disagreements. REVIEW_ACCEPTANCE.json records completed collection, and FINAL_ACCEPTANCE.json validates this manuscript integration.

The original existing-label analysis remains in results/rene_relevance_sensitivity_20261021/preview/. The earlier content/numerical audit remains in results/rene_review_final_reanalysis_20261021/ as a historical version. The sampling-uncertainty extension is results/rene_relevance_sampling_uncertainty_20261021/ and is integrated into the current manuscript under results/rene_contribution_strength_20261021/. These diagnostics remain separate from the frozen 15-table, 179-row current bundle and do not rerun or revise its B/G, confirmation, retriever, weight or full-H effect estimates.


## Sampling Uncertainty Analysis

### Target and design

The completed relevance diagnostic in relevance_sensitivity_results.md reports identification ranges over unknown code relevance in a reviewed sample. This analysis separately quantifies sampling uncertainty in those ranges. Its target is the finite CodeBERT H8530 query frame, with each reviewer's final relevance rule, policy and cutoff treated as fixed. The original eight strata and their frame sizes are retained; each contributes 16 queries selected by salted hash order. Inference interprets this procedure as pseudorandom simple random sampling without replacement within strata. Queries are the sampling units; the 120 associated groups are not a replacement sampling design.

For each query, shared-text unknown assignments produce lower and upper paired RR contrasts, l(q) and u(q). We estimate each finite-frame endpoint by the original weighted mean, sum_h (N_h/8530) mean_h. Reviewer A and B and relevance thresholds 2 and 1 remain separate. The intervals include sampling uncertainty and unknown-relevance identification uncertainty. They do not estimate annotation error or a population of reviewers, replace the original qrels, or change the full-H MRR@100 result.

This is an analysis after the original effects, labels and sample ranges were visible. The relevance diagnostic was specified before supplementary judgments, not before the original sample was reviewed. We therefore report design-based intervals conditional on treating the final rule as fixed; the nominal coverage is not a guarantee adjusted for data-informed choice of that rule. This diagnostic is not a new method-effect confirmation.

### Conservative finite-frame envelope

RR@1 has support {0,1}; RR@3 has support {0,1/3,1/2,1}. Query-level contrast endpoints lie in the full difference support. For RR@3 this is {-1,-2/3,-1/2,-1/3,-1/6,0,1/6,1/3,1/2,2/3,1}. Unobserved support values are retained.

At each support threshold in each stratum, we invert a one-sided hypergeometric tail for the finite-population count at or below that threshold. For the lower endpoint, upper CDF limits yield a lower mean limit; for the upper endpoint, lower CDF limits yield an upper mean limit. The discrete identity E[X] = x_min + sum_j (x_(j+1)-x_j) P(X > x_j) combines the counts into endpoint limits. Original frame weights combine the stratum means. Bonferroni controls both endpoint directions, eight strata and all support thresholds. At RR@3 this allocates 0.05/(2 × 8 × 10) to each one-sided tail. Under the stated fixed-rule sampling interpretation, the resulting 95% outer envelope has conservative finite-sample design coverage for both population endpoints.

This construction is intentionally conservative. Sixteen queries per stratum, full-support count limits and simultaneous tail control allow substantial mass at values absent from the sample. A wide envelope describes the resolution of this procedure; it does not establish a large underlying loss or gain.

### Approximate precision reference

We also report the Neyman variance estimate with finite-population correction:

```text
Vhat(endpoint) = sum_h (N_h/8530)^2 (1 - n_h/N_h) s_h^2/n_h
```

Welch–Satterthwaite degrees of freedom and one-sided t critical values at alpha/2 combine the lower and upper endpoint limits into an approximate 95% outer envelope. With 16 queries per stratum and sparse nonzero contrasts, this is a precision reference with approximate coverage. It does not replace the conservative finite-frame method or provide a finite-sample guarantee.

### Results for the declared verified-policy diagnostic

All rows use RIDGE_verified minus the original query at RR@3 with the original stratum weights. Each 95% envelope below covers the two endpoints within one reviewer/threshold combination under its stated assumptions. Threshold 2 is the primary relevance diagnostic; threshold 1 includes partially relevant code.

|Relevance threshold|Reviewer|Sample identification range|Conservative finite-frame 95% envelope|Stratified-t approximate 95% envelope|
|---:|---|---|---|---|
|2|A|[+0.005843, +0.018263]|[-0.400821, +0.417409]|[-0.008645, +0.033373]|
|2|B|[+0.001279, +0.016604]|[-0.401856, +0.420320]|[-0.009318, +0.033184]|
|1|A|[+0.003297, +0.005007]|[-0.395174, +0.400332]|[-0.006890, +0.014480]|
|1|B|[+0.001948, +0.010069]|[-0.406116, +0.413443]|[-0.014874, +0.025913]|

We additionally control the four reviewer-by-threshold combinations together by using alpha=0.05/4 for each conservative envelope. The four-combination family was declared for this post-hoc uncertainty report; it does not amend the original diagnostic protocol or the B/G comparison families.

|Relevance threshold|Reviewer|Conservative envelope with simultaneous 95% coverage across these four combinations|
|---:|---|---|
|2|A|[-0.450782, +0.466530]|
|2|B|[-0.451641, +0.469207]|
|1|A|[-0.445096, +0.450156]|
|1|B|[-0.455725, +0.462720]|

The sample identification ranges are positive in all four weighted comparisons. Every conservative and approximate sampling envelope spans zero. These observations support sensitivity of the sampled fixed-policy utility to relevance coverage; they do not establish positive utility in H8530, equivalence to zero, or a stable deployment benefit. Strict unweighted reviewer-B results, the changes relative to original qrels, and risk-stage sensitivities retain their original interpretation. No interval includes annotation error or generalizes beyond the fixed frame.

### Evidence and validation

The input is the unchanged 6,144 query-level records in results/rene_review_completion_20261021/query_bounds.jsonl. All 48 existing reviewer/threshold/cutoff/policy combinations are evaluated, without selecting by interval sign. Each row has its own 95% envelope; there is no claim of simultaneous 95% coverage across all 48 combinations. The four-combination family above is the separately reported simultaneous result.

Methods and code are in experiments/rene_relevance_sampling_uncertainty_20261021/. RESULTS.json, FINDINGS.md and NUMERICAL_VALIDATION.json are in results/rene_relevance_sampling_uncertainty_20261021/. Numerical replay separately checks the original weighted ranges, Fraction-based variances, hypergeometric inversions and aggregate envelopes for all combinations. Small-population enumeration checks the finite-frame coverage and variance contracts. Original labels, unknowns, disagreements, qrels and the accepted 15-table bundle are preserved.


## Paired Two-Weight Confirmation Report

The independent new query/connected-group sample contains 25,600 groups and a fixed corpus of 32,768 code snippets. Eligibility, sample size, models, complete generation/retrieval configuration and the six-comparison family were locally sealed before new generation and effect inspection. This was not external preregistration.

### All six prespecified comparisons

| Comparison | Estimate | Adjusted t interval | Adjusted group-bootstrap interval | Direction | Nonzero groups |
|---|---:|---|---|---|---:|
|local::ENTR_B|+0.002803|[+0.001565, +0.004041]|[+0.001576, +0.004024]|Positive|9,734|
|local::ENTR_G|+0.003241|[+0.002409, +0.004073]|[+0.002422, +0.004091]|Positive|11,173|
|t5base::ENTR_B|+0.004282|[+0.002941, +0.005624]|[+0.002946, +0.005626]|Positive|9,781|
|t5base::ENTR_G|+0.003520|[+0.002636, +0.004404]|[+0.002636, +0.004427]|Positive|11,485|
|base_minus_local::ENTR_B|+0.001479|[-0.000325, +0.003282]|[-0.000336, +0.003267]|Unresolved|10,982|
|base_minus_local::ENTR_G|+0.000279|[-0.000909, +0.001467]|[-0.000942, +0.001477]|Unresolved|12,416|

All four ENTR components meet the prespecified six-comparison correction and joint interval-direction rule. In this new StaQC sample, with fixed CodeBERT and the specified complete generation evaluation contract, the paired confirmation supports positive B/G directions for both T5 weights. This strengthens the second contribution's evidence across the tested weight conditions.

Paired weight differences subtract local from base for each query. A resolved difference would concern the complete weight conditions rather than isolate CQC training itself; unresolved differences do not imply equivalence. This experiment neither specifies nor tests an effect-equivalence margin.

### Comparisons and effect scale

B=S(1,mean)−S(3,mean), G=S(3,mean)−S(3,best3). S is the ENTR RR@100 advantage over exact random selection within the complete position pool for each weight. Positive values indicate a smaller relative advantage along the specified evaluation-contract steps. These quantities are neither automatic-execution gains, generation-call budgets nor unique causal attributions.

| Weights | S(1,mean) | S(3,mean) | S(3,best3) | NoRewrite MRR@100 |
|---|---:|---:|---:|---:|
|local|+0.005374|+0.002571|-0.000671|0.113417|
|t5base|+0.007713|+0.003431|-0.000089|0.113417|

PROB, ENTR−PROB and the complete k1/2/3 × mean/best-of-k results remain prespecified descriptive controls in COMPLETE_CURVES.csv and RESULTS.json. They do not add confirmatory rejection claims.

### Eligibility and planning

After excluding historical references, 84,618 single-target StaQC connected groups remained. Excluding the previous complete corpus, queries, historical generation and normalized exposure left 68,233 eligible components. Fixed hashes selected one item per component, query components and distractor corpus components. New and old complete-corpus components did not overlap.

Planning used only consumed matched caches from the two domains and the larger local-weight result. The target resolution was .005 RR with a 95% power target for each direction; at 25,600 groups, the planned lower bound for all four directions was 0.9652. This plan assumes transferable variance for a .005 effect; it does not guarantee resolution of a smaller true effect.

The planned adjusted half-width for each of the two weight differences was at most .005. Here .005 is a precision target, not an equivalence margin. There was no interim effect-based stopping, seed addition, model fitting or policy promotion.

Accepted single-code-answer titles and snippets define one fixed retrieval condition. Other snippets were not exhaustively judged irrelevant. Unknown base-model pretraining exposure, semantic clones and undocumented historical exposure are outside the local eligibility guarantee.

### Relationship to earlier evidence

The historical 512-query matched analyses in each domain retain their 24-comparison family. The earlier StaQC 13,312-group/16,384-code study retains its two-comparison family. The 25,600 groups form a separate new component sample and fixed corpus; its six-comparison family does not retrospectively change earlier conclusions. The results do not establish a finding across model families or all retrievers, or attribution to historical full training.

### Recorded validation

Validation included 14 evaluation-contract tests; replay of 360 historical-query position spans under both weights and their complete ENTR/PROB orderings; checks of all query/corpus provenance, group independence, prior exposure, candidate/action coverage and rankings; independent rational-number/CDF reconstruction of all B/G values; independent replay of six t intervals and 20,000 common-group bootstrap resamples; and actual retriever reload with full-corpus ranking replay for 32 queries and 32 code snippets.

The frozen experiment export contains DESIGN.json, ELIGIBILITY.json, PRECISION_PLAN.json, PREOPEN_AUDIT.json, NUMERICAL_AUDIT.json, MODEL_AUDIT.json and FINAL_ACCEPTANCE.json. Delivered results are in `results/experiment_exports/rene_weight_confirmation_20261021/` and `results/paired_weight_confirmation_results.json`.
