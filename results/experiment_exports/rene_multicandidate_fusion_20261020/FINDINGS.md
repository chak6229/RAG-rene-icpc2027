# Same-pool candidate fusion: results and interpretation

Date: 2026-10-20. The experiment passed independent rank, utility and interval replay and the final acceptance checks.

**Positive net gain over NoRewrite was not established for three-candidate fusion or for fusion of four lists including the original query. Fusion reduced some of the loss from forced Top1, with a clear improvement relative to forced Top1 in the tested BM25 setting.** This is a controlled comparison of executable candidate use. Automatic adoption in general and new UQ methods are outside this finding.

## Data and design

The analysis uses 1490 queries in 512 linked groups and 6267 documents from previously used CoSQA development D. CodeBERT and BM25 use the same three ENTR candidates and frozen Top100 lists from the local CQC T5. RRF uses constant 60 and equal weights, deduplicates identical input text, computes exact rational scores, breaks ties by ascending document ID and returns Top100. Actions were built without labels and sealed before the evaluation labels were opened.

The study added no training, generation, retrieval, CE, risk gate, or searches over weights and thresholds. SSQR entropy still selects the three candidates. Fusion uses these candidates without an additional UQ head.

## Primary results

All ΔMRR values use NoRewrite as reference. Pointwise intervals are 95% linked-group bootstrap intervals. The four primary effects also have Bonferroni-corrected 98.75% intervals.

|Policy|CodeBERT ΔMRR|BM25 ΔMRR|CodeBERT Useful/harmful|BM25 Useful/harmful|
|---|---:|---:|---:|---:|
|NoRewrite|0|0|0/0|0/0|
|ENTR Top1|−.01365851|−.02035096|292/388|45/555|
|RRF3: Three candidates|−.01009966|−.00357041|303/352|92/411|
|RRF4: Original query+ Three Candidates|−.00630794|−.00202436|287/284|81/349|
|ENTR best3 Labels support opportunities|+.06037845|+.00977629|528/125|128/319|

|Main comparisons|Points 95%|Four family correction blocks.|Decision|
|---|---|---|---|
|CodeBERT RRF3−original query|[−.01971797,−.00039499]|[−.02279004,+.00227612]|Corrected interval spans 0; positive gain not established|
|CodeBERT RRF4−original query|[−.01356337,+.00068063]|[−.01556317,+.00252055]|Interval spans 0; positive gain not established|
|BM25 RRF3− Original query|[−.00605671,−.00109454]|[−.00681099,−.00040266]|Correct negative direction|
|BM25 RRF4−original query|[−.00375980,−.00041231]|[−.00419819,+.00002045]|Corrected interval spans 0; positive gain not established|

After correction across the four primary effects, only BM25 RRF3 remains negative. The other three effects are unresolved. No equivalence margin was specified for these contrasts.

## Comparisons with forced Top1 and the contribution of the original query

RRF4 minus ENTR Top1 is +.00735057 for CodeBERT (95% interval [−.00181023,+.01681903]) and +.01832660 for BM25 ([+.01338007,+.02354481]). BM25 RRF3 minus Top1 is also positive, +.01678055 ([+.01193765,+.02195994]). These paired contrasts are descriptive secondary results.

For RRF4 minus RRF3, the point estimates are +.00379172 for CodeBERT and +.00154605 for BM25. Their intervals are [−.00116375,+.00886707] and [−.00023031,+.00338637]. The incremental gain from including the original query is unresolved. The direct paired difference between retrievers also spans 0.

The common denominator is 1490 queries:

- CodeBERT harm frequency: Top1 26.04% → RRF3 23.62% → RRF4 19.06%.
- BM25 harm frequency: Top1 37.25% → RRF3 27.58% → RRF4 23.42%.
- From RRF3 to RRF4, CodeBERT positive gain per query changes from .03079309 to .02195648 and loss per query from .04089274 to .02826442. BM25 positive gain changes from .00358844 to .00215269 and loss from .00715884 to .00417705.

Including the original query reduces both positive gain and loss in the measured pool. Loss magnitude remains above positive gain, so the net ΔMRR estimates remain negative. Harm frequency and net utility describe separate quantities.

Best3 is the label-assisted opportunity from executing one candidate. Fusion constructs a different ranking. RRF3/RRF4 exceed the individual RRs of all three candidates on 54/59 CodeBERT queries and 174/223 BM25 queries. Best3 is therefore not used as a retention denominator for fusion.

## Workload accounting

|Policy|Only search input average for each query|Total single input|Location Generations|
|---|---:|---:|---:|
|NoRewrite|1.0000|1490|0|
|ENTR Top1|1.0000|1490|9915|
|RRF3|2.9799|4440|9915|
|RRF4|3.9799|5930|9915|

Standalone Top1 requires only the rewritten input. Adding the common original-query retrieval gives 2980 inputs. The rewrite policies share the existing full-position generation pool. The four-list cache contains 5930 unique inputs for the controlled comparison. The rerun made 0 new neural calls. The table reports input counts per policy, which do not measure online time, monetary cost or token count.

## Role in the replication

The study evaluates how much candidate opportunity from SSQR-style rewriting is realized by automatic execution and what evidence helps determine when to adopt a rewrite. Fusion supplies a controlled executable comparison within the external replication.

|Evidentiary component|What can we support now?|Can't replace anything.|
|---|---|---|
|Original frozen CoSQA confirmation|The specified CodeBERT ENTR−RAND20 best3 contrast used 7718 queries in 1963 groups. The difference was −.00083652 and both 95% intervals fell within the prespecified ±.005 margin. This bounds the advantage for the designated reconstruction and setting.|The contrast does not recover the authors' weights, isolate a causal effect of query source, or establish general ineffectiveness of UQ.|
|Step 1 A Direct comparison with pool|CoSQA/CodeBERT ENTR Relative random stand-alone +.02793583; Direct deviation from best3 relative effects +.02556783, corrected to be positive; Entropy signals have measurable candidate ranking values. PROB Contrast and precise random 3 Sensitivity retention.|Top1/best3 The gap itself is already in the original language; The difference in this round involves both budget and max aggregation, not pure UQ mechanism causality.|
|Step B, fixed-action adoption|For fixed ENTR Top1 and four equally supervised heads, incremental adoption gain from generation confidence was not established. With 296 adoptions in CodeBERT, the full head improves on the common features by only +.00002276 and increases harm from 53 to 81.|This leaves broader UQ utility unresolved, does not establish a stage interaction, and retains the historical positive result on 1536 queries with four seeds in its own scope.|
|Same-pool RRF3/4|The executable multiple-candidate control uses no additional UQ or supervised head. Fusion can mitigate some losses from forced Top1. Positive gain over NoRewrite was not established.|The controlled fusion baseline expands the evaluated use of candidates. Other automatic use policies remain outside the comparison.|

The evidence supports the specified replication and controlled extension. SSQR Table 5 already reports Top1 versus Top3, and QueryRoute and query performance prediction studies address the general gap between candidate opportunity and selection. This experiment's contribution concerns the direct same-pool contrast under the stated selected counts and evaluation contracts. See `PRIOR_WORK_COMPARISON.md` for related work.

The fusion comparison had no risk gate. The negative point estimates therefore cannot be explained by risk fallback in this comparison. `risk_certified=false`.

## Historical follow-up and delivery decisions

1. The planned report would align the frozen confirmation, analyses A and B, and the fusion comparison with the specific SSQR contract and retrievers. The general gap between candidate opportunity and adoption had already been studied.
2. The two t5-base summaries existed, while the rankings and per-query RR had yet to be retained for independent replay. Use as strong breadth evidence depended on filling and auditing that gap. The later two-weight reports document its status.
3. A query-source explanation would require a matched target and corpus with user queries and docstrings, and would still not isolate phrasing. Confirmation of a new policy would require eligible new groups, precision planning and a single frozen policy. Previously used CoSQA D/H remain in their recorded roles.
4. Delivery would include the verified clean-directory replay, experiment code, actions, statistics, provenance and dependency notices. An efficiency claim about latency would require timing measurements beyond the recorded input counts.

See the historical `paper/rene2027/implementation_fidelity.md` for the unavailable author T5 checkpoint and the limits of local CQC training, decoding and original production-aggregation recovery. This run did not reopen H or retune the policies.

## Independent verification

- All 8 evaluation-contract tests passed.
- An independent audit without production ranking, deduplication or metric modules recomputed all 5960 fused Top100 lists with `Fraction`. All rankings matched.
- Independent checks covered 14900 RRs, 29800 per-query effects, 23 scalar contrasts and 146 interval fields. RR and effect discrepancies were 0. The maximum interval discrepancy was 9.71e−17.
- Hashes matched for 37 frozen inputs, source and protected files, and 2 sealed action files. Earlier analyses A/B, the H confirmation and the five recorded pointers were retained.
- `FINAL_ACCEPTANCE.json` records PASS. The role was `consumed_development_post_hoc_extension`.

Tables: `READOUT.md` and `RESULTS.json`. Protocol: `PROTOCOL.json`. Independent evidence: `AUDIT.json` and `FINAL_ACCEPTANCE.json`. Replay: `experiments/rene_multicandidate_fusion_20261020/README.md`.
