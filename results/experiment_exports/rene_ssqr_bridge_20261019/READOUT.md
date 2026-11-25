# SSQR Recover equity patches: original data bridge and sensitivity

Status: Actual generation, double searcher ranking, independent RR and real T5 check completed. (a) The evidentiary role as complementary diagnosis after confirmation; CoSQA Develop only D and do not change existing H confirmation.

## Let's see.

The source data bridge also does not create a positive advantage that can be clearly attributed to the negative entropy; The confirmation of CoSQA cannot be attributed to a true search migration. The source-side differences and the non-recoverable training/weight factors remain to be explained separately, and the list of discrepancies is not included in this round as an established cause of failure.

The main line continues to be the external recovery and automatic adoption analysis of SSQR candidate returns. This paper can report on the results of the reconciliation, limited realization of sensitivity and effectiveness of the freeze phase; It's not an additional diagnosis that would pretend to be a new independent confirmation, nor would it promote the full range of UQ valid or ineffective.

## Same realized source data bridge

Former AdvTest fixed Hash to 2048/594 repositories, searching for all 19210 original test functions in the author's split order. Query and code-by-code URL Consistent against author input. CoSQA Duplicate D1490 / 512,626 7 Function. The two sides are the same T5, the same CodeBERT, 1...n positions, greedy/total decoder20, k = 3 and RAND20.

The following table best3 does not contain NoRewrite; It's relative to each other, absolutely MRR without direct comparison. The compartment is 10000 associate group bootstrap95%.
This table follows the definition of frozen roll-by query with maximum RR. Repetition of the open assessor URL processing with two other polymers; see also `../rene_ssqr_aggregation_audit_20261019/READOUT.md `; there is no claim that the production polymer path of the original author has been restored.

| retriever | Data and queries | NoRewrite MRR | ENTR best3 ΔMRR | ENTR−RAND20 | 95% Range |
|---|---|---:|---:|---:|---|
| CodeBERT | AdvTest 2048 | 0.30874417 | +0.02399075 | -0.00435302 | [-0.00822558, -0.00033732] |
| CodeBERT | CoSQA D1490 | 0.51890040 | +0.06037845 | +0.00236800 | [-0.00354366, +0.00796936] |
| BM25 | AdvTest 2048 | 0.21275343 | +0.00415845 | -0.00133742 | [-0.00331732, +0.00061543] |
| BM25 | CoSQA D1490 | 0.22412467 | +0.00977629 | -0.00194770 | [-0.00494417, +0.00106370] |

Direct estimate source-CoSQA ENTR−RAND effect differential, cross-domain shared repository synthesizes the same reset group and maintains agreement difference:

| retriever | Source - CoSQA Effect differential | Group t95% | Group bootstrap95% | Shared/Joint Group |
|---|---:|---|---|---:|
| CodeBERT | -0.00672102 | [-0.01372981, +0.00028777] | [-0.01364477, +0.00045645] | 34/1072 |
| BM25 | +0.00061028 | [-0.00294121, +0.00416177] | [-0.00288441, +0.00418412] | 34/1072 |

Original SSQR Table 4 ENTR− RAND =.2221 −.1886 =+.0335; Local weights, training details and running times are different. Original values cannot be used as local contrast arms. The source data bridge still changes the text, length of queries, labels and difficulties at the same time, so no pure search sources are identified as cause or effect.

## Six limited settings sensitive

The following two domains are a subset of 512 queries that pre-established one article in each group, all profile using identical inquiries; This is not the same as the above table query weighted main bridge. total decoder Length consists of start, 10 corresponding to 9 new tokens and is visible m10 operating rather than claiming the restoration of historical counts. Sampling per position 1 bar, fixed location localRNG, common location reuse; Not multiple seed integration. The following are all below full-label entropy ENTR−RAND20.

| Settings | AdvTest CodeBERT | CoSQA CodeBERT | AdvTest BM25 | CoSQA BM25 |
|---|---:|---:|---:|---:|
| greedy20_source | -0.00442004 | +0.00434089 | -0.00198183 | -0.00059215 |
| greedy20_all | -0.01226867 | -0.00238535 | -0.00276119 | +0.00016080 |
| greedy10_source | -0.00434817 | +0.00452735 | -0.00211089 | -0.00059357 |
| greedy10_all | -0.01218871 | -0.00225219 | -0.00280263 | +0.00015811 |
| sample10_source | -0.00870789 | -0.00046114 | -0.00098711 | +0.00039942 |
| sample10_all | -0.00447282 | -0.00104229 | -0.00095484 | +0.00266645 |

![](implementation_sensitivity.png)

A limited pair of modified CodeBERT results (shared with Query, Sampling); Modified by the effect of ENTR−RAND:

| Changes | Data | Effect Change | Doublebootstrap 95% |
|---|---|---:|---|
| position_greedy20 | source | -0.00784863 | [-0.01722201, +0.00102902] |
| length_source | source | +0.00007186 | [-0.00024619, +0.00042480] |
| sampling_source | source | -0.00435972 | [-0.01705003, +0.00878252] |
| sampling_all | source | +0.00771589 | [-0.00417838, +0.01957007] |
| paper_oriented_joint | source | -0.00005279 | [-0.01291643, +0.01292661] |
| position_greedy20 | cosqa | -0.00672623 | [-0.01451093, +0.00027443] |
| length_source | cosqa | +0.00018646 | [-0.00001035, +0.00047182] |
| sampling_source | cosqa | -0.00498849 | [-0.01730013, +0.00708399] |
| sampling_all | cosqa | +0.00120991 | [-0.01199839, +0.01438868] |
| paper_oriented_joint | cosqa | -0.00538317 | [-0.01859540, +0.00753700] |

Full PROB, lexical-only and between settings `metrics/RESULTS.json `. All profile are retained; no confirmation configurations or upgrades of official modules based on estimates of these points have been replaced.

## The Triple Convergence itself changes the object of the estimate

The same list of candidates and rankings, first on query take the best RR and then average it with the whole sequence first calculated MRR before taking the best two questions. The latter indicator is not equivalent to a policy that can be deployed without looking at labels. Public reader End-line behaviour as a third-calibre standalone.

| Data/researcher | PV query best3 E−R | Three rank-wise MRR Best taken E−R | Endline E−R |
|---|---:|---:|---:|
| source/CodeBERT | -0.00435302 | +0.00156252 | +0.00172419 |
| source/BM25 | -0.00133742 | +0.00110986 | +0.00171944 |
| cosqa/CodeBERT | +0.00236800 | +0.02294983 | -0.00046317 |
| cosqa/BM25 | -0.00194770 | +0.00369643 | -0.00682315 |

For CoSQA D/CodeBERT, the point estimate under the best rank-wise contract is +0.02294983. The conservative 95% interval propagated from the simultaneous bounds for 66 means is [-0.09028707, +0.13570056]. The direction remains unresolved, and the per-query best3 equivalence decision applies to a separate contract. The aggregation audit reports all three definitions and intervals.

Directly execute the author's three-line summary of the same URL as evaluator, with the end result: 0,PV query best3 1; This confirms the public reader coverage behaviour and does not restore the aggregate path of the author ' s production forms for the year. The paper will be clearly written as external reconstruction under a predefined indicator of ex-query opportunities, without considering the full calibre of the original table 4 to have been accurately restored.

## How to retain existing independent confirmations

The CoSQA H effects were opened once at 21:37. For the 7718-query, 1963-group primary ENTR−RAND contrast, the difference was −.00083652. Both the 95% group t interval [−.00320993,+.00153688] and the 95% bootstrap interval [−.00325479,+.00154146] fell within the prespecified ±.005 margin. The source bridge and D sensitivities leave this designated comparison intact. The H intervals do not apply to other profiles.

±.005 Ex ante freeze; As a measure, it is about 14.9% of the original table 4 negative entropy-random difference.0335. .005 MRR It's equivalent to... 1% the average change caused by rank2 rising to rank1. This is a readable indicator measure and is not evidence of user satisfaction studies or minimal significant differences for all applications.

Three candidates for all H8530 best3 Δ CodeBERT+.05655770, BM25+.01358236; These candidate opportunities and benefits. CE The risk phase results support the research backbone of the original document and cannot be equated with a new UQ algorithm that has stabilized to increase automatic utility.

## Discrepancies between completed recovery bases and reservations

1. Original data query, complete search pool, source line/URL and qrels identity check; The original paper search pool was not replaced by 4096 pools or 45244 supporting files.
2. Two data bridges with the same weight, a direct gap between areas, six profile and two entropy masks; The melting position does not change the common candidate or random stream of sampling.
3. The paper achieves the difference sheet and methodological text, distinguishing between revised query / entropy/CodeBERT objectives and remaining different CQC training; Add a new public evaluator repeat URL cover and three-calibre check.
4. Independent RR, real model check and portable cache reset; confirmed with five official points.

Original author weights are not available, local T5 source of training/cover/optimizer/long is different from open trainer, PROB original realized undisclosed, historical m count/production tie / three candidate production aggregate paths cannot be restored. These have been written in Design and artifact; Numerical audits cannot be extrapolated to exactly the same as author models.

## Receiving and inspection and delivery

- Protocol SHA256:`7d3d4c0728aa413a21dfc748d4a01dc4f29d5d59a0b4a44591eb48793dc3d95e `; 13 Contract tests passed.
- Independent Fraction RR check: 271096 Item, Max. 2.22e-16.
- Real T5 sampling corresponds to FP64 entropy: 6 candidate, maximum entropy 6.13e-07; sample generation token.
- Portable artifact Additional acceptances `../rene_ssqr_bridge_release_20261019/RELEASE_ACCEPTANCE.json `; independent scripts do not import workspace and cache recurrence does not pretend to be a complete retrain.
- Code:`experiments/rene_ssqr_bridge_20261019/`; reports and master data: this catalogue; papers:`paper/rene2027/implementation_fidelity.md `and `replication_results.md `.
- Five formal points, old weights and confirmed results are retained, risk_certified = false, NoRewrite status maintained.
