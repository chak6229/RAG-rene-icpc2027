# Supplementary information

Position Selection Advantage and Executable Utility: An External Replication of SSQR for Code Search

## Ethics and informed consent

Both reviewers gave informed consent before participating, after learning the study purpose and review tasks, and agreed to share de-identified annotations and rationales. The paper identifies reviewers as A/B and reports aggregate statistics; the reproduction materials retain anonymous item-level annotations and rationales without reviewer identities. No formal ethics review or exemption determination was sought for this study.

## Table S1. Specified CoSQA replication comparison

| Comparison | Queries / groups | ΔMRR@100 | Bootstrap 95% CI | Margin | Decision |
| --- | --- | --- | --- | --- | --- |
| CodeBERT ENTR-RAND20 best3 | 7,718 / 1,963 | -0.00083652 | [-0.00325479, +0.00154146] | ±0.005 | Equivalent |

One frozen confirmatory comparison on capped CoSQA H. The displayed interval is the group-bootstrap 95% interval. Practical equivalence uses the frozen joint rule: both cluster-t and bootstrap 95% intervals strictly inside ±0.005, with the required changed-group count. The rule applies to this ENTR−RAND20 per-query best3 comparison.

## Table S2. Endpoint contrasts under the same candidate pool

| Domain / retriever | Metric | Estimate | 95% bootstrap CI | Adjusted 95% CI |
| --- | --- | --- | --- | --- |
| AdvTest / CB | Single advantage | +0.00083866 | [-0.00431406, +0.00589894] | — |
| AdvTest / CB | Best3 advantage | -0.00435302 | [-0.00824417, -0.00039766] | — |
| AdvTest / CB | Direct endpoint difference | +0.00519168 | [-0.00016682, +0.01020699] | [-0.00154743, +0.01167464] |
| AdvTest / BM25 | Single advantage | -0.00216420 | [-0.00468566, +0.00027304] | — |
| AdvTest / BM25 | Best3 advantage | -0.00133742 | [-0.00332788, +0.00063524] | — |
| AdvTest / BM25 | Direct endpoint difference | -0.00082678 | [-0.00371686, +0.00190719] | [-0.00447948, +0.00263736] |
| CoSQA D / CB | Single advantage | +0.02793583 | [+0.01922302, +0.03670910] | — |
| CoSQA D / CB | Best3 advantage | +0.00236800 | [-0.00348469, +0.00831373] | — |
| CoSQA D / CB | Direct endpoint difference | +0.02556783 | [+0.01660135, +0.03437975] | [+0.01416754, +0.03661756] |
| CoSQA D / BM25 | Single advantage | -0.00169972 | [-0.00589223, +0.00242177] | — |
| CoSQA D / BM25 | Best3 advantage | -0.00194770 | [-0.00497482, +0.00096444] | — |
| CoSQA D / BM25 | Direct endpoint difference | +0.00024798 | [-0.00443244, +0.00478773] | [-0.00586250, +0.00587435] |

Post hoc paired diagnostic. Single-candidate advantage is ENTR Top1 minus uniform single selection; best3 advantage uses ENTR best3 minus RAND20. Their direct difference changes both count and aggregation. Adjusted intervals apply to the four direct endpoint contrasts. This quantity is distinct from the exact-random B+G decomposition in S8.

## Table S3. Fixed20 adoption with matched prediction heads

| Retriever | Head | ΔMRR@100 | 95% bootstrap CI | Adopted | Harmful |
| --- | --- | --- | --- | --- | --- |
| CodeBERT | M0 | +0.00134060 | [-0.00188537, +0.00467032] | 296 | 53 |
| CodeBERT | MT | -0.00066742 | [-0.00366518, +0.00250298] | 296 | 55 |
| CodeBERT | MR | +0.00107971 | [-0.00249419, +0.00470514] | 296 | 75 |
| CodeBERT | MRT | +0.00136336 | [-0.00207056, +0.00499405] | 296 | 81 |
| BM25 | M0 | -0.00104922 | [-0.00163569, -0.00050988] | 274 | 94 |
| BM25 | MT | -0.00117883 | [-0.00182518, -0.00056966] | 274 | 93 |
| BM25 | MR | -0.00190590 | [-0.00314008, -0.00094161] | 274 | 92 |
| BM25 | MRT | -0.00225271 | [-0.00354231, -0.00122280] | 274 | 96 |

CoSQA D: 1,490 queries, 512 groups. Fixed20 adopts the top 20% of eligible actions in each held-out fold, rounded down, including negative predictions. All heads share actions and folds. Intervals are 95% group-bootstrap intervals conditional on fitted out-of-fold (OOF) heads. Conditional feature increments retain the separate eight-comparison family in the source protocol; head-mean intervals shown here are not adjusted increment tests.

## Table S3 continued — zero-threshold adoption

| Retriever / head | ΔMRR@100 | Cluster-t 95% CI | Bootstrap 95% CI | Adopted / harmful |
| --- | --- | --- | --- | --- |
| CodeBERT / M0 | -0.00031563 | [-0.00464508, +0.00401381] | [-0.00468573, +0.00398034] | 515 / 101 |
| CodeBERT / MT | -0.00147678 | [-0.00582330, +0.00286974] | [-0.00584448, +0.00286956] | 526 / 111 |
| CodeBERT / MR | +0.00280718 | [-0.00211878, +0.00773315] | [-0.00204468, +0.00785279] | 673 / 158 |
| CodeBERT / MRT | +0.00265119 | [-0.00239337, +0.00769576] | [-0.00237067, +0.00779101] | 665 / 157 |
| BM25 / M0 | -0.00082091 | [-0.00137612, -0.00026570] | [-0.00139271, -0.00029974] | 241 / 77 |
| BM25 / MT | -0.00103303 | [-0.00164871, -0.00041736] | [-0.00167126, -0.00044556] | 253 / 84 |
| BM25 / MR | -0.00295111 | [-0.00444120, -0.00146101] | [-0.00459446, -0.00164039] | 343 / 123 |
| BM25 / MRT | -0.00302710 | [-0.00453735, -0.00151684] | [-0.00467936, -0.00168200] | 344 / 122 |

These descriptive head-versus-original-query results use the zero-threshold policy: adopt an eligible action only when its fitted OOF predicted gain is positive. The original S3 table reports Fixed20. All head results use every development query as the evaluation denominator. Intervals are conditional on the fitted OOF scores; they do not refit the models during resampling.

Source: `results/experiment_exports/rene_core_step_b_20261020/RESULTS.json`

## Table S3 continued — conditional feature increments

| Retriever / policy / contrast | ΔMRR@100 | Adjusted cluster-t CI | Adjusted bootstrap CI | Nonzero groups |
| --- | --- | --- | --- | --- |
| CodeBERT / Fixed20 MRT-MT | +0.00203078 | [-0.00211904, +0.00618060] | [-0.00193256, +0.00631901] | 100 |
| CodeBERT / Fixed20 MRT-MR | +0.00028365 | [-0.00225016, +0.00281746] | [-0.00214964, +0.00299282] | 34 |
| CodeBERT / Zero threshold MRT-MT | +0.00412798 | [-0.00189823, +0.01015418] | [-0.00200575, +0.01025486] | 146 |
| CodeBERT / Zero threshold MRT-MR | -0.00015599 | [-0.00321687, +0.00290489] | [-0.00315262, +0.00291544] | 42 |
| BM25 / Fixed20 MRT-MT | -0.00107388 | [-0.00261526, +0.00046750] | [-0.00291996, +0.00018168] | 72 |
| BM25 / Fixed20 MRT-MR | -0.00034681 | [-0.00085060, +0.00015698] | [-0.00090410, +0.00008002] | 27 |
| BM25 / Zero threshold MRT-MT | -0.00199406 | [-0.00404324, +0.00005512] | [-0.00443257, -0.00034870] | 75 |
| BM25 / Zero threshold MRT-MR | -0.00007599 | [-0.00057188, +0.00041990] | [-0.00055360, +0.00047407] | 36 |

The original eight-comparison family comprises MRT−MT and MRT−MR, two policies and two retrievers. Both interval columns apply its Bonferroni adjustment. MRT−MT adds retrieval-change evidence conditional on generation uncertainty; MRT−MR adds generation uncertainty conditional on retrieval-change evidence.
For BM25 with the zero-threshold policy, MRT−MT has an adjusted bootstrap interval below zero, while the adjusted cluster-t interval includes zero. The joint decision therefore remains unresolved. These details expose existing results; no comparison family or outcome has been changed.

Source: `results/experiment_exports/rene_core_step_b_20261020/RESULTS.json`

## Table S4. Executable fusion and label-assisted candidate opportunity

| Retriever / policy | ΔMRR@100 | 95% bootstrap CI | Adjusted CI | Harmful | Executable |
| --- | --- | --- | --- | --- | --- |
| CodeBERT ENTRTop1 | -0.01365851 | [-0.02269701, -0.00478947] | — | 388 | Yes |
| CodeBERT RRF3 | -0.01009966 | [-0.01971797, -0.00039499] | [-0.02279004, +0.00227612] | 352 | Yes |
| CodeBERT RRF4 | -0.00630794 | [-0.01356337, +0.00068063] | [-0.01556317, +0.00252055] | 284 | Yes |
| CodeBERT ENTRBest3 | +0.06037845 | [+0.05146242, +0.06938687] | — | 125 | Label-assisted |
| BM25 ENTRTop1 | -0.02035096 | [-0.02571306, -0.01522141] | — | 555 | Yes |
| BM25 RRF3 | -0.00357041 | [-0.00605671, -0.00109454] | [-0.00681099, -0.00040266] | 411 | Yes |
| BM25 RRF4 | -0.00202436 | [-0.00375980, -0.00041231] | [-0.00419819, +0.00002045] | 349 | Yes |
| BM25 ENTRBest3 | +0.00977629 | [+0.00518148, +0.01468792] | — | 319 | Label-assisted |

CoSQA D: 1,490 queries, 512 groups. Changes are relative to the original query. RRF3/RRF4 comparisons retain their four-comparison family; other displayed policies are descriptive. ENTR best3 needs relevance labels and is not executable. Fusion creates a new ranking, so candidate best3 is not its upper bound.

## Table S4 continued — fusion versus the original query

| Retriever / contrast | ΔMRR@100 | Adjusted cluster-t CI | Adjusted bootstrap CI | Nonzero groups |
| --- | --- | --- | --- | --- |
| CodeBERT RRF3-NoRewrite | -0.01009966 | [-0.02262627, +0.00242696] | [-0.02279004, +0.00227612] | 322 |
| CodeBERT RRF4-NoRewrite | -0.00630794 | [-0.01539844, +0.00278257] | [-0.01556317, +0.00252055] | 297 |
| BM25 RRF3-NoRewrite | -0.00357041 | [-0.00675369, -0.00038712] | [-0.00681099, -0.00040266] | 269 |
| BM25 RRF4-NoRewrite | -0.00202436 | [-0.00414045, +0.00009174] | [-0.00419819, +0.00002045] | 242 |

These four RRF3/RRF4-versus-original-query comparisons retain the original four-comparison Bonferroni family. Both adjusted interval methods support a negative direction only for BM25 RRF3. The other comparisons remain unresolved; positive net utility is not established.

Source: `results/experiment_exports/rene_multicandidate_fusion_20261020/RESULTS.json`

## Table S4 continued — descriptive paired fusion comparisons

| Retriever / contrast | ΔMRR@100 | Cluster-t 95% CI | Bootstrap 95% CI | Nonzero groups |
| --- | --- | --- | --- | --- |
| CodeBERT RRF4-RRF3 | +0.00379172 | [-0.00125756, +0.00884100] | [-0.00116375, +0.00886707] | 232 |
| CodeBERT RRF3-ENTRTop1 | +0.00355885 | [-0.00783924, +0.01495694] | [-0.00776614, +0.01479171] | 341 |
| CodeBERT RRF4-ENTRTop1 | +0.00735057 | [-0.00220585, +0.01690699] | [-0.00181023, +0.01681903] | 339 |
| BM25 RRF4-RRF3 | +0.00154605 | [-0.00025732, +0.00334942] | [-0.00023031, +0.00338637] | 202 |
| BM25 RRF3-ENTRTop1 | +0.01678055 | [+0.01181982, +0.02174128] | [+0.01193765, +0.02195994] | 295 |
| BM25 RRF4-ENTRTop1 | +0.01832660 | [+0.01327651, +0.02337670] | [+0.01338007, +0.02354481] | 296 |

These descriptive paired comparisons retain ordinary 95% intervals. RRF4−RRF3 measures the addition of the original-query list; RRF3/RRF4−ENTRTop1 compares fusion with direct Top1 adoption. They are not additional members of the four-comparison family above. Comparison with a negative Top1 policy does not establish a positive difference from the original query.

Source: `results/experiment_exports/rene_multicandidate_fusion_20261020/RESULTS.json`

## Table S5. Frozen full-H execution stages

| Retriever / policy | ΔMRR@100 | 95% bootstrap CI | Adopted / harmful | Positive retention | CE pairs |
| --- | --- | --- | --- | --- | --- |
| CodeBERT ENTR_top1 | -0.02110467 | [-0.02513943, -0.01704688] | 8,530 / 2,328 | 0.42518589 | 0 |
| CodeBERT HGB_gain_control | -0.00287610 | [-0.00566332, -0.00005520] | 5,305 / 1,277 | 0.27017404 | 0 |
| CodeBERT RIDGE_gain | +0.00007864 | [-0.00224526, +0.00241015] | 6,034 / 1,317 | 0.23043068 | 0 |
| CodeBERT RIDGE_verified | +0.00084705 | [-0.00158595, +0.00326493] | 5,890 / 1,280 | 0.24348541 | 17,056 |
| CodeBERT RIDGE_risk | -0.00017777 | [-0.00055106, +0.00019457] | 1,006 / 60 | 0.00769091 | 17,056 |
| BM25 ENTR_top1 | -0.01443510 | [-0.01692009, -0.01204855] | 7,955 / 2,929 | 0.39104616 | 0 |
| BM25 HGB_gain_control | -0.00271496 | [-0.00392964, -0.00154317] | 2,615 / 853 | 0.11797753 | 0 |
| BM25 RIDGE_gain | -0.00207943 | [-0.00277119, -0.00140306] | 3,559 / 1,072 | 0.05630485 | 0 |
| BM25 RIDGE_verified | -0.00171681 | [-0.00228134, -0.00115385] | 2,926 / 906 | 0.04425327 | 17,056 |
| BM25 RIDGE_risk | +0.00000000 | [+0.00000000, +0.00000000] | 0 / 0 | 0.00000000 | 17,056 |

Full H: 8,530 queries, 1,963 groups. These predefined descriptive estimates are separate from the capped-H replication test. Retention is a proportion of nonnegative candidate gain with the original query available. CE pairs count incurred cross-encoder work, including work preceding fallback; they do not measure latency. RIDGE_verified also includes queries that were not actually verified and retain gain-stage predictions.

## Table S6. Top1 and best3 under two T5 weights

| Domain / weights | Metric | Estimate | 95% bootstrap CI | Adjusted CI |
| --- | --- | --- | --- | --- |
| AdvTest / local | Top1 change | -0.03025188 | [-0.04521606, -0.01620265] | [-0.05170186, -0.01129044] |
| AdvTest / local | Best3 change | +0.02998920 | [+0.01828534, +0.04234607] | [+0.01375933, +0.04728210] |
| AdvTest / local | ENTR−uniform | -0.00582032 | [-0.01697531, +0.00548747] | — |
| AdvTest / t5base | Top1 change | -0.01336644 | [-0.02433999, -0.00262637] | [-0.02934959, +0.00089111] |
| AdvTest / t5base | Best3 change | +0.03852459 | [+0.02586949, +0.05180257] | [+0.02087319, +0.05713880] |
| AdvTest / t5base | ENTR−uniform | +0.00912318 | [-0.00030040, +0.01862476] | — |
| CoSQA D / local | Top1 change | -0.02019729 | [-0.03674618, -0.00434649] | [-0.04341906, +0.00123292] |
| CoSQA D / local | Best3 change | +0.06130778 | [+0.04602760, +0.07699339] | [+0.04002769, +0.08301839] |
| CoSQA D / local | ENTR−uniform | +0.02106576 | [+0.00781139, +0.03400633] | — |
| CoSQA D / t5base | Top1 change | -0.04270304 | [-0.06296258, -0.02299205] | [-0.07135004, -0.01609019] |
| CoSQA D / t5base | Best3 change | +0.05669980 | [+0.03920094, +0.07485249] | [+0.03189047, +0.08242730] |
| CoSQA D / t5base | ENTR−uniform | +0.03475321 | [+0.01882089, +0.05044681] | — |

Each domain uses the same 512 matched historical queries/groups. Each weight retains its own complete candidate pool. Top1 and best3 changes use the original query as reference; ENTR−uniform uses a single-candidate random reference. Eight Top1/best3 effects form the original adjusted family. These used-sample sensitivity analyses are separate from the new 25,600-group confirmation in main Table II.

## Table S7. Earlier uncertainty comparisons with their original targets

| Historical target | Comparison | Estimate | 97.5% CI |
| --- | --- | --- | --- |
| 1536 downstream repositories; full pipeline | uq increment | +0.00395239 | [+0.00133043, +0.00698473] |
| 1024 downstream repositories; fixed upstream action | full minus common | -0.00007971 | [-0.00361242, +0.00345300] |
| 1024 downstream repositories; fixed upstream action | full minus support | +0.00134653 | [-0.00121464, +0.00404959] |

UQ denotes the historical uncertainty comparison. Intervals retain the source 97.5% level. The 1,536-repository full-pipeline target and the 1,024-repository fixed-upstream-action target are different. The frozen full-minus-common equivalence result must not be substituted for the unresolved full-minus-support increment or for full-H code-search results.

## Table S8. Candidate-count and aggregation decomposition

| Condition / component | Estimate | Cluster-t CI | Bootstrap CI | Direction |
| --- | --- | --- | --- | --- |
| AdvTest / CB ENTR B | -0.00119189 | [-0.00771701, +0.00533323] | [-0.00793894, +0.00546475] | Unresolved |
| AdvTest / CB ENTR G | +0.00638704 | [+0.00192951, +0.01084457] | [+0.00195385, +0.01072038] | Positive |
| AdvTest / CB PROB B | +0.00060037 | [-0.00608963, +0.00729036] | [-0.00568646, +0.00742975] | Unresolved |
| AdvTest / CB PROB G | +0.00611574 | [+0.00143046, +0.01080102] | [+0.00137037, +0.01093849] | Positive |
| AdvTest / CB ENTR−PROB B | -0.00179226 | [-0.00813981, +0.00455530] | [-0.00805951, +0.00403014] | Unresolved |
| AdvTest / CB ENTR−PROB G | +0.00027130 | [-0.00325425, +0.00379685] | [-0.00315565, +0.00374282] | Unresolved |
| AdvTest / BM25 ENTR B | -0.00184620 | [-0.00516147, +0.00146906] | [-0.00532895, +0.00136783] | Unresolved |
| AdvTest / BM25 ENTR G | +0.00103996 | [-0.00141628, +0.00349619] | [-0.00146605, +0.00352039] | Unresolved |
| AdvTest / BM25 PROB B | -0.00283219 | [-0.00645869, +0.00079431] | [-0.00645139, +0.00082012] | Unresolved |
| AdvTest / BM25 PROB G | +0.00085974 | [-0.00175317, +0.00347265] | [-0.00168482, +0.00351185] | Unresolved |
| AdvTest / BM25 ENTR−PROB B | +0.00098599 | [-0.00219540, +0.00416737] | [-0.00223506, +0.00422136] | Unresolved |
| AdvTest / BM25 ENTR−PROB G | +0.00018021 | [-0.00125167, +0.00161210] | [-0.00128016, +0.00147299] | Unresolved |
| CoSQA D / CB ENTR B | +0.01321969 | [+0.00088510, +0.02555428] | [+0.00088210, +0.02481866] | Positive |
| CoSQA D / CB ENTR G | +0.01193674 | [+0.00488512, +0.01898835] | [+0.00490738, +0.01893323] | Positive |
| CoSQA D / CB PROB B | +0.01061844 | [-0.00038211, +0.02161898] | [-0.00071012, +0.02122361] | Unresolved |
| CoSQA D / CB PROB G | +0.01337689 | [+0.00646803, +0.02028576] | [+0.00640977, +0.02045887] | Positive |
| CoSQA D / CB ENTR−PROB B | +0.00260125 | [-0.00787221, +0.01307472] | [-0.00806805, +0.01277867] | Unresolved |
| CoSQA D / CB ENTR−PROB G | -0.00144016 | [-0.00678992, +0.00390960] | [-0.00679460, +0.00421164] | Unresolved |
| CoSQA D / BM25 ENTR B | -0.00167774 | [-0.00761380, +0.00425833] | [-0.00787790, +0.00396237] | Unresolved |
| CoSQA D / BM25 ENTR G | +0.00111513 | [-0.00246287, +0.00469313] | [-0.00249380, +0.00471802] | Unresolved |
| CoSQA D / BM25 PROB B | -0.00047832 | [-0.00619434, +0.00523770] | [-0.00624677, +0.00497573] | Unresolved |
| CoSQA D / BM25 PROB G | +0.00156260 | [-0.00238413, +0.00550933] | [-0.00213394, +0.00545720] | Unresolved |
| CoSQA D / BM25 ENTR−PROB B | -0.00119942 | [-0.00567494, +0.00327610] | [-0.00592436, +0.00301265] | Unresolved |
| CoSQA D / BM25 ENTR−PROB G | -0.00044746 | [-0.00289354, +0.00199861] | [-0.00277231, +0.00215915] | Unresolved |

Post hoc analysis, with 24 multiplicity-adjusted comparisons. AdvTest: 2,048 queries/594 groups; CoSQA D: 1,490/512. B=S(1,mean)−S(3,mean); G=S(3,mean)−S(3,best3). Positive B/G indicates a reduction in selector advantage. Direction requires both adjusted intervals on the same side of zero and sufficient nonzero groups. ENTR−PROB rows are direct component differences.

## Table S9. StaQC confirmation and descriptive controls

| Condition / component | Estimate | Cluster-t CI | Bootstrap CI | Direction |
| --- | --- | --- | --- | --- |
| StaQC / CB ENTR B | +0.00325690 | [+0.00172135, +0.00479245] | [+0.00174049, +0.00481414] | Positive |
| StaQC / CB ENTR G | +0.00360952 | [+0.00257670, +0.00464235] | [+0.00257627, +0.00464886] | Positive |
| StaQC / CB PROB B | +0.00334428 | [+0.00202162, +0.00466694] | [+0.00203870, +0.00468680] | Descriptive |
| StaQC / CB PROB G | +0.00454320 | [+0.00362730, +0.00545910] | [+0.00361597, +0.00546743] | Descriptive |
| StaQC / CB ENTR−PROB B | -0.00008738 | [-0.00146543, +0.00129067] | [-0.00147939, +0.00130673] | Descriptive |
| StaQC / CB ENTR−PROB G | -0.00093368 | [-0.00167943, -0.00018792] | [-0.00168251, -0.00019122] | Descriptive |
| StaQC / BM25 ENTR B | -0.00111320 | [-0.00195840, -0.00026800] | [-0.00194622, -0.00028469] | Descriptive |
| StaQC / BM25 ENTR G | +0.00066536 | [+0.00006839, +0.00126233] | [+0.00006729, +0.00127318] | Descriptive |
| StaQC / BM25 PROB B | -0.00045452 | [-0.00129984, +0.00039081] | [-0.00129531, +0.00038140] | Descriptive |
| StaQC / BM25 PROB G | +0.00037411 | [-0.00023690, +0.00098513] | [-0.00023369, +0.00097621] | Descriptive |
| StaQC / BM25 ENTR−PROB B | -0.00065868 | [-0.00145373, +0.00013637] | [-0.00146522, +0.00011509] | Descriptive |
| StaQC / BM25 ENTR−PROB G | +0.00029125 | [-0.00017803, +0.00076053] | [-0.00017763, +0.00075977] | Descriptive |

StaQC: 13,312 queries/groups and a 16,384-code corpus. The two CodeBERT ENTR B/G rows are the frozen confirmation family. All PROB/BM25 and direct-selector controls remain prespecified descriptive comparisons with ordinary 95% intervals. “Descriptive” is a role, not a directional decision. The design was locally sealed, not externally preregistered.

## Table S10. UniXcoder and paired retriever contrasts

| Condition / component | Estimate | Cluster-t CI | Bootstrap CI | Direction |
| --- | --- | --- | --- | --- |
| CoSQA D / UX ENTR B | +0.00552941 | [-0.00351994, +0.01457877] | [-0.00340806, +0.01463530] | Unresolved |
| CoSQA D / UX ENTR G | +0.00988737 | [+0.00459118, +0.01518356] | [+0.00438606, +0.01501782] | Positive |
| CoSQA D / UX−CB ENTR B | -0.00769027 | [-0.02021135, +0.00483080] | [-0.02011087, +0.00423375] | Unresolved |
| CoSQA D / UX−CB ENTR G | -0.00204936 | [-0.00996899, +0.00587026] | [-0.01001622, +0.00591535] | Unresolved |
| CoSQA D / UX−BM25 ENTR B | +0.00720715 | [-0.00330833, +0.01772264] | [-0.00295167, +0.01815455] | Unresolved |
| CoSQA D / UX−BM25 ENTR G | +0.00877224 | [+0.00234397, +0.01520050] | [+0.00210047, +0.01463773] | Positive |
| StaQC / UX ENTR B | +0.00220045 | [+0.00056790, +0.00383299] | [+0.00054416, +0.00384455] | Positive |
| StaQC / UX ENTR G | +0.00248619 | [+0.00134990, +0.00362249] | [+0.00132662, +0.00360560] | Positive |
| StaQC / UX−CB ENTR B | -0.00105645 | [-0.00347928, +0.00136637] | [-0.00354868, +0.00127622] | Unresolved |
| StaQC / UX−CB ENTR G | -0.00112333 | [-0.00281972, +0.00057306] | [-0.00283860, +0.00057611] | Unresolved |
| StaQC / UX−BM25 ENTR B | +0.00331365 | [+0.00126932, +0.00535797] | [+0.00114868, +0.00538720] | Positive |
| StaQC / UX−BM25 ENTR G | +0.00182083 | [+0.00041917, +0.00322249] | [+0.00037989, +0.00317908] | Positive |

Twelve adjusted comparisons on previously used CoSQA D and StaQC samples. UX=UniXcoder; CB=CodeBERT. Paired differences subtract the named retriever component using the same query outcomes and retain dependence. Unresolved direct differences do not establish retriever equivalence.

## Table S11. Matched two-weight B/G sensitivity

| Condition / component | Estimate | Cluster-t CI | Bootstrap CI | Direction |
| --- | --- | --- | --- | --- |
| AdvTest / local ENTR B | -0.00523902 | [-0.01968085, +0.00920282] | [-0.02007272, +0.00872153] | Unresolved |
| AdvTest / local ENTR G | +0.00347926 | [-0.00684997, +0.01380849] | [-0.00680788, +0.01380625] | Unresolved |
| AdvTest / local PROB B | -0.00203640 | [-0.01612976, +0.01205695] | [-0.01651426, +0.01181990] | Unresolved |
| AdvTest / local PROB G | +0.00502779 | [-0.00555955, +0.01561513] | [-0.00566646, +0.01672560] | Unresolved |
| AdvTest / t5base ENTR B | +0.00123669 | [-0.01182483, +0.01429821] | [-0.01219948, +0.01424884] | Unresolved |
| AdvTest / t5base ENTR G | +0.00548310 | [-0.00398600, +0.01495220] | [-0.00413981, +0.01484683] | Unresolved |
| AdvTest / t5base PROB B | +0.00723781 | [-0.00581928, +0.02029491] | [-0.00538510, +0.02073730] | Unresolved |
| AdvTest / t5base PROB G | +0.00934385 | [+0.00047983, +0.01820787] | [+0.00035460, +0.01804358] | Positive |
| AdvTest / base_minus_local ENTR B | +0.00647571 | [-0.01438962, +0.02734104] | [-0.01412962, +0.02732462] | Unresolved |
| AdvTest / base_minus_local ENTR G | +0.00200384 | [-0.01135239, +0.01536007] | [-0.01120842, +0.01532646] | Unresolved |
| AdvTest / base_minus_local PROB B | +0.00927422 | [-0.00949607, +0.02804450] | [-0.00902755, +0.02834715] | Unresolved |
| AdvTest / base_minus_local PROB G | +0.00431606 | [-0.00906134, +0.01769346] | [-0.00891878, +0.01779407] | Unresolved |
| CoSQA D / local ENTR B | +0.00581524 | [-0.01360608, +0.02523655] | [-0.01291157, +0.02523169] | Unresolved |
| CoSQA D / local ENTR G | +0.00986920 | [-0.00197935, +0.02171774] | [-0.00224435, +0.02182716] | Unresolved |
| CoSQA D / local PROB B | +0.01155186 | [-0.00793568, +0.03103940] | [-0.00842134, +0.03120363] | Unresolved |
| CoSQA D / local PROB G | +0.01162515 | [-0.00043762, +0.02368792] | [-0.00028326, +0.02333292] | Unresolved |
| CoSQA D / t5base ENTR B | +0.01896745 | [-0.00353662, +0.04147152] | [-0.00389371, +0.04094188] | Unresolved |
| CoSQA D / t5base ENTR G | +0.01208661 | [-0.00194116, +0.02611438] | [-0.00159665, +0.02594004] | Unresolved |
| CoSQA D / t5base PROB B | +0.01431626 | [-0.00697456, +0.03560708] | [-0.00673739, +0.03563037] | Unresolved |
| CoSQA D / t5base PROB G | +0.01835961 | [+0.00359494, +0.03312428] | [+0.00358998, +0.03331018] | Positive |
| CoSQA D / base_minus_local ENTR B | +0.01315221 | [-0.01615679, +0.04246122] | [-0.01622112, +0.04250096] | Unresolved |
| CoSQA D / base_minus_local ENTR G | +0.00221741 | [-0.01654053, +0.02097534] | [-0.01631975, +0.02043194] | Unresolved |
| CoSQA D / base_minus_local PROB B | +0.00276440 | [-0.02695878, +0.03248758] | [-0.02723466, +0.03282539] | Unresolved |
| CoSQA D / base_minus_local PROB G | +0.00673446 | [-0.01276299, +0.02623191] | [-0.01267514, +0.02626633] | Unresolved |

AdvTest and CoSQA each use 512 matched queries/groups, a fixed CodeBERT retriever, and each T5 weight's own pool. The 24-comparison family includes two domains, ENTR/PROB B/G and local/base/base-minus-local. Unresolved differences do not establish weight equivalence. Main Table II reports the separate six-comparison confirmation on 25,600 new groups.

## Table S12. Agreement of the original human-review returns

| Metric | n | Agreement | κ | Type | Coverage |
| --- | --- | --- | --- | --- | --- |
| semantics | 383 | 0.76501305 | 0.52514120 | nominal | — |
| relevance including unknown | 327 | 0.53822630 | 0.37064889 | nominal | — |
| relevance ordinal judged subset | 260 | 0.66923077 | 0.56850015 | linear weighted | 0.7951070336391437 |
| clarity | 128 | 0.72656250 | 0.02862099 | nominal | — |

Unweighted agreement for the original returned judgments. Unknown relevance is a category in the nominal analysis; linear-weighted κ uses the jointly judged 0/1/2 subset with coverage reported. These results are separate from the supplemental 344-panel-per-reviewer agreement. Both reviewers' original labels remain unchanged.

## Table S13. Weighted human-review query events

| Query event | Proportion | Conservative 95% CI | Estimated frame count |
| --- | --- | --- | --- |
| Any definite semantic issue (both reviewers) | 0.35005129 | [+0.09859320, +0.69601407] | 2985.9375 |
| Any rewrite not marked preserved by at least one reviewer | 0.75463072 | [+0.39601407, +0.95381008] | 6437.0000 |
| Any indeterminate semantics | 0.26869871 | [+0.05638921, +0.62579132] | 2292.0000 |
| Verified-stage adoption with definite semantic issue | 0.04161049 | [+0.00093787, +0.36565064] | 354.9375 |
| Verified-stage rank loss with preserved meaning | 0.12910317 | [+0.07620164, +0.40316530] | 1101.2500 |
| Unrealized meaning-preserving positive opportunity | 0.14283411 | [+0.04279015, +0.45896835] | 1218.3750 |
| Displayed non-target code relevant to both reviewers | 0.11992966 | [+0.01289566, +0.46178195] | 1023.0000 |
| Known target incomplete according to both reviewers | 0.45087925 | [+0.15662368, +0.77561547] | 3846.0000 |
| Reviewer A: any semantic issue | 0.75419109 | [+0.39554513, +0.95369285] | 6433.2500 |
| Reviewer B: any semantic issue | 0.37557151 | [+0.10914420, +0.72039859] | 3203.6250 |

Eight-stratum weights Nh/nh estimate query-event proportions in the fixed CodeBERT H8530 frame. Intervals combine simultaneous exact stratum-wise hypergeometric bounds under the pseudorandom sampling interpretation. Events overlap, so they are not causal shares or mutually exclusive error rates. Full stratum counts, weights and bounds remain in TABLES.json.

## Table S14. Weighted, overlapping semantic issue types

| Query event | Proportion | Conservative 95% CI | Estimated frame count |
| --- | --- | --- | --- |
| A: Unsupported constraint | 0.60723183 | [+0.26905041, +0.87854631] | 5179.6875 |
| A: Lost constraint | 0.00000000 | [+0.00000000, +0.30000000] | 0.0000 |
| A: Operation change | 0.07055979 | [+0.00539273, +0.39894490] | 601.8750 |
| A: Entity/API change | 0.03451055 | [+0.00058617, +0.35474795] | 294.3750 |
| A: Polarity change | 0.01885991 | [+0.00011723, +0.33001172] | 160.8750 |
| A: Incoherent | 0.14441676 | [+0.01852286, +0.49179367] | 1231.8750 |
| B: Unsupported constraint | 0.22178341 | [+0.05310668, +0.56096131] | 1891.8125 |
| B: Lost constraint | 0.01885991 | [+0.00011723, +0.33001172] | 160.8750 |
| B: Operation change | 0.10506301 | [+0.00750293, +0.45134818] | 896.1875 |
| B: Entity/API change | 0.06788540 | [+0.00597890, +0.39695193] | 579.0625 |
| B: Polarity change | 0.00000000 | [+0.00000000, +0.30000000] | 0.0000 |
| B: Incoherent | 0.02608441 | [+0.00035170, +0.34138335] | 222.5000 |
| Both, same rewrite: Unsupported constraint | 0.20757620 | [+0.04583822, +0.54888628] | 1770.6250 |
| Both, same rewrite: Lost constraint | 0.00000000 | [+0.00000000, +0.30000000] | 0.0000 |
| Both, same rewrite: Operation change | 0.04565504 | [+0.00199297, +0.36811254] | 389.4375 |
| Both, same rewrite: Entity/API change | 0.01480803 | [+0.00023447, +0.32344666] | 126.3125 |
| Both, same rewrite: Polarity change | 0.00000000 | [+0.00000000, +0.30000000] | 0.0000 |
| Both, same rewrite: Incoherent | 0.00682151 | [+0.00011723, +0.31078546] | 58.1875 |
| Either: Unsupported constraint | 0.60723183 | [+0.26905041, +0.87854631] | 5179.6875 |
| Either: Lost constraint | 0.01885991 | [+0.00011723, +0.33001172] | 160.8750 |
| Either: Operation change | 0.12996776 | [+0.01090270, +0.48206331] | 1108.6250 |
| Either: Entity/API change | 0.08714830 | [+0.00762016, +0.42203986] | 743.3750 |
| Either: Polarity change | 0.01885991 | [+0.00011723, +0.33001172] | 160.8750 |
| Either: Incoherent | 0.16367966 | [+0.02286049, +0.51359906] | 1396.1875 |

Each row is a query-level event marked by reviewer A, reviewer B, both on the same rewrite, or either reviewer. Rows can overlap. Estimates and conservative intervals concern the same finite H8530 frame as S13. The complete eight-stratum records remain in TABLES.json.

## Table S15. Matched-budget training sensitivity

| Domain / arm−L / metric | Estimate | Adjusted cluster-t CI | Adjusted bootstrap CI |
| --- | --- | --- | --- |
| AdvTest / O−L S3 | -0.00361666 | [-0.01753811, +0.01030480] | [-0.01775081, +0.00920340] |
| AdvTest / O−L top1_gain | -0.01083141 | [-0.02977581, +0.00811299] | [-0.03099507, +0.00696240] |
| AdvTest / S−L S3 | -0.00279246 | [-0.01831817, +0.01273326] | [-0.01879184, +0.01271154] |
| AdvTest / S−L top1_gain | -0.01095998 | [-0.02773445, +0.00581449] | [-0.02821080, +0.00472164] |
| AdvTest / N−L S3 | +0.00000000 | [+0.00000000, +0.00000000] | [+0.00000000, +0.00000000] |
| AdvTest / N−L top1_gain | +0.00000000 | [+0.00000000, +0.00000000] | [+0.00000000, +0.00000000] |
| AdvTest / W−L S3 | +0.00149885 | [-0.01151804, +0.01451573] | [-0.01173091, +0.01451837] |
| AdvTest / W−L top1_gain | -0.00890648 | [-0.02723410, +0.00942114] | [-0.02675286, +0.00892952] |
| AdvTest / P−L S3 | -0.00114332 | [-0.01365588, +0.01136924] | [-0.01347887, +0.01066236] |
| AdvTest / P−L top1_gain | -0.01179079 | [-0.02865390, +0.00507232] | [-0.02949673, +0.00379373] |
| AdvTest / PN−L S3 | +0.00507610 | [-0.00998714, +0.02013934] | [-0.00944618, +0.02073715] |
| AdvTest / PN−L top1_gain | -0.00693821 | [-0.02303554, +0.00915911] | [-0.02340819, +0.00919458] |
| AdvTest / PS−L S3 | +0.00240475 | [-0.01040835, +0.01521784] | [-0.01035257, +0.01576811] |
| AdvTest / PS−L top1_gain | -0.01611474 | [-0.03221445, -0.00001503] | [-0.03271290, -0.00059109] |
| CoSQA D / O−L S3 | +0.00386055 | [-0.01656198, +0.02428308] | [-0.01701880, +0.02440301] |
| CoSQA D / O−L top1_gain | -0.00507920 | [-0.02632109, +0.01616268] | [-0.02663010, +0.01518443] |
| CoSQA D / S−L S3 | +0.00134978 | [-0.01711086, +0.01981042] | [-0.01738840, +0.01994906] |
| CoSQA D / S−L top1_gain | +0.00245741 | [-0.02047427, +0.02538909] | [-0.01975316, +0.02413766] |
| CoSQA D / N−L S3 | +0.00000000 | [+0.00000000, +0.00000000] | [+0.00000000, +0.00000000] |
| CoSQA D / N−L top1_gain | +0.00000000 | [+0.00000000, +0.00000000] | [+0.00000000, +0.00000000] |
| CoSQA D / W−L S3 | +0.00308944 | [-0.01558002, +0.02175889] | [-0.01489043, +0.02226375] |
| CoSQA D / W−L top1_gain | -0.00733077 | [-0.02741488, +0.01275335] | [-0.02768186, +0.01242557] |
| CoSQA D / P−L S3 | -0.00396651 | [-0.02305141, +0.01511840] | [-0.02318952, +0.01510376] |
| CoSQA D / P−L top1_gain | +0.00148675 | [-0.01935724, +0.02233074] | [-0.01792189, +0.02302523] |
| CoSQA D / PN−L S3 | +0.00279995 | [-0.01496912, +0.02056901] | [-0.01364006, +0.02132589] |
| CoSQA D / PN−L top1_gain | +0.00555303 | [-0.01541460, +0.02652066] | [-0.01384100, +0.02485300] |
| CoSQA D / PS−L S3 | +0.00655420 | [-0.01161185, +0.02472025] | [-0.01205323, +0.02486696] |
| CoSQA D / PS−L top1_gain | -0.01427124 | [-0.03513717, +0.00659468] | [-0.03628325, +0.00488926] |

Eight arms share 32,768 training rows and 1,024 updates. Each displayed contrast subtracts baseline L; own denotes the arm's own generated pool. S3 is ENTR best3 minus exact-random best3; Top1 gain is relative to the original query. The original family contains 28 comparisons. These finite-budget controls do not recover the full effects of historical 93,750-update training. Settings and the separate fixed-historical-pool view are documented in S21 and the source protocol.

## Table S16. Software and hardware environment

| Item | Configuration | Source |
| --- | --- | --- |
| Operating system | Windows; runtime identifier Windows-10-10.0.26200-SP0 (build 26200) | R01, R02 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU; 8,151 MiB memory recorded at runtime | R01, R02 |
| GPU driver | 577.05 | R01 |
| CPU | AMD Ryzen 9 8945HX with Radeon Graphics; 16 cores and 32 logical processors (hardware recheck) | R34 |
| Memory | 32 GiB (hardware recheck) | R34 |
| Python | 3.10.11, 64-bit | R01, R02 |
| PyTorch / CUDA | 2.7.0+cu128 / 12.8 (PyTorch build and runtime) | R01, R02 |
| Transformers / Tokenizers | 4.57.6 / 0.22.2 | R01 |
| SentencePiece / Safetensors | 0.2.2 / 0.8.0 | R01 |
| NumPy / SciPy | 2.2.6 / 1.15.3 | R01, R02 |
| Scikit-learn / Joblib | 1.7.2 / 1.6.0 | R07, R09 |
| Threadpoolctl / Matplotlib | 3.6.0 / 3.10.9 | R01 |
| Training precision | Main T5, CodeBERT and limited-training controls use BF16; limited-control model weights are FP32 | R03, R05, R15 |
| Generation and retrieval precision | Main T5 generation/scoring and neural retrieval use FP32 with TF32 disabled; cross-encoder verification uses BF16 autocast and converts outputs to FP32 | R02, R06, R20 |
| Inference state | Models use eval; dropout is disabled for generation, scoring and retrieval; training retains each model's training configuration | R17, R20, R31 |

Software, GPU and driver values come from historical run records. CPU and memory were checked when the original settings supplement was compiled. CUDA 12.8 denotes the runtime used by PyTorch.

## Table S17. Training settings for the rewriter, neural retriever and verifier

| Parameter | Local CQC T5 | CodeBERT retriever | Cross-encoder verifier |
| --- | --- | --- | --- |
| Base model | T5-base | CodeBERT-base | CodeBERT-base |
| Training data | First 1,000,000 comments in the local CODEnn source | 251,820 CodeXGLUE query-code pairs | Historical large_dense_4096: 4,096 queries, 16,384 pairs |
| Objective | Recover one contiguous masked span of length max(1, round(0.15n)) | Full query-to-code in-batch cross-entropy | Binary relevance BCEWithLogits |
| Epochs / updates | 3 / 93,750 | 2 / 15,740 | 1 / 512 |
| Microbatch / accumulation | 8 / 4 | 32 / 1 | 8 / 4 |
| Effective batch | 32 | 32 | 32 |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 0.001 | 0.00005 | 0.00002 |
| Schedule | Constant | 10% warmup (1,574 updates), then linear decay | Constant |
| Weight decay | 0.01 (execution-code default) | 0 | 0.01 |
| Gradient clipping | L2 norm capped at 1 | L2 norm capped at 1 | L2 norm capped at 1 |
| Input length | 512 | 256 each for query and code | 64 query / 160 code, plus special tokens |
| Seed | 101 | 123456 | 141414 |
| Weight selection | Best source-validation loss, update 92,000 | Prespecified final weights after two epochs | Historical completed-training weights, frozen before H inference |
| Training precision | BF16 | BF16 | BF16 autocast |

Sources: T5, R03/R04/R29; CodeBERT, R05; verifier, R27/R28. The verifier retains historical training weights. Its data and negative construction remain in the source code; it was not retrained using H labels.
Main T5 training completed 93,750 updates. Generation uses the source-validation-best weights at update 92,000. This training is separate from the eight 1,024-update sensitivity configurations.

## Table S18. Generation, scoring, selection and retrieval settings

| Stage | Setting | Source |
| --- | --- | --- |
| Main insertion positions | Split into n whitespace-delimited words; insert after each word from 1 through n; generate 1 candidate per position | R06, R17 |
| Main generation | greedy; do_sample=false; num_beams=1; num_return_sequences=1; max_new_tokens=19 | R06, R14 |
| Length convention | Decoder length at most 20 including the start token; inputs longer than 512 raise an error rather than being silently truncated | R17 |
| Batch sizes and seed | Early/CoSQA/limited-training generation: 8; StaQC and new paired-weight generation: batch 32 and query block 32; seed 101 throughout | R06, R07, R11, R14, R16 |
| T5 tokenization | T5Tokenizer legacy=True; local T5-base SentencePiece; saved tokenizer vocabulary checked | R17 |
| ENTR / PROB | Same generated label sequence; ENTR averages full-vocabulary predictive entropy over non-padding positions (lower first); PROB averages token log probabilities (higher first) | R17 |
| Scoring labels | <extra_id_0> + span + <extra_id_1>, including EOS; break score ties by original insertion position | R17 |
| Selection and aggregation | k=1,2,3, capped at pool size; mean / label-assisted best-k; no additional original-query option | R26, R14 |
| Duplicates and empty spans | Retain original positions and their selection probability; do not pad short pools with the original query | R11, R14 |
| Exact random / RAND20 | Exact random is the uniform expectation over positions sampled without replacement; RAND20 seeds 2026101900–2026101919 | R06, R07, R26 |
| CodeBERT representation | pooler_output; no normalization; FP32 raw dot product; query and code lengths capped at 256 | R05, R11 |
| CodeBERT batch sizes | CoSQA H encoding: 16; new paired-weight StaQC encoding: 32; ranking score block: 128 | R20, R14 |
| BM25 | k1=1.2, b=0.75; IDF=ln(1+(N−df+0.5)/(df+0.5)); local lexical implementation | R07, R18 |
| BM25 tokenization and scoring | Split lower-case/digit-to-upper-case camel boundaries, then extract lower-case [a-z][a-z0-9]*; unique query terms; no document truncation; FP64 | R18 |
| UniXcoder | <encoder-only>; query 128, code 256; mean final-layer non-padding token representations including special tokens; L2 normalization and FP32 dot product; batch 16 | R12, R31 |
| Ranking | Top100; descending score, then ascending original document ID; all within-condition controls fix corpus and labels | R06, R11, R12, R14 |

temperature=1, top_k=0 and top_p=1 apply to sampling sensitivity configurations; the main greedy setting does not use these sampling cutoffs. Selection k counts candidates retained from a pool; generation still enumerates all specified positions.

## Table S19. Study weight identities and checksums

| Model | Identity and use | Weight-file SHA-256 |
| --- | --- | --- |
| W1 Local CQC T5 | Independent CQC training from T5-base; source-validation-best weights at update 92,000 | 7acd4cb8379a1f5297bb978a748bc002b708ef5fd60a8b4737a6f8c7d2ebcaa5 |
| W2 Public T5-base | Public base weights; paired-weight comparison and shared initialization for limited training | a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4 |
| W3 CodeBERT retriever | SSQR pooler/dot-product objective; final weights after two epochs | 0d533c04602916c22d1f13d282056f081b6bd04f9866ccc1ded46a0516e13e3d |
| W4 UniXcoder-base | Public zero-shot weights; revision 5604afdc964f6c53782a6813140ade5216b99006 | 6732b60037b086557e4c731c3748c99b4fec735d9ffe3c4c70061fd142e6b9d9 |
| W5 CodeBERT cross-encoder | Historical large_dense_4096 cross-encoder; frozen before full H | 3a0f26dcae197b2e7cb97b62349ccd743e4211a835a8e2ab2842376a7c4c2f39 |

Sealed identities: W1–W3, R14; W4, R12; W5, R07. The original settings audit calculated each file hash and matched the sealed record. File locations are listed in the source index.
Paired-weight analyses use each weight's own complete candidate pool. Queries, corpus, labels and retriever are matched; candidate pools are not assumed identical across weights.

## Table S20. Six generation configurations

| Configuration | Positions | Decoding | Max. new tokens | Length incl. start |
| --- | --- | --- | --- | --- |
| greedy20_source | 1…n | Greedy | 19 | 20 |
| greedy20_all | 0…n | Greedy | 19 | 20 |
| greedy10_source | 1…n | Greedy | 9 | 10 |
| greedy10_all | 0…n | Greedy | 9 | 10 |
| sample10_source | 1…n | Sampling | 9 | 10 |
| sample10_all | 0…n | Sampling | 9 | 10 |

Source: R06. Each domain fixes 512 queries/groups matched across configurations. Sampling uses temperature=1, top_k=0 and top_p=1. SHA-256(seed=101, query_id, position) sets each query-position random stream; one candidate is generated per position. Greedy batch size is 8; sampling batch size is 1.

## Table S21. Eight training configurations with matched update budgets

| Arm | Optimizer | Schedule | Schedule horizon | Length | Weight decay | Seed |
| --- | --- | --- | --- | --- | --- | --- |
| L | AdamW | Constant | — | 512 | 0.01 | 101 |
| O | Adafactor | Constant | — | 512 | 0.01 | 101 |
| S | AdamW | Linear | 100,000 | 512 | 0.01 | 101 |
| N | AdamW | Constant | — | 128 | 0.01 | 101 |
| W | AdamW | Constant | — | 512 | 0.0 | 101 |
| P | Adafactor | Linear | 100,000 | 128 | 0.0 | 101 |
| PN | Adafactor | Linear | 1,093 | 128 | 0.0 | 101 |
| PS | Adafactor | Linear | 100,000 | 128 | 0.0 | 42 |

Sources: R15/R16/R30. All arms start from the same T5-base weights, share 32,768 training rows and data order, and run 1,024 updates. Microbatch 8; accumulation 4; effective batch 32; initial learning rate 0.001; gradient norm cap 1; BF16 computation with FP32 weights.
The linear schedule horizon differs from the actual 1,024-update budget. Adafactor fixes scale_parameter=false and relative_step=false. Each arm uses final weights without early-stopping selection. Evaluation separately retains own-generated pools and rescoring of the fixed historical pool.

## Table S22. Fixed-action adoption and rank-fusion settings

| Item | Setting | Source |
| --- | --- | --- |
| Sample and action | CoSQA D: 1,490 queries in 512 connected groups; the same ENTR Top1 action is fixed | R09 |
| Training target | Fixed-action RR@100 minus original-query RR@100; queries weighted equally; relevance labels do not enter features | R09 |
| Feature blocks | M0=27, MT=37, MR=61, MRT=71 dimensions; complete feature names and partitions remain in the protocol | R09 |
| Fitting | StandardScaler + Ridge; alpha=1; solver=svd; intercept included; no prediction clipping or tuning | R09 |
| OOF split | Reuse 5 folds of whole connected groups; fit standardization and models using training folds only | R09 |
| Eligible actions | Change both query text and the full Top100 ranking; all prediction heads share this observable requirement | R09 |
| Fixed20 | In each held-out fold adopt floor(eligible actions/5) with the highest predictions, including negative scores; break ties by ascending query_id | R09 |
| Zero threshold | Adopt only eligible actions with raw predicted gain >0; both policies use all D queries as denominator | R09 |
| Resampling scope | Conditional on fitted OOF models; do not refit or change quotas within bootstrap resamples | R09 |
| RRF inputs | RRF3 uses three ENTR lists; RRF4 additionally includes the original-query list; deduplicate complete input text before equal-weight fusion | R10 |
| RRF parameters | Constant 60; input and output depth 100; score Σ1/(60+rank) | R10 |
| RRF arithmetic and ties | Integer ranking equivalent to exact rational arithmetic; ascending document-ID ties; return top 100 from the retrieved-document union | R10 |
| Workload | Count deduplicated retrieval inputs; do not interpret counts as online latency; no risk gate in this control | R10 |

Fixed-action Ridge uses alpha=1. Full-H models in Table S23 use alpha=10. These are separate experiments with their original settings.

## Table S23. Frozen full-H models and decision rules

| Item | Setting | Source |
| --- | --- | --- |
| Three D roles | Gain fit: 953 queries/307 groups; verification-update fit: 288/103; risk fit: 249/102; mutually exclusive whole groups | R08 |
| Evaluation sample | Full H: 8,530 queries in 1,963 groups; inference follows freezing of fitting and thresholds | R07 |
| Gain prediction | 71 actual input dimensions; StandardScaler + Ridge alpha=10; weight each candidate by 1/candidate count within query | R07, R19 |
| Base harm prediction | StandardScaler + LogisticRegression; C=0.1, lbfgs, max_iter=2000, random_state=261019 | R19 |
| Gain-stage adoption | Choose the highest predicted gain within the same ENTR min(3,n) set; adopt only if prediction >0 and ranking changes; clip prediction to [−1,1] | R19 |
| Verification quota | Order positive base predicted gains; at most min(floor(N/4), eligible queries); break ties by query_id | R07, R19 |
| Verification evidence | 8 distinct codes per query; construct from original-query success anchors, current winner/competing candidates and rank shifts | R07, R19 |
| Verifier input | Original query paired with code; truncate separately to 64/160 tokens plus special tokens; inference batch 32 | R20 |
| Verifier precision | Frozen eval model; BF16 autocast with FP32 output; scoring uses the original query rather than the rewrite | R20 |
| Update model | 71+2+18=91 dimensions; weighted StandardScaler + Ridge alpha=10; add 0.5×predicted residual to the prediction, then clip to [−1,1] | R19 |
| Risk model | negative/zero/positive three-class model; zero is reference; base-probability offset; 20 inputs; mask duplicate verification-availability column | R07, R19 |
| Risk regularization and optimizer | Intercept penalty 1; verification-indicator coefficient penalty 50; other coefficients 10; L-BFGS-B, maxiter=2000, ftol=1e−11, gtol=1e−7 | R19 |
| Fallback | Threshold 0.075; when exceeded, restore the original full Top100 without choosing another candidate; retain incurred verification cost | R07, R19 |
| Numerical zero and small samples | TOL=1e−10; zero residual below 30 fitting rows; retain probability offset below 30 risk rows or with fewer than two classes | R19 |

Full-H Ridge alpha=10 differs from the fixed-action alpha=1 in Table S22. The risk threshold is an execution rule; observed harm counts do not certify population risk.
HGB secondary control (R07): squared_error; max_iter=200; learning_rate=0.05; max_leaf_nodes=7; max_depth=3; min_samples_leaf=30; l2_regularization=10; max_bins=63; early_stopping=false; random_state=20261019.

## Table S24. Comparison families, resampling and study samples

| Analysis | Queries/groups and corpus | Family | Resamples / seed | Primary setting |
| --- | --- | --- | --- | --- |
| Specified CoSQA replication | Capped H: 7,718 queries / 1,963 groups; at most 16 per group | 1 | 10,000 / 261019 | Two-sided 95% dual intervals; strictly within ±0.005 and at least 20 changed groups |
| Full-H stages | Full H: 8,530 queries / 1,963 groups | Descriptive | 10,000 / 261019 | Stage-mean intervals; retention-ratio seed 261020 |
| Same-pool B/G | AdvTest 2,048/594; CoSQA D 1,490/512; two retrievers | 24 | 10,000 / 20261020 | ENTR/PROB B/G and direct component differences |
| StaQC confirmation | 13,312 queries/groups; corpus of 16,384 codes | 2 | 10,000 / 202610202 | CodeBERT ENTR B/G; PROB/BM25 descriptive |
| UniXcoder extension | Reuse CoSQA D and the preceding StaQC sample | 12 | 10,000 / 202610203 | B/G in both domains and paired differences versus CodeBERT/BM25 |
| Matched weights | 512 queries/groups per AdvTest/CoSQA domain | 24 | 20,000 / 202610204 | Two domains × two selectors × B/G × within-weight effects and direct differences |
| New paired-weight confirmation | 25,600 new StaQC queries/groups; corpus of 32,768 codes | 6 | 20,000 / 202610215 | ENTR B/G under both weights and paired weight differences; no equivalence test |
| Fixed-action adoption | CoSQA D 1,490/512 | 8 | 10,000 / 20261020 | Two conditional increments × two adoption policies × two retrievers |
| RRF fusion | CoSQA D 1,490/512 | 4 | 10,000 / 20261020 | RRF3/RRF4 versus original query; two retrievers |
| Limited training | 512 queries/groups per AdvTest/CoSQA domain | 28 | 10,000 / 20261020 | Retain sealed primary family and both candidate-pool views |

Sources: R07/R09–R16/R21/R26. Queries are the weighting unit. Resampling retains whole connected or repository groups and shares paired resamples within each study. Multiplicity-adjusted comparisons use Bonferroni cluster-t and group-percentile-bootstrap intervals.
B/G direction requires both adjusted intervals to exclude zero on the same side and at least 20 nonzero groups. Study families and populations remain separate. CoSQA ±0.005 is its specified equivalence margin; 0.005 in new paired-weight planning is a precision target.
The limited-training resampling seed follows the actual analysis entry R35; other settings follow R07/R09–R16/R21/R26.

## Table S25. Human review and supplemental relevance judgments

| Item | Setting | Source |
| --- | --- | --- |
| Frame and strata | CodeBERT full H8530; 8 disjoint strata fixed before collecting labels; 16 queries sampled per stratum | R22 |
| Fixed sampling rule | SHA-256 pseudorandom ordering; salt rene-mechanism-review-20261020-v1; exact concatenation rule in source code | R22, R23 |
| Review size | 128 queries; each of two reviewers independently assesses 383 distinct rewrites and 327 code panels | R33 |
| Blinding | Hide method, score, ranking outcome, stratum and target role; separate semantics from code-relevance assessment | R33 |
| Semantic classes | Meaning-preserving, meaning-shifting, unreadable, indeterminate; rationale for every item | R33 |
| Code classes | 2/1/0 or unknown, relative to original query; unknown is not 0; displayed panels do not relabel the full corpus | R33 |
| Agreement | Semantic exact agreement and nominal Cohen κ; nominal code κ including unknown; linear-weighted κ and coverage on jointly judged panels | R33 |
| Weighted events | Original Nh/nh stratification weights; overlapping events; simultaneous stratum-wise hypergeometric bounds and fixed-H8530 95% intervals | R22, R33 |
| Top-3 supplemental review | Union of original-query and three-candidate Top-3 lists: 610 panels; reuse 266 judgments and add 344 per reviewer across 126 of the same queries | R25, R33 |
| Primary supplemental diagnostic | Fixed RIDGE_verified, RR@3, grade-2 relevance, original eight-stratum weights; share unknown relevance assignments for identical text | R25 |
| Sensitivity settings | Also report grades 1/2, unweighted analyses, each stage and RR@1; retain both reviewers' labels and disagreements | R25 |
| Identification and sampling envelopes | Sample bounds handle unjudged relevance; finite-H8530 sampling envelopes separately use hypergeometric inversion and Bonferroni; Neyman/Welch–Satterthwaite t is approximate | R24 |
| Coverage scope | Simultaneous envelopes for four reviewer × threshold primary-policy combinations; no claim of joint coverage for all 48 combinations; annotation error excluded | R24, R25 |

Sources: R22–R25 and Chinese-manuscript snapshot R33. Original independent labels, supplemental judgments and disagreements remain separate. Diagnostic RR@3, sample identification bounds and sampling envelopes do not replace full-H MRR@100.
Both reviewers gave informed consent before participating, after learning the study purpose and review tasks, and agreed to share de-identified annotations and rationales. The paper identifies reviewers as A/B and reports aggregate statistics; the reproduction materials retain anonymous item-level annotations and rationales without reviewer identities. No formal ethics review or exemption determination was sought for this study.
