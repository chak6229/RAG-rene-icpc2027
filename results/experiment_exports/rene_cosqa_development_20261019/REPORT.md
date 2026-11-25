# CoSQA Full development and same candidate direct benefit selection

Developments: 1,490 Articles enquiring, 512 associated groups; Add a new 1,234/426 group. Candidature 9,915, of which 1,734 is from the original storage test run.
Confirms that the pool 8,530/1,963 group remains inactive. All results are evidence of development; Keep the original run and the new group separately.

## RQ1: Candidate set quality and automatic Top1

| Subset | retriever | Original Query MRR | ENTR best3 Δ | PROB best3 Δ | RAND20 best3 Δ | ENTR Top1 Δ |
|---|---|---:|---:|---:|---:|---:|
| Develop All | CodeBERT | 0.51890040 | +0.06037845 | +0.05942630 | +0.05801045 | -0.01365851 |
| Develop All | BM25 | 0.22412467 | +0.00977629 | +0.00920534 | +0.01172399 | -0.02035096 |
| Add Development | CodeBERT | 0.52345225 | +0.05826850 | +0.05702422 | +0.05651256 | -0.01192462 |
| Add Development | BM25 | 0.23070947 | +0.00740594 | +0.00689088 | +0.01075713 | -0.01941608 |
| Original Run | CodeBERT | 0.49695906 | +0.07054904 | +0.07100508 | +0.06523072 | -0.02201640 |
| Original Run | BM25 | 0.19238386 | +0.02120211 | +0.02036177 | +0.01638454 | -0.02485738 |

best3 Select the best candidate using a true relevance label, which is the upper limit of quality; Automatic gains are given by actual decision-making in the following table.

| Total Developer Comparisons | Difference | 95% Cluster | Four years at the same time, 98.75% of the area. | Holm p | Change Group |
|---|---:|---|---|---:|---:|
| CodeBERT:ENTR_minus_RAND_best3 | +0.00236800 | [-0.0034829919970795317, 0.008060199498210272] | [-0.005275393345087954, 0.009579717389316109] | 1 | 359 |
| CodeBERT:ENTR_minus_PROB_best3 | +0.00095215 | [-0.003070399740126509, 0.004884943190997341] | [-0.004205847483185126, 0.006135158387294364] | 1 | 99 |
| BM25:ENTR_minus_RAND_best3 | -0.00194770 | [-0.004953250553748411, 0.0010438214673831878] | [-0.005808722000289659, 0.0016967222468728047] | 0.806272 | 294 |
| BM25:ENTR_minus_PROB_best3 | +0.00057095 | [-0.0015523790220462557, 0.0027322601464592005] | [-0.002214865598753, 0.003323102854175866] | 1 | 85 |

## RQ2: Fixed ENTR Auto-decision of the Triple Pool

The study fixed 71 features, five group-separated folds, model parameters and a zero predicted-gain threshold in advance. GEN_HGB is the generation-and-text feature control. GAIN_RIDGE and the main GAIN_HGB model use retrieval evidence. All arms reuse existing caches.
forced Selection of candidates in the same pool; selective Use the original query as a known zero-gain action. Calculate using a change in ranking by full Top100.

| retriever | Decision-making | ΔMRR | 95% Cluster | Actual use | Harmful | Use of internal harmful rates |
|---|---|---:|---|---:|---:|---:|
| CodeBERT | ENTR_top1 | -0.01365851 | [-0.02278317550167372, -0.00471390429051388] | 1490/1490 | 388 | 26.04% |
| CodeBERT | PROB_in_ENTR3 | -0.01469737 | [-0.023829920842391966, -0.005672562686406501] | 1490/1490 | 390 | 26.17% |
| CodeBERT | GEN_HGB_forced | -0.01035448 | [-0.019543475433057317, -0.0013103139598761866] | 1490/1490 | 398 | 26.71% |
| CodeBERT | GEN_HGB_selective | -0.00276537 | [-0.00918193510184267, 0.0035475800836726667] | 624/1490 | 171 | 27.40% |
| CodeBERT | GAIN_RIDGE_forced | +0.00228470 | [-0.004185601022269578, 0.008711168467492805] | 1490/1490 | 336 | 22.55% |
| CodeBERT | GAIN_RIDGE_selective | +0.00409250 | [-0.0007154683728199816, 0.00899725770006996] | 1048/1490 | 230 | 21.95% |
| CodeBERT | GAIN_HGB_forced | -0.00123662 | [-0.009260756374958538, 0.006908237507686326] | 1490/1490 | 352 | 23.62% |
| CodeBERT | GAIN_HGB_selective | +0.00020512 | [-0.006559487706033841, 0.006959119368519185] | 958/1490 | 216 | 22.55% |
| BM25 | ENTR_top1 | -0.02035096 | [-0.025629373791534852, -0.015299321038490896] | 1382/1490 | 555 | 40.16% |
| BM25 | PROB_in_ENTR3 | -0.01896348 | [-0.024151011543841434, -0.014112625161852643] | 1370/1490 | 553 | 40.36% |
| BM25 | GEN_HGB_forced | -0.01416430 | [-0.019443601062307143, -0.009035317309968639] | 1204/1490 | 459 | 38.12% |
| BM25 | GEN_HGB_selective | -0.00407599 | [-0.006230425028583611, -0.0022216210033838865] | 217/1490 | 85 | 39.17% |
| BM25 | GAIN_RIDGE_forced | -0.00654378 | [-0.0091686851385096, -0.0039248610866578736] | 1276/1490 | 436 | 34.17% |
| BM25 | GAIN_RIDGE_selective | -0.00248255 | [-0.0036330173947350204, -0.0015351484645518482] | 603/1490 | 201 | 33.33% |
| BM25 | GAIN_HGB_forced | -0.00994891 | [-0.013499181451750041, -0.006611173379824396] | 1249/1490 | 434 | 34.75% |
| BM25 | GAIN_HGB_selective | -0.00499483 | [-0.007547461279136794, -0.002639743478584744] | 447/1490 | 161 | 36.02% |

| Main Selection Comparison | Pair Incremental | 95% Cluster | Four comparisons Holm p |
|---|---:|---|---:|
| CodeBERT:GAIN_HGB_forced_minus_ENTR_top1 | +0.01242189 | [0.0027994337552855316, 0.02242440593946667] | 0.0291223 |
| CodeBERT:GAIN_HGB_selective_minus_NoRewrite | +0.00020512 | [-0.006559487706033841, 0.006959119368519185] | 0.952315 |
| BM25:GAIN_HGB_forced_minus_ENTR_top1 | +0.01040205 | [0.005506739047250899, 0.01544494971449307] | 0.000214222 |
| BM25:GAIN_HGB_selective_minus_NoRewrite | -0.00499483 | [-0.007547461279136794, -0.002639743478584744] | 0.000257617 |

OOF Intersections describe inter-group variations for fixed external discount projections and do not contain full retrainment uncertainties. This will require freezing the single policy to be developed in its entirety. This round is open to door or authentication.

## Identification of effectiveness planning

Poor target...005, 80% efficacy; The following table follows RQ1 of the four ICPs. Save RQ1+RQ2 Eight is more sensitive.

| RQ1 Contrast | Original 86 Group Planning | 512 Group Conservative Planning | Change Group | 1,963 Group capacity is sufficient | Group size sensitivity planning |
|---|---:|---:|---:|---|---:|
| CodeBERT:ENTR_minus_RAND_best3 | 2448 | 2475 | 359 | False | 3662 |
| CodeBERT:ENTR_minus_PROB_best3 | 1617 | 1262 | 99 | True | 1868 |
| BM25:ENTR_minus_RAND_best3 | 1059 | 691 | 294 | True | 1023 |
| BM25:ENTR_minus_PROB_best3 | 433 | 493 | 85 | True | 730 |

D/H The median size of the query group is 2 / 3, with a maximum of 42 / 137. (b) H/D Design factor ratio to 1.4795, under the group mean variance homogeneity assumption; This is a planning sensitivity, not a known difference in recognition.
The determination of size, joint multiplicity and complete adoption policy has not yet been frozen and the above development point estimates cannot be used as a confirmed conclusion.

## Receiving and Inspection and Restatement

Independent ranking/full wordsheet FP64 recalculated pass; 71,520 RR Fields are rationalized, with maximum error 2.22e-16. 30 Reloading of a folding model with all predictions due to error 0.
The original 256 list of candidates, rankings and the results of each query are kept in full; Language vector is not recalculated and T5 seed/mask is not added. Five formal points and model weights are unchanged, risk_certified = false, with deployment remaining NoRewrite.
This round is a source code-led reconstruction: position 1 n, greedy19, freezing local weights; Discrepancies such as 0... n/sampling m10 of the original paper continue to be disclosed according to the original agreement. Only known positive targets and known upstream code exposures follow the data review range.

Main evidence: PROTOCOL.json, DECISIONS_SEALED.json, PREOUTCOME_AUDIT.json, POSTOUTCOME_AUDIT.json, RQ1_DEVELOPMENT.json, RQ2_DEVELOPMENT.json, POWER_PLAN.json.
