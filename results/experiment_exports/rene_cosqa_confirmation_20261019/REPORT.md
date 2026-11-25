# RENE independent confirmation: precision and full-stage verification

The original design with four primary comparisons required 2475 groups, or 3662 groups under the D/H group-size sensitivity. The available 1963 groups could not meet these targets. The confirmation design was revised and the prior plan was retained.

The revised primary comparison returns to CodeBERT ENTR−RAND in the unified plan. It retains all 1963 linked groups and selects 7718 queries in the fixed SHA256 order, capping each group at 16. The metric remains query-weighted MRR in the frozen sample. The precision calculation applies to that design.

The within-group cap was selected by feasibility under D variances. The prespecified options were 4/8/12/16/all. The largest cap whose conservative group requirement stayed within 95% of capacity was chosen. This used neither H predictions or labels nor the sign of the D effects. The full 8530-query analysis was specified in advance as a sensitivity analysis.

The target effect remained .005, with two-sided alpha=.05 and 80% power. The 95% upper bound on D variance from 10000 group bootstrap resamples was 0.0055310118. The group-size sensitivity multiplier was 1.055852. The conservative requirement was 1836 groups compared with capacity of 1963 groups. Planned power was 82.58%, and the minimum detectable difference at 80% power was 0.004834.

The precision plan assumes transferable conditional variance from D to H and independent groups. Unobserved H variance remains uncertain. Actual intervals are reported for the observed effects.

The primary ΔMRR was -0.00083652. The t 95% interval was [-0.0032099252076455208, 0.00153688292345661], and the group bootstrap 95% interval was [-0.0032547866853966396, 0.0015414627313371035]. There were 1500 nonzero groups.

Interpret by freezing rules:`{'direction': 'difference from zero unresolved', 'practical_equivalence_within_prespecified_plusminus005': True, 'advantage_above005_excluded': True, 'does_not_imply_all_UQ_is_ineffective': True}`.

An interval spanning 0 leaves the effect direction unresolved. Practical equivalence requires both 95% intervals to lie wholly within ±.005 and at least 20 nonzero groups. The rule applies to the specified negative-entropy position-selection comparison.

The full analysis comprises 8530 queries in 1963 linked groups. Both retrievers use the same three ENTR candidates, or min(3,N) when the pool is smaller, and the frozen policy. The stage results and their intervals below were specified in advance as descriptive estimates.

| retriever | Phase | ΔMRR@100 | Use/harmful | Use of internal harmful rates | Retention of positive proceeds | CE logarithm |
|---|---|---:|---:|---:|---:|---:|
| CodeBERT | ENTR Candidate security oracle | +0.06609266 | — | — | 100% | 0 |
| CodeBERT | ENTR_top1 | -0.02110467 | 8530/2328 | 27.29% | 42.52% | 0 |
| CodeBERT | HGB_gain_control | -0.00287610 | 5305/1277 | 24.07% | 27.02% | 0 |
| CodeBERT | RIDGE_gain | +0.00007864 | 6034/1317 | 21.83% | 23.04% | 0 |
| CodeBERT | RIDGE_verified | +0.00084705 | 5890/1280 | 21.73% | 24.35% | 17056 |
| CodeBERT | RIDGE_risk | -0.00017777 | 1006/60 | 5.96% | 0.77% | 17056 |

CodeBERT before fallback: +0.00084705 − forgone positive gain 0.01558429 + avoided loss 0.01455946 = after fallback -0.00017777. The identity error was 1.65e-18.

CodeBERT stage contrasts, paired descriptive 95% intervals:
- RIDGE_gain_minus_NoRewrite：+0.00007864，[-0.002245257024612571, 0.0024101501576805177]。
- RIDGE_verified_minus_RIDGE_gain：+0.00076841，[-0.00013923283332923714, 0.001697267782023984]。
- RIDGE_risk_minus_RIDGE_verified：-0.00102482，[-0.0033672545992514858, 0.0013354596622335685]。
- RIDGE_risk_minus_NoRewrite：-0.00017777，[-0.0005510607141544759, 0.00019457275126158517]。
- RIDGE_gain_minus_HGB_gain_control：+0.00295474，[0.00029360745847181486, 0.005618695994944198]。

| BM25 | ENTR Candidate security oracle | +0.01941318 | — | — | 100% | 0 |
| BM25 | ENTR_top1 | -0.01443510 | 7955/2929 | 36.82% | 39.10% | 0 |
| BM25 | HGB_gain_control | -0.00271496 | 2615/853 | 32.62% | 11.80% | 0 |
| BM25 | RIDGE_gain | -0.00207943 | 3559/1072 | 30.12% | 5.63% | 0 |
| BM25 | RIDGE_verified | -0.00171681 | 2926/906 | 30.96% | 4.43% | 17056 |
| BM25 | RIDGE_risk | +0.00000000 | 0/0 | Undefined | 0.00% | 17056 |

BM25 before fallback: -0.00171681 − forgone positive gain 0.00085910 + avoided loss 0.00257590 = after fallback +0.00000000. The identity error was 4.34e-19.

BM25 stage contrasts, paired descriptive 95% intervals:
- RIDGE_gain_minus_NoRewrite：-0.00207943，[-0.002771194646960858, -0.001403062954189902]。
- RIDGE_verified_minus_RIDGE_gain：+0.00036263，[-0.00017839013976350904, 0.0009198235114857065]。
- RIDGE_risk_minus_RIDGE_verified：+0.00171681，[0.0011538519600744273, 0.0022813407795633494]。
- RIDGE_risk_minus_NoRewrite：+0.00000000，[0.0, 0.0]。
- RIDGE_gain_minus_HGB_gain_control：+0.00063553，[-0.0006226355950279119, 0.0018789390581726744]。

Positive-gain retention uses the total available positive gain from the same-query safe oracle as its denominator, avoiding division by near-zero net gain. Verification can change the selected action and is reported separately. Risk accounting holds the verified actions fixed.

The interval for harm among adopted actions uses a linked-group bootstrap. Within-group dependence prevents treating the query-level outcomes as independent Bernoulli trials for risk certification. `risk_certified=false`. The recorded deployment decision remains NoRewrite.

See `INFERENCE_LEDGER.json` and `metrics/*_full_H.json` for CE input tokens, distinct physical pairs and sunk costs by policy.

Independent checks verified 119420 reloaded model decisions and 249102 Fraction RR calculations. The largest numerical discrepancy was 2.22e-16.

The clean-directory replay reconstructed the main tables from cached rankings and decisions. The pre-opening audit separately checked T5 full-vocabulary FP64 calculations, retrieved ranks, CE and model inference.

See `STEP1_POLICY_FREEZE.md` for the frozen policy, `STEP2_PRECISION.md` for precision, and `STEP3_STAGE_RESULTS.md` for the stage breakdown.
