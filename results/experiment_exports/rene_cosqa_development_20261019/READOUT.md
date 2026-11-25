# Development Extension Interpretation

This page is broken down by gain-and-loss algorithms after the result has been opened, with no new model, threshold or method selection. Four formal RQ1 comparisons and presets of the main revenue model remain unchanged.

The extension of 1,490 / 512 associated groups, the five-sum benefit option and independent recovery have been completed. Confirms that the pool 8,530/1,963 group is not yet operational.

## What does the current result support?

- In the same ENTR three candidate pools, the main yield model improves relative to original entropy Top1 CodeBERT +.01242189, BM25 +.01040205 (development OOF); The selection of task-related oversight reduces some of the original ranking losses.
- Positive net gain over the original query was not established. With the zero-gain NoRewrite action, the main model gives +.00020512 / −.00499483. The linear control gives +.00409250 / −.00248255. The intervals for the positive CodeBERT point estimates span zero.
- All four candidate-quality intervals for ENTR versus RAND and PROB span zero. The BM25 ENTR−RAND point estimate is negative. The results leave general ineffectiveness and equivalence unestablished.

## Where exactly did the loss occur?

This round does not have a back-to-back risk door.

| retriever | There are better candidate queries in the pool | All three candidates are worse. | The main model gets positive returns/ queries | Main Model Loss/ Query | Net proceeds from the main model |
|---|---:|---:|---:|---:|---:|
| CodeBERT | 528/1490 | 125/1490 | +0.02018953 | −0.01998441 | +0.00020512 |
| BM25 | 128/1490 | 319/1490 | +0.00114914 | −0.00614396 | -0.00499483 |

CodeBERT The main model captures approximately 29.6% candidates with positive returns, but the resulting losses almost fully offset the gains; BM25 The catch margin 7.8% indicates that the searcher has both fewer candidates and selective identification issues. The available proceeds here use labels, only the diagnostic ceiling.

## What do you expect to lose?

| retriever | Average projection after selection of the main model | Actual | High forecast. |
|---|---:|---:|---:|
| CodeBERT | +0.01276952 | +0.00020512 | +0.01256440 |
| BM25 | +0.00316669 | -0.00499483 | +0.00816152 |

The overall candidate average is close to the real average and does not mean that it will be calibrated after taking the maximum value and re-screening the projected positive returns. The above table is an observed post-selection prediction deviation that cannot be attributed to a statistical mechanism alone.

The CodeBERT main model's net gains for the original pilot subset and the additional development subset under the same five-fold OOF procedure are +.01555543 / −.00297939. The linear control gives +.00371982 on the additional subset, with an interval spanning zero. The full development estimate of +.00020512 does not establish a stable positive gain.

## Basis for further decision-making

Retain predefined master models and linear comparisons; The single recognition policy will be frozen first by evidence of and calibration of mission benefits after the existing development of a cache test option. There is no basis for resolving the wrong selection of this round by simply easing the risk threshold.

The effectiveness gap is still: CodeBERT ENTR − RAND It's...005  /  80% / Four relative conservative planning needs 2,475 Group, confirm the pool only 1,963 Groups; The working assumption for known group size differences is 3,662 group sensitive. The remaining three meet the original variable number of events. Prior to confirmation, the precision target and the joint comparison family need to be clearly identified and maintained.005 Objectives are not changed to a broader threshold based on current results.

Full values, compartments, audits and freezing of evidence are available in REPORT.md, RQ1_DEVELOPMENT.json, RQ2_DEVELOPMENT.json, POWER_PLAN.json. The old model is retained with the official pointer, risk_certified = false and deployment remains NoRewrite.
