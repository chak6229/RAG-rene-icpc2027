# Provenance and Meaning of the Frozen Risk Working Point

The full-H pipeline inherits the fixed probability threshold of 0.075 from the historical development protocol. During selected-action risk development, raw-tight and the additional risk heads applied the same rule: predicted harm probability must not exceed 0.075. The subsequent cleaned three-class anchor-risk pipeline retained this operating point. The full-H design sealed it again; H labels were not used for fitting, threshold selection or a threshold sweep.

This provenance establishes inheritance of a fixed operating point. It does not establish that 0.075 is optimal or derive a certified calibration formula from 0.1.

| Quantity | Role | Interpretation |
|---|---|---|
| threshold = 0.075 | Threshold on predicted harm probability for the proposed action | Exceeding it restores the complete original-query Top100 list without selecting another action; incurred verification work remains |
| target_harm = 0.1 | Historical target for the harm rate among adopted actions | Distinct from the prediction threshold; it does not certify population risk for full H |
| Observed harm rate or count | Empirical result under a particular sample and policy | Reported separately from positive-gain retention, net change and the adoption denominator |

Historical selected-action and three-class designs, component versions and the sealed full-H design sources are retained in `provenance/review_fixes/`. The full-H statuses `risk_certified=false` and `deployment_policy_promoted=false` remain unchanged. Stage results describe the gain–loss trade-off at this frozen operating point; they do not establish an optimal gate or a result for every threshold.
