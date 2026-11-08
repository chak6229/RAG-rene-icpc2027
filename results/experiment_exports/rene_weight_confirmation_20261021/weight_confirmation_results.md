# Prospective paired T5-weight confirmation of B/G

A separately frozen study uses 25,600 previously unused connected StaQC components and a fixed 32,768-snippet corpus. Both T5 weight conditions are paired on queries, targets, corpus and the fixed CodeBERT retriever, with their own complete insertion-position pools. No retraining or parameter search is performed. The old 512-query analyses and their original 24-comparison family remain unchanged.

The six prespecified contrasts comprise ENTR B/G under each weight and paired base-minus-local B/G. Two-sided Bonferroni family alpha is .05. A direction requires both the adjusted t and 20,000-draw paired component-bootstrap intervals to exclude zero on the same side, with at least 20 nonzero components. PROB and the complete curves are descriptive. The .005 planning resolution and difference-precision target do not define equivalence margins.

|Contrast|Estimate|Adjusted t interval|Adjusted bootstrap interval|Decision|
|---|---:|---|---|---|
|local::ENTR_B|+0.002803|[+0.001565, +0.004041]|[+0.001576, +0.004024]|positive|
|local::ENTR_G|+0.003241|[+0.002409, +0.004073]|[+0.002422, +0.004091]|positive|
|t5base::ENTR_B|+0.004282|[+0.002941, +0.005624]|[+0.002946, +0.005626]|positive|
|t5base::ENTR_G|+0.003520|[+0.002636, +0.004404]|[+0.002636, +0.004427]|positive|
|base_minus_local::ENTR_B|+0.001479|[-0.000325, +0.003282]|[-0.000336, +0.003267]|unresolved|
|base_minus_local::ENTR_G|+0.000279|[-0.000909, +0.001467]|[-0.000942, +0.001477]|unresolved|

All four within-weight ENTR B/G components meet the prespecified positive-direction rule, confirming the direction in both tested T5 weight conditions on this new StaQC setting.

The paired differences compare complete weight conditions, not an isolated effect of CQC training. Unresolved differences do not establish equivalence. The scope is the specified StaQC title/snippet setting, fixed CodeBERT and two weights within T5-base; it does not establish cross-family universality, automatic-policy benefit or historical training attribution.

Sampling excludes all prior StaQC query and corpus components, archived partial generations, normalized exposed text and inherited training/reference overlaps. The outcome-blind precision plan used consumed development data only; the four-direction joint power lower bound at .005 is 0.9652, conditional on variance transport. The protocol was locally sealed before new generation and effects; it was not externally preregistered. Source mappings, complete candidate/action/ranking coverage, exact-random arithmetic, paired intervals and representative model/ranking reloads passed the specified audits.
