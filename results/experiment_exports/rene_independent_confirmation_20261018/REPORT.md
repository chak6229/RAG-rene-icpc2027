# RENE independent confirmation

This is a one-time descriptive confirmation of the frozen weak46 UQ candidate-selection interface.
The CodeBERT retriever was trained on the source partition; this upstream exposure is disclosed and prevents a claim of fully unseen model generalization.

| method | delta MRR | coverage | positive | harmful | harmful rate |
|---|---:|---:|---:|---:|---:|
| no_uq | +0.000000 | 0.000 | 0 | 0 | NA |
| seed_uq | -0.002316 | 0.833 | 1 | 8 | 0.1 |

Oracle candidate ceiling: +0.095736; seed-101 best-candidate ceiling: +0.051607.
No risk certificate is issued. The result is not a replacement for the larger independent confirmation required for a publication claim.

## Validity status

This run is retained as a sealed diagnostic smoke test, not as a valid publication-level UQ ablation. The no-UQ branch zeroed the seed features of a model trained with those features, and the weak46 model was trained with eight T5 seeds while this run used four. See `VALIDITY_REVIEW.md` before interpreting the negative Delta-MRR.
