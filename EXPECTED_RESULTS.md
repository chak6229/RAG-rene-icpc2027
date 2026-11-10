# Expected results and verification scope

## Portable numerical replay

`artifact/numerical/STUDIES.json` contains the full original estimates, corrected interval settings, seeds, comparison families and expected decision rules. `reproduced/numerical/RESULTS.csv` contains fresh recomputation. Floating-point comparisons use absolute tolerance **1e-11**. Fractions in the human review relevance analysis are compared exactly.

The numerical entry point covers 14 units, 103 statistical comparisons and 99,904 anonymous records. All input hashes are checked before replay. It computes candidate RR@100 from target ranks, exact-random expectations, selector curves, B/G and paired differences, the specified CoSQA primary comparison, and full-H stage accounting. Counts refer to overlapping study records, not 99,904 unique queries.

Selected checks include CoSQA/CodeBERT development ENTR B = +0.01321969 and G = +0.01193674 (rounded), and the first StaQC confirmation B = +0.00325690 and G = +0.00360952. The 25,600-group paired confirmation supports each weight's ENTR B/G directions; paired weight differences remain unresolved. The full, unrounded values and original intervals are machine-readable in `STUDIES.json` and the result tables.

| Supplement label / historical key | Delivered evidence | Complete portable recomputation |
|---|---|---|
| S1 / T1 | CoSQA primary comparison | Yes, numerical replay |
| S2 / T2 | Original four-comparison contract analysis | Table and source; same candidate ranks also feed the later B/G replay; original four-comparison family not rerun here |
| S3 / T3 | Fixed20 adoption, four Ridge heads | Table, protocol, aggregate reports and source; not OOF model refitting |
| S4 / T4 | RRF3/RRF4 fusion | Table, protocol, aggregate reports and source; not raw list fusion/retrieval replay |
| S5 / T5 | Frozen H stages | Yes, cached outcomes and work accounting |
| S6 / T6 | Earlier two-weight Top1/best3 evidence | Table and source; complete later B/G weight comparison replayed as S11 |
| S7 / T7 | Full-H opportunity-loss and verification accounting | Yes, descriptive arithmetic from included frozen stage records; tools/rebuild_s7.py --check |
| S8 / T8 | Same-pool count/aggregation decomposition | Yes, original 24-comparison family |
| S9 / T9 | StaQC confirmation and descriptive controls | Yes |
| S10 / T10 | UniXcoder extension | Yes, original 12-comparison family |
| S11 / T11 | Complete matched two-weight B/G | Yes, original 24-comparison family |
| S12 / T12 | human review agreement | Yes, anonymous labels |
| S13 / T13 | human review weighted query events | Yes, from frozen derived per-query event flags and eight-stratum design |
| S14 / T14 | human review weighted error types | Yes, from frozen derived per-query flags and eight-stratum design |
| S15 / T15 | Eight-arm limited training sensitivity | Table, configuration, outputs and training/analysis source; not retraining |
| Separate paired-weight confirmation | New StaQC 25,600 groups | Yes, original six-comparison family |
| human review relevance sensitivity and sampling | Labels, numeric Top3 panels, sample design, reference output | Yes, 6,144 exact query bounds, 48 summaries, exact finite-frame envelopes and secondary t approximations |

`tools/check_tables.py` checks S1–S15 CSV exports against `results/TABLES.json` and confirms the expected 179 rows. This is export integrity, distinct from rerunning each source analysis.

## human review acceptance

Expected initial records: 1,676 = two reviewers × (128 query-clarity + 383 rewrite + 327 code judgments). Supplemental records: 688 = A/B × 344. The sample remains 128 queries / 120 associated groups from the eight-stratum H8530 frame.

Initial rewrite exact agreement is 0.7650130548302873, Cohen κ = 0.5251412040225927. Code agreement including unknown is 0.5382262996941896, κ = 0.37064889047503724. All original unknowns, rationales and disagreements are retained. The exporter checks equality of the label/rationale multisets against the prior accepted anonymous export.

The main RIDGE_verified, threshold-2 RR@3 sample identification ranges are approximately A [+0.005843, +0.018263] and B [+0.001279, +0.016604]. The separate finite-frame sampling envelopes include zero. The sample identification bounds, finite-frame uncertainty and full-H MRR@100 are separate quantities.

## Limits of these checks

This release does not regenerate candidate texts or audit cached ranks against downloaded corpora/qrels. Raw-text/model stages require the external inputs and source layouts documented separately. No new labels, effect experiments, risk certification or equivalence conclusions are created by packaging or numerical replay.
