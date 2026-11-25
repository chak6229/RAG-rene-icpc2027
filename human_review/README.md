# Human review

Two independent reviewers, A and B, supplied the judgments. Both authorized sharing their de-identified labels and rationales; the research lead confirmed A's independent supplemental completion. The original category labels, unknowns and disagreements are preserved. The de-identified rationales are supplied in English translation.

## Files

| File | Contents |
|---|---|
| `data/original_annotations.jsonl` | 1,676 initial records: A/B × (128 clarity, 383 rewrites, 327 code panels), with English rationales |
| `data/supplement_annotations.jsonl` | 688 supplemental judgments: 344 per reviewer, with English rationales, with English rationales |
| `data/sampling_design.jsonl` | 128 queries, 120 groups, original eight strata, inclusion probabilities and weights |
| `data/relevance_pools.jsonl` | Anonymous target/panel identities, original and candidate Top3 numeric lists, fixed policy actions, inherited labels and code fingerprints |
| `data/query_event_flags.jsonl` | Frozen derived per-query event/error-type flags used for the weighted rates |
| `data/weighted_query_bounds.jsonl` | Reference 6,144 exact query-level bounds, recomputed from panels and labels |
| `data/stimulus_fingerprints.jsonl` | Query/rewrite digests and original candidate positions for matching externally acquired stimuli |
| `reference/` | Accepted agreement, weighted-rate, relevance-bound and sampling-uncertainty results |
| `EXPORT_VALIDATION.json` | Counts, sharing/source description, and label-preservation checks |
| `reproduce.py`, `analysis_functions.py`, `finite_frame.py` | Portable statistical replay; extracted original formulas |

## Run

From the repository root, after installing `requirements.txt`:

```sh
python human_review/reproduce.py --output reproduced/human_review
python human_review/verify_human_reviews.py
```

The first command recomputes agreement, 10 weighted events, 24 weighted error types, 6,144 exact relevance bounds, 48 summaries and their finite-frame sampling envelopes. It compares against the accepted results at tolerance 1e-11, with exact comparisons for rational bounds. Weighted event/type replay begins at the supplied derived flags; inspecting their semantic construction uses the archived source and externally acquired stimuli.

Opaque Q/G/I/P/D references are consistent within this folder. Original identities, source query/document IDs, reviewer-return attachments, stimulus text and process-time fields are omitted. Raw stimuli are obtained upstream under their terms; fingerprints permit matching. The numeric replay is self-contained and does not repeat annotation or regenerate query text.

The fixed sample and weighted sample identification ranges are distinct from finite-frame sampling envelopes. Neither is the complete H MRR@100 measurement or a population risk certificate. Reviewers and relevance thresholds are reported separately; there is no forced consensus or hidden recoding of unknowns.

## Ethics and informed consent

Both reviewers gave informed, voluntary consent after learning the study purpose and tasks, including consent to share de-identified annotations and rationales. The paper uses A/B identifiers and aggregate statistics; replication materials retain anonymous item-level judgments without reviewer identities. No formal ethics approval or exemption determination was sought.

English translations appear in the `note` fields only. Archived source files can be identified by the hashes in [edition provenance](../provenance/ENGLISH_EDITION.json). The English [results guide](RESULTS_AND_SCOPE.md) and [case analysis](../docs/supporting/review_disagreement_discussion.md) provide the current reading layer.
