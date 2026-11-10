# Confidential Reviewer Artifact

This artifact accompanies a submission about SSQR reconstruction, selected-count/aggregation effects (B/G), and executable policy utility. It is supplied to the assigned editors, reviewers and artifact evaluators for confidential evaluation.

## Run

Use 64-bit Python 3.9-3.12 on Windows, Linux or macOS. No GPU, model download, API key, account or network connection is used during replay. Installing the pinned dependencies normally requires a network connection.

```sh
python -m pip install -r requirements.txt
python -I reproduce.py
```

The second command verifies all input hashes, recomputes candidate RR@100, exact-random expectations, selector curves, B/G and paired differences, and replays the original t/bootstrap comparisons. It also recomputes the main CoSQA confirmation and frozen full-H stage accounting from cached numeric outcomes. The output directory is `reproduced/`; `ACCEPTANCE.json` must say `passed: true` and `full_bootstrap: true`. `RESULTS.csv` lists the recomputed estimates. Repeating the command replaces only those generated output files.

For a faster preliminary check:

```sh
python -I reproduce.py --quick
```

Quick mode omits all bootstrap intervals and their joint direction decisions. It is not the full validation.

## What Is Recomputed

| Evidence | Inputs | Recomputed scope |
|---|---|---|
| CoSQA primary comparison (T1) | 7,718 anonymous paired metric records / 1,963 groups | ENTR minus RAND20 best3, original group-t and bootstrap intervals, prespecified equivalence rule |
| Development B/G (T8) | Candidate target ranks and fixed selection orders; source and CoSQA; CodeBERT/BM25 | ENTR/PROB/random k=1/2/3 mean/best-k curves, all 24 original primary component comparisons |
| StaQC confirmation (T9) | 13,312 anonymous groups, two retrievers | Original two CodeBERT primary comparisons and descriptive controls |
| Third retriever (T10) | Paired UniXcoder/CodeBERT/BM25 numeric inputs | All 12 original corrected comparisons |
| Matched T5 sensitivity (T11) | 512 anonymous query/groups per domain, two weight conditions | Complete selector curves and all 24 original corrected comparisons |
| New T5-weight confirmation | 25,600 anonymous groups; two weight conditions | Complete selector curves, four within-weight ENTR B/G comparisons and two paired weight differences; original six-comparison family |
| Full-H execution stages (T5) | 8,530 anonymous metric/decision records per retriever | Delta intervals, adoption/harm/improvement counts, retained positive gain, losses, charged verification pairs/tokens |

The study IDs in `STUDIES.json` and replay output map these computations to the table names above. That file records the original statistical configuration and expected results. Source hashes and the source-to-export checks are recorded in `EXPORT_CHECKS.json`.

`reference_tables/` contains the previously integrated T1-T15 aggregate exports and the new weight-confirmation exports for inspection. The replay recomputes the scopes listed above. T2-T4, T6 and T12-T15 are reference-only here; it does not reconstruct their complete original analysis. Human-review summaries are aggregates only; original labels and notes are not included or re-created.

## Numerical Inputs And Boundaries

Each compressed JSONL file uses anonymous query and group integers. A candidate rank of 0 means no judged target within the first 100 retrieved items, not proof that every retrieved item is irrelevant. Positive ranks yield RR = 1/rank. Candidate positions, including repeated texts, retain their original probability mass. Exact random samples positions without replacement; k is clipped to pool size. Selection orders are cached decisions rather than freshly recomputed model scores.

Mean aggregation is expected RR when adopting one selected candidate uniformly. Best-k uses relevance labels and represents opportunity, not automatic selection. B/G measures changes in advantage against matched random. Execution-stage utility uses the original query as reference. These are different estimands and are not a causal decomposition of policy loss. Pairing, original group order, bootstrap seeds/repetitions and correction families are preserved. The comparison tolerance is 1e-11.

The artifact begins at cached target ranks or cached policy outcomes. It does not rerun T5 generation, neural retrieval, model fitting, human annotation or training. It cannot verify cached ranks against raw corpora/qrels because those are excluded. It does not restore the original author's checkpoint, original training CSV or production aggregation. The 1,024-update configuration study is not evidence about every difference in historical full training. Two T5 weight conditions are one model family; unresolved paired differences are not equivalence. No deployment risk certificate is claimed.

## Contents And Use

- `reproduce.py`, `requirements.txt`: portable numerical replay.
- `data/*.jsonl.gz`: de-identified numeric inputs, without query/code text or source IDs.
- `STUDIES.json`, `EXPORT_CHECKS.json`: statistical contracts, expected results and source hash lineage.
- `reference_tables/`: aggregate tables for inspection.
- `REVIEWER_ONLY_NOTICE.md`: permitted confidential evaluation use.
- `MANIFEST.json`: SHA-256 inventory.

The manuscript is submitted separately through the venue. Training code, model weights, full production implementation, raw benchmark text, qrels, individual review returns, reviewer identities, internal logs and future research plans are excluded. There is no telemetry or external callback.

Please retain and handle this material according to the venue's review confidentiality and anonymity policies. This is not a public release. The submitting authors retain rights in their original material, subject to applicable third-party rights and venue policy.

Current S7 is separately recomputed by ../../tools/rebuild_s7.py --check from the same frozen full-H stage records.
