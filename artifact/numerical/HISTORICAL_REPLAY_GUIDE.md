# Confidential review replay package

This historical guide describes the standalone numerical replay package. For the integrated English review edition, use [the current numerical guide](README.md).

The numerical package was prepared for designated editors, reviewers, and artifact evaluators. Distribute it through the venue’s confidential supplementary-materials channel and follow the venue’s anonymity and confidentiality requirements. The manuscript is submitted separately. The standalone numerical package contains no byline, affiliations, contact details, or editable manuscript.

## Running the replay

After extraction, run the following from this directory using 64-bit Python 3.9–3.12:

```sh
python -m pip install -r requirements.txt
python -I reproduce.py
```

Installing dependencies normally requires internet access. The replay itself runs on CPU without a GPU, model downloads, an account, or a network connection. A successful full run sets `passed` and `full_bootstrap` to `true` in `reproduced/ACCEPTANCE.json`. Recomputed values are written to `RESULTS.csv` in that directory. `--quick` skips bootstrap and serves only as a quick check.

## What can be verified

The standalone package covers 14 study units and 103 comparisons: the primary CoSQA confirmation, development-set B/G contrasts, StaQC confirmation, a third retriever, historical matched comparisons and a new 25,600-group confirmation across two T5 weight sets, and gains, harm and verification workload over complete H. Grouping, pairing, comparison families, seeds and resampling counts follow their corresponding studies. The English numerical README maps tables to replay coverage.

The core replay recomputes RR, exact random expectations, and statistical intervals from target ranks at anonymous candidate positions and frozen selection orders. Later execution-stage quantities are recomputed from anonymous numerical records. This standalone numerical subpackage does not contain original queries, code text, benchmark IDs, qrels, weights, training code, full reviewer returns, internal logs or future plans.

The replay verifies cached quantities. Regenerating candidates or rankings, retraining models, and recovering the original production pipeline are separate tasks. T2–T4, T6–T7, and T12–T15 in `reference_tables/` are supplied for inspection in the standalone package, and its human-review data are aggregate statistics only. The two-weight contrast remains unresolved; the evidence does not establish equivalence.

## Confidentiality

The files are authorized for confidential review. See [the review-only notice](../../REVIEW_ONLY_NOTICE.md) for distribution terms. Confidential distribution and retaining submission records can reduce misuse risk. File hashes verify content identity; they do not independently verify timestamps or authorship.
