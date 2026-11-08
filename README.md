# SSQR external replication artifact

English review materials for **Position Selection Advantage and Executable Utility: An External Replication of SSQR for Code Search**, aligned with the cumulative English manuscript dated 8 November 2026. Release version **2026.11.08**. The manuscript is delivered separately.

Repository: https://github.com/chak6229/RAG-rene-icpc2027

[Download the artifact ZIP and the W1/W3/W5 model attachments](https://github.com/chak6229/RAG-rene-icpc2027/releases/tag/rene-review-20261108).

The package supports the paper's documented reconstruction, same-pool position-selection comparisons, automatic-execution analyses and independent human-review diagnostics. This release reorganizes existing evidence; it introduces no new experimental results.

## Start here

1. Read the [supplementary tables S1–S25](supplementary/SSQR_Supplementary_S1-S25_EN.pdf), also available as [Word](supplementary/SSQR_Supplementary_S1-S25_EN.docx) and [Markdown](supplementary/SSQR_Supplementary_S1-S25_EN.md).
2. Use the [appendix index](docs/APPENDIX_INDEX.md) to open the reports named in the paper.
3. Follow [installation and replay](INSTALL.md) and [expected results](EXPECTED_RESULTS.md).
4. Consult [environment and settings](docs/ENVIRONMENT.md), [data access](docs/DATA_AVAILABILITY.md), [model attachments](docs/MODEL_ASSETS.md) and [CODEnn reconstruction](docs/CODENN_RECONSTRUCTION.md).
5. Read the [human-review guide](human_review/README.md), [ethics and consent statement](docs/ETHICS_AND_CONSENT.md), and [current status and historical records](docs/CURRENT_STATUS.md).

## Reproduce the statistical evidence

Use 64-bit Python 3.12 and the pinned dependencies in `requirements.txt`.

```sh
python -m venv .venv
# Activate the environment using the platform instructions in INSTALL.md.
python -m pip install -r requirements.txt
python tools/reproduce_all.py
```

The unified entry point verifies integrity and replays **14 numerical study units and 103 statistical comparisons with full bootstrap**. It also recomputes human-review statistics, **6,144 exact query-level bounds** and **48 sampling-envelope combinations**. It uses included ranks, outcomes and labels without downloading models. Generated outputs go to `reproduced/`.

## Contents and scope

| Path | Contents |
|---|---|
| `supplementary/` | English S1–S25, additional S3/S4 comparison details, exact table data and source mapping |
| `docs/supporting/` | English reports referenced by the manuscript |
| `artifact/numerical/` | Anonymous target ranks, selection orders, policy outcomes and statistical replay |
| `human_review/` | Original independent A/B labels and rationales, supplemental judgments, sampling inputs and replay |
| `results/` | The 176 integrated S1–S15 result records, paired-weight confirmation and frozen experiment exports |
| `experiments/` and root scripts | Historical model, training, data-preparation and analysis implementations |
| `provenance/` | Source manifests, historical records and release validation |
| `tools/` | Bundle checks, replay and installation of model attachments |

All current reading guides and supplementary documents are in English. Original annotations and frozen historical records retain their source language and bytes; they are evidence, not newly translated reviewer judgments. The current English reports provide the interpretation and navigation needed for review.

The portable replay begins at cached ranks/outcomes and does not regenerate raw query text or all retrieval lists. Model stages require the three separately hashed W1/W3/W5 attachments and lawful upstream data acquisition. Full retraining, arbitrary new-sample generation and online-latency validation are outside the checked replay. Table-specific coverage is stated in `EXPECTED_RESULTS.md`.

## Access and integrity

This private repository requires access for the intended reviewers. The current release record is [RELEASE_STATUS.json](provenance/RELEASE_STATUS.json). Run `python tools/verify_bundle.py` to check the delivered files against `SHA256SUMS.json`. Keep [the review notice](REVIEW_ONLY_NOTICE.md) and third-party notices in `docs/licenses/` with the package.
