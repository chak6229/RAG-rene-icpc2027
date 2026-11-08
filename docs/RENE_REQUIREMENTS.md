# ICPC 2027 RENE: requirement mapping

Official source, checked 27 October 2026: [ICPC 2027 Replications and Negative Results](https://conf.researchr.org/track/icpc-2027/icpc-2027-replications-and-negative-results--rene-).

The call specifies the IEEE conference template **for the paper**. It does not prescribe a separate GitHub directory template. This repository uses a conventional reproducibility layout adapted to the call; it is not an official RENE repository template or an artifact-badging decision.

| Official requirement relevant to the artifact | Material supplied here |
|---|---|
| Replications go beyond reimplementation/rerun and apply the approach to new datasets | CoSQA and StaQC conditions, study/sample roles, paired-weight and third-retriever outputs; `EXPERIMENT_MAP.csv` |
| Clearly describe recovered results and non-replicable aspects | `DATA_AVAILABILITY.md`, `supporting/implementation_fidelity.md`, numerical scope in `EXPECTED_RESULTS.md` |
| Clearly identify artifacts used and provide artifact links in the submission | Upstream links, exact available hashes/settings, source/delivery indices and portable entry points |
| Artifact availability and methodological rigor are evaluation criteria | Sealed numerical inputs, protocols, correction families, 14/103 replay, human review records/statistics, checksums and release validation |

The call says RENE is not double-anonymous. A/B are still used to protect the identities of the human reviewers, with sharing authorization recorded in `human_review/EXPORT_VALIDATION.json` and `provenance/SOURCE_PROVENANCE.json`.

The official IEEE format and 10-page + 2 reference-page allowance concern the new-replication paper, which is intentionally excluded. Uploading this directory does not submit the paper. A private GitHub URL must actually be accessible to assigned reviewers; before submission, arrange access through the venue's permitted review channel. A future archival/public release and any artifact evaluation are separate actions.
