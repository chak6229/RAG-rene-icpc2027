# Data, models, source access and rights

## Included evidence

The replication ZIP includes the English supplementary tables S1–S25, the paired-weight confirmation, source protocols and reports, anonymous per-query ranks and outcomes for 14 replay units, and the authorized human-review labels, rationales, numerical panels and sampling design.

The [English submission release](https://github.com/chak6229/RAG-rene-icpc2027/releases/tag/rene-review-20261125-english) also provides seven archives of frozen candidate records, ordered Top100 rankings and scores, matrices, vectors and execution outputs. Benchmark text is represented by SHA-256 identifiers. Obtain the original corpora and relevance judgments through the upstream routes below. The [experiment data guide](https://github.com/chak6229/RAG-rene-icpc2027/releases/download/rene-review-20261125-english/EXPERIMENT_RUN_DATA_README.md) maps the archives to the research questions. The [per-file manifest](https://github.com/chak6229/RAG-rene-icpc2027/releases/download/rene-review-20261125-english/RUN_DATA_REVIEW_MANIFEST.json) records file identities and text-field transformations.

The same release provides the W1/W3/W5 study weights. See [model acquisition and installation](MODEL_ASSETS.md) and the [weight manifest](assets/STUDY_ASSETS.json) for hashes and installation paths.

## Upstream artifacts

| Input | Upstream access | Identity in this package |
|---|---|---|
| SSQR | [SSQR source](https://github.com/RedSmallPanda/SSQR); [implementation notes](supporting/implementation_fidelity.md); exact captured identities in `../provenance/UPSTREAM_SNAPSHOTS.json` | Original author checkpoint, original training CSV and production aggregation unavailable; reconstruction source included |
| CoSQA / CoCLR | [CoCLR](https://github.com/Jun-jie-Huang/CoCLR), [CodeXGLUE CoSQA](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery/CoSQA) | Source/overlap audit and protocol hashes; derived ranks/labels included; upstream C-UDA terms retained |
| CodeSearchNet / CodeXGLUE AdvTest | [CodeSearchNet](https://github.com/github/CodeSearchNet), [CodeXGLUE code search Adv](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-Adv) | Source bridge, role/sample and reconstruction protocols; raw corpus/qrels obtained upstream |
| StaQC | [StaQC](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset) | Single-code-answer query setup, source and grouping receipts, new-group/paired-weight settings; CC BY 4.0 notice retained |
| CODEnn / DeepCS Keras inputs | [Original acquisition README](https://github.com/guxd/deep-code-search/blob/master/keras/README.md) and its real-data download links | Exact raw and processed hashes, first-1M/next-10K mapping and commands in [CODEnn reconstruction](CODENN_RECONSTRUCTION.md) |
| T5 base | [google-t5/t5-base](https://huggingface.co/google-t5/t5-base) | Exact resolved file identities in S19; no weight files in this package |
| Local CQC T5 | Original training/data preparation scripts, S17/S19 and local training-config receipts | Selected update-92,000 checkpoint in W1 release attachment; full training 93,750 updates; exact per-file identities and installer supplied |
| CodeBERT | [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base), source training contract | Exact study-specific checkpoint/hash and pooler/unnormalized-dot settings in S19; fine-tuned weights in W3 release attachment, with exact hash and installer |
| UniXcoder | [microsoft/unixcoder-base](https://huggingface.co/microsoft/unixcoder-base), [official code](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) | Pinned official encoder-only configuration and exact weight hashes/settings |
| Cross-encoder verifier | `provenance/settings_sources/R27/` and R28 source/training-status receipt | Actual archived training source/settings and hashes; trained weights in W5 release attachment, with exact hash and installer |
| human review | `../human_review/data/` | A/B labels, rationales, sample/numbered panels and fingerprints; both authorized sharing |

These are acquisition links, not claims that current upstream `main` reproduces a historical snapshot. Follow the exact recorded revision and file hash where available. A receipt is retained as an original source reference even when the referenced large/raw file is not distributed. Do not interpret a historical workspace path as a present repository path.

## Licenses and access

Retain the supplied third-party notices in `licenses/`. CoSQA C-UDA allows computational use and distinguishes derived results from source data; redistributing source data requires attribution and binding downstream recipients to its terms. StaQC/CodeSearchNet/code components retain their own attribution and license conditions. Human-review sharing authorization concerns the de-identified labels and reasons, not a transfer of rights in source code/query stimuli.

`human_review/data/stimulus_fingerprints.jsonl` and the relevance panel code digests support exact matching after lawful upstream acquisition. The original stimulus text itself is not reconstructed by the portable replay. Source IDs are replaced consistently by repository-local opaque IDs; no identity or review process-time record is included.

The authors' original code/results are provided for confidential review under `../REVIEW_ONLY_NOTICE.md`; this package does not grant a public open-source license. Refer to `ENVIRONMENT.md` before rerunning model stages. Install the W1/W3/W5 attachments to resolve study-specific weights, then obtain the external corpora/qrels under their terms. Until these inputs are present, use the self-contained numerical/label replay. Asset installation and statistical replay do not constitute end-to-end retraining.
