# Study model attachments and installation

The three study-specific model archives are supplied separately from the small source repository. Their binary identities are unchanged from the experiments. W1/W3 retain their original packaging date. W5 is repackaged with a separate verification-model directory in release 2026.11.10; its weight and tokenizer contents are unchanged. Exact sizes, SHA-256 values and per-file identities are in [STUDY_ASSETS.json](assets/STUDY_ASSETS.json).

| Attachment | Contents | Installed paths |
|---|---|---|
| SSQR_RENE_W1_CQC_T5_20261031.zip | Selected update-92,000 CQC T5 weights, configuration, tokenizer and SentencePiece inputs | `models/t5-cqc-codenn-1m/best/` and required `models/t5-base/` tokenizer input |
| SSQR_RENE_W3_CodeBERT_20261031.zip | Final two-epoch CodeBERT weights, tokenizer and representation contract | `models/codebert-ssqr-pooler-2epoch-20261019/` |
| SSQR_RENE_W5_CrossEncoder_20261110.zip | Frozen large_dense_4096 verifier and its base configuration/tokenizer | `models/ssqr-verifier/ce_large_dense_4096.pt` and `models/codebert-base/` |

Download W1/W3 from the [original model release](https://github.com/chak6229/RAG-rene-icpc2027/releases/tag/rene-review-20261108) and W5 from the [10 November model release](https://github.com/chak6229/RAG-rene-icpc2027/releases/tag/rene-review-20261110-paper). Place all three archives in a directory named `release-assets` next to the extracted artifact folder. From the artifact root, run:

```sh
python tools/install_study_assets.py --asset-dir ../release-assets --check-only
python tools/install_study_assets.py --asset-dir ../release-assets
```

The first command checks archive and internal file hashes without loading models. The second installs files at the documented relative paths and stops if an existing file has a different hash. The installer rejects paths outside the destination root.

The archives contain the weights actually used, without optimizer states or unused intermediate checkpoints. W1 includes the SentencePiece file read from the base T5 path by the historical pipeline. This adds a required input without changing the trained weights. These identities correspond to W1, W3 and W5 in S19.

Model installation resolves model paths, not the complete experiment inputs. Obtain frozen query/corpus inputs and upstream data under the terms in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). The historical model environment differs from the portable CPU replay environment. Identity checks, installation and cached statistical replay do not constitute full retraining or end-to-end reconstruction of all original ranks.

W1 derives from [T5-base](https://huggingface.co/google-t5/t5-base); W3 and W5 derive from [CodeBERT-base](https://huggingface.co/microsoft/codebert-base). Retain their model cards and license terms. The release identities are recorded in [the publication status](../provenance/RELEASE_STATUS.json); a private repository requires separately granted reviewer access.
