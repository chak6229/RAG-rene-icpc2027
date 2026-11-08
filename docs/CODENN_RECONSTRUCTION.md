# CODEnn inputs and preprocessing reconstruction

The local CQC training used the DeepCS Keras data source. The upstream [Keras README](https://github.com/guxd/deep-code-search/blob/master/keras/README.md) links to the [Google Drive data folder](https://drive.google.com/drive/folders/1GZYLT_lzhlVczXjD6dgwVUvDDPHMB6L7?usp=sharing) and a [Baidu alternative](https://pan.baidu.com/s/1U_MtFXqq0C-Qh8WUFAWGvg). Obtain the real dataset, rather than the small dummy data in the source repository.

The required files are `train.desc.h5` and `vocab.desc.pkl`. Raw and decoded comments are not redistributed in this small artifact. Their exact identities are in [CODENN_INPUTS.json](assets/CODENN_INPUTS.json). Upstream links identify acquisition routes; compare downloaded hashes before treating a file as the historical input.

Place the raw files in `external_inputs/codenn/` and run from the artifact root:

```sh
python tools/verify_codenn_inputs.py --raw-dir external_inputs/codenn
python prepare_codenn_cqc.py --h5 external_inputs/codenn/train.desc.h5 --vocab external_inputs/codenn/vocab.desc.pkl --output-dir data/codenn_cqc --train-samples 1000000 --validation-samples 10000 --seed 101
python tools/verify_codenn_inputs.py --raw-dir external_inputs/codenn --prepared-dir data/codenn_cqc
```

Preprocessing uses PyTables, which belongs to the model-data environment and is not included in the NumPy/SciPy-only statistical replay dependencies.

| Delivered or acquired path | Historical path | Role |
|---|---|---|
| `external_inputs/codenn/train.desc.h5` | `imports/codenn_keras/keras/train.desc.h5` | Natural-language comments decoded in source order |
| `external_inputs/codenn/vocab.desc.pkl` | `imports/codenn_keras/keras/vocab.desc.pkl` | Token-to-ID vocabulary |
| `data/codenn_cqc/train.txt` | Same relative path | First 1,000,000 decodable comments |
| `data/codenn_cqc/valid.txt` | Same relative path | Following 10,000 comments |
| `data/codenn_cqc/valid_fixed_cqc.jsonl` | Same relative path | Fixed contiguous-span masking validation set |

Training uses source indices 0–999999 and validation uses 1000000–1009999. No record was discarded while reading these first 1,010,000 rows. The complete rules and original preprocessing manifest are in [CODENN_PREPROCESSING_MANIFEST.json](assets/CODENN_PREPROCESSING_MANIFEST.json). Checks cover both raw files and all three prepared outputs.

These inputs reconstruct this study's independent training data, not the unavailable original SSQR training CSV. S17/S19 preserve W1's training settings, completed 93,750 updates and selection of the update-92,000 checkpoint.
