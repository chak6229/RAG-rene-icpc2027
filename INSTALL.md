# Install and reproduce

## Supported and checked environment

The complete release validation uses 64-bit Python 3.12 and the pins in `requirements.txt` (NumPy 1.26.4, SciPy 1.13.1, threadpoolctl 3.6.0). CPU replay requires no GPU, model, account or API key. Reserve 1 GB RAM and 300 MB free storage for installation and outputs. Installing dependencies normally needs network access; the replay itself uses local files only. A recorded runtime receipt is supplied in `provenance/validation/` after release validation; actual time depends on the machine.

## Windows PowerShell

Run in the extracted repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/reproduce_all.py
```

Calling the virtual environment's interpreter directly avoids changing PowerShell execution policy.

## Linux or macOS

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/reproduce_all.py
```

Linux/macOS commands are provided for the portable Python code; the release validation receipt identifies the operating system actually tested.

## Separate entry points

```sh
python tools/verify_bundle.py
python -I artifact/numerical/reproduce.py --output reproduced/numerical
python human_review/reproduce.py --output reproduced/human_review
python human_review/verify_human_reviews.py
```

`reproduced/numerical/ACCEPTANCE.json` must report `passed: true`, `full_bootstrap: true`, `studies: 14`, `contrasts: 103`. `reproduced/human_review/ACCEPTANCE.json` must report `passed: true`, `recomputed_exact_fraction_bounds: 6144` and `recomputed_sampling_envelopes: 48`.

`--quick` on the numerical script skips bootstrap and is only a preliminary check. The unified entry point always runs full bootstrap. Repeated runs replace generated files only; source inputs remain sealed. Output folders must be outside `human_review/` and outside sealed numerical inputs.

## Model pipelines

Do not install the CPU replay pins as the historical GPU environment. See `docs/ENVIRONMENT.md` and the exact settings. For model stages, first download and install the separately hashed W1/W3/W5 release attachments using `tools/install_study_assets.py`. Follow `docs/MODEL_ASSETS.md` and `docs/CODENN_RECONSTRUCTION.md` for the exact historical-to-delivery input mapping. Other datasets/qrels and public baseline weights remain upstream inputs. They are source archives rather than a second one-command entry point. The portable commands above reproduce statistics from the included numeric evidence.

## Troubleshooting

- File/hash failure: extract a fresh ZIP; check that files were not edited or autoformatted. `.gitattributes` preserves line endings.
- `ModuleNotFoundError`: use the same virtual environment interpreter for installation and execution.
- Bootstrap/statistic failure: inspect the generated acceptance/report files; use the pinned dependencies and report the exact file/hash and interpreter version.
