# Environment and settings

## Portable evidence replay

The complete repository replay is checked with Python 3.12 (64-bit), NumPy 1.26.4, SciPy 1.13.1 and threadpoolctl 3.6.0, on CPU. Exact installed versions and measured elapsed time are recorded in `../provenance/validation/RELEASE_VALIDATION.json` after release validation. The replay runs on anonymous numeric inputs and labels; it does not need the historical GPU environment.

## Historical model experiments

Historical run records: Windows build 26200, Python 3.10.11, PyTorch 2.7.0+cu128 / CUDA runtime 12.8, Transformers 4.57.6, Tokenizers 0.22.2, SentencePiece 0.2.2, Safetensors 0.8.0, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, Joblib 1.6.0, threadpoolctl 3.6.0, Matplotlib 3.10.9.

GPU: NVIDIA GeForce RTX 5070 Laptop, recorded memory 8,151 MiB, driver 577.05. CPU/RAM were checked later: AMD Ryzen 9 8945HX, 16 cores / 32 logical processors, 32 GiB. That hardware recheck is distinguished from the run-time software/GPU receipts.

Exact per-study settings and weight file identities:

- [S16–S25 settings](settings/experiment_environment_settings.md).
- [Machine-readable settings](settings/experiment_environment_settings.json).
- [Source receipts R01–R35](../provenance/SETTINGS_SOURCE_INDEX.json).

S16 environment; S17 training; S18 generation/scoring/retrieval; S19 weights/hashes; S20 six generation configurations; S21 eight limited-training configurations; S22 adoption/fusion; S23 H workflow; S24 statistical families and samples; S25 human review and supplemental judging.

Settings remain specific to each experiment. The replay pins intentionally differ from the historical model environment. The historical version inventory is not claimed to be a complete cross-platform GPU lockfile. Source receipts retain original file hashes; delivery hashes record any normalization of machine-local paths. A historical Word reference is not delivered because the paper is excluded; the human review machine inputs and protocols are included separately.

Optional source preparation also imports PyTables (`tables`) for CODEnn input files. Its exact historical version is not established by the captured core run inventory; it is not required for the portable replay. Other model-stage dependencies follow the archived imports and study-specific environment receipts.
