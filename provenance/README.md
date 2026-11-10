# Source and delivery records

- `SOURCE_PROVENANCE.json`: origin of the core numeric evidence and authorized 人工评审 export.
- `FILE_LINEAGE.json`: original versus delivered file SHA-256 values, including normalization of machine-local paths.
- `TABLE_SOURCE_INDEX.json`, `table_sources/`: accepted result JSON sources for S1–S15.
- `SETTINGS_SOURCE_INDEX.json`, `settings_sources/`: R01–R35 environment/configuration sources; the paper is intentionally omitted.
- `EXPERIMENT_FILE_INDEX.json`: historical result bundle → delivered protocol/result file; source type **人工评审** is used for human review.
- `UPSTREAM_SNAPSHOTS.json`: recorded upstream repository/commit identities and capture hashes.
- `TABLE_EXPORT_VALIDATION.json`: the 15 CSV tables were re-exported with latest source-pointer columns; original scientific result cells were unchanged.
- `INVENTORY.json`: delivered file inventory and GitHub file-size check.
- `validation/`: actual release validation receipts, including a full run from a ZIP extracted outside the repository tree.

Original `results/rene_...` and `models/...` paths in receipts describe source history, not missing portable-replay inputs. Large/raw inputs are separately identified in `docs/DATA_AVAILABILITY.md`. Reference numeric-subpackage acceptance from an earlier release retains its original environment scope; current complete-package checks are in `validation/RELEASE_VALIDATION.json`.

`SCOPE_CLEANUP_20261110.json` lists 26 author-side files archived outside the review bundle. FILE_LINEAGE records for these paths retain their previous identity and explicitly mark that they are no longer delivered. Scientific source receipts, experiment histories and human labels remain included.
