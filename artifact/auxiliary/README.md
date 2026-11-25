# Additional anonymous per-query evidence

These compressed JSONL files retain the numeric per-query outcomes, predictions and fixed decisions for the Fixed20 adoption, RRF fusion and eight-arm limited-training studies. Query/group references are consistent across these auxiliary files, but form their own namespace separate from the older core replay and Manual evaluation exports. Source text and original identifiers are omitted. `INDEX.json` records row counts and source hashes.

They supplement S3/S4/S15 and their analysis source. This archive does not rerun OOF model fitting, list fusion against the raw corpus, or training. Extract with Python's `gzip`/`json` standard-library modules. The complete portable numerical acceptance remains the 14-unit/103-comparison scope listed in the repository root.
