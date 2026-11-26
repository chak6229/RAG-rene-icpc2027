# Artifact scope

This release supports the 25 November 2026 English manuscript for the frozen study cited as version 2026.11.10. The studies for the three research questions cover SSQR reconstruction and fidelity checks, selected-count/aggregation analyses, tested retriever and T5-weight conditions, fixed-action adoption, fusion, complete-H execution, limited training sensitivity, and the original and supplemental human-review diagnostics.

Table S7 is a descriptive accounting of the three frozen Ridge stages, using the same complete-H query records as Table S5. All other result-table rows, cached numerical inputs, human judgments and checkpoint tensors retain their original values. The selected verification-model training functions are supplied separately in experiments/rene_verifier_training/; their original function bodies are preserved.

Standalone earlier multi-seed experiments, probabilistic ranking pilots, old model registries, unused confirmation/policy-development branches and obsolete integration tools are excluded. Exact original copies remain in the author's local archive. Earlier consumed-sample identities, exclusions, historical protocol hashes and model-training provenance remain traceable where they affect the current study. Historical paths in these records identify past inputs; they are not runnable additional research modules. A normal source-control update does not erase earlier Git commits.

Use tools/reproduce_all.py for the included numerical and human-review replay, including tools/rebuild_s7.py --check for S7. Model-stage source inputs retain the limitations stated in EXPECTED_RESULTS.md. The main manuscript and its LaTeX source are supplied separately.
