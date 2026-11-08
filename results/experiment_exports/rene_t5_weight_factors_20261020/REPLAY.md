# Replay

Use the workspace `.venv-models` Python runtime. The analysis consumes existing candidate JSONL, candidate RR, per-query anchors, persisted Top100 rankings and qrels listed in DESIGN.json. No model weights or accelerator are loaded. The matched samples have already been consumed.

Original workflow (new output directory, only once):

1. Run the contract suite `python -m unittest experiments.rene_t5_weight_factors_20261020.test_contract` and save TEST_RESULTS.json.
2. Run `python -m experiments.rene_t5_weight_factors_20261020.run freeze`.
3. Run `python -m experiments.rene_t5_weight_factors_20261020.run evaluate`.
4. Run `python -m experiments.rene_t5_weight_factors_20261020.audit`.
5. Run `python -m experiments.rene_t5_weight_factors_20261020.deliver`.

The scientific pipeline refuses to overwrite sealed files. For an independent replay of a completed result, copy the workspace or archive the new experiment directory first; preserve bound inputs and the original scientific files. The audit independently reconstructs RR, factorial fields, pairs, all summary means, and primary intervals. Delivery code was added after the scientific design was sealed and never changes its estimands or statistics.

No additional generation, retrieval, model spot check, or historical full-training attribution is part of this cache-only task. Existing evidence completion provides the prior retrieval/model acceptance chain. FINAL_ACCEPTANCE.json and DELIVERY_ACCEPTANCE.json describe the new scientific and export checks separately.
