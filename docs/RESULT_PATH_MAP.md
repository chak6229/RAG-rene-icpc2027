# Historical experiment references → delivered evidence

Original `results/rene_<study>_<date>/` paths identify historical input/output bundles. The clean repository delivers them at the export locations below; raw caches are external. Source/delivery hashes are indexed in `provenance/`.

| Historical bundle | Delivered result location | Source |
|---|---|---|
| `rene_cosqa_ssqr_audit_20261019` | [Reconstruction and source/overlap audit](../results/experiment_exports/rene_cosqa_ssqr_audit_20261019/) | Computational experiment |
| `rene_ssqr_alignment_20261019` | [T5/CodeBERT implementation alignment](../results/experiment_exports/rene_ssqr_alignment_20261019/) | Computational experiment |
| `rene_ssqr_bridge_20261019` | [Original-data bridge](../results/experiment_exports/rene_ssqr_bridge_20261019/) | Computational experiment |
| `rene_ssqr_aggregation_audit_20261019` | [Aggregation audit](../results/experiment_exports/rene_ssqr_aggregation_audit_20261019/) | Computational experiment |
| `rene_ssqr_rewriter_envelope_20261020` | [Generation profiles](../provenance/table_sources/) | Computational experiment |
| `rene_cosqa_development_20261019` | [CoSQA development](../results/experiment_exports/rene_cosqa_development_20261019/) | Computational experiment |
| `rene_cosqa_confirmation_20261019` | [CoSQA primary and full H](../results/experiment_exports/rene_cosqa_confirmation_20261019/) | Computational experiment |
| `rene_core_step_a_20261020` | [Top1 versus best3 contract](../results/experiment_exports/rene_core_step_a_20261020/) | Computational experiment |
| `rene_core_step_b_20261020` | [Fixed-action Ridge adoption](../results/experiment_exports/rene_core_step_b_20261020/) | Computational experiment |
| `rene_multicandidate_fusion_20261020` | [RRF3/RRF4](../results/experiment_exports/rene_multicandidate_fusion_20261020/) | Computational experiment |
| `rene_envelope_completion_20261020` | [Full-H opportunity accounting](../results/experiment_exports/rene_envelope_completion_20261020/) | Computational experiment |
| `rene_budget_aggregation_20261020` | [Same-pool B/G](../results/experiment_exports/rene_budget_aggregation_20261020/) | Computational experiment |
| `rene_factor_confirmation_20261020` | [StaQC B/G confirmation](../results/experiment_exports/rene_factor_confirmation_20261020/) | Computational experiment |
| `rene_third_retriever_20261020` | [UniXcoder extension](../results/experiment_exports/rene_third_retriever_20261020/) | Computational experiment |
| `rene_t5_weight_factors_20261020` | [Matched T5 B/G](../results/experiment_exports/rene_t5_weight_factors_20261020/) | Computational experiment |
| `rene_weight_confirmation_20261021` | [New paired-weight confirmation](../results/experiment_exports/rene_weight_confirmation_20261021/) | Computational experiment |
| `rene_mechanism_review_20261020` | [Initial A/B review](../results/experiment_exports/rene_mechanism_review_20261020/) | human review |
| `rene_review_returns_20261020` | [Initial review statistics](../results/experiment_exports/rene_review_returns_20261020/) | human review |
| `rene_review_final_reanalysis_20261021` | [Supplemental A/B review](../results/experiment_exports/rene_review_final_reanalysis_20261021/) | human review |
| `rene_review_completion_20261021` | [Completed relevance bounds](../results/experiment_exports/rene_review_completion_20261021/) | human review |
| `rene_relevance_sensitivity_20261021` | [Expanded relevance analysis](../results/experiment_exports/rene_relevance_sensitivity_20261021/) | human review |
| `rene_relevance_sampling_uncertainty_20261021` | [Sampling uncertainty](../results/experiment_exports/rene_relevance_sampling_uncertainty_20261021/) | human review |
| `rene_cqc_training_fidelity_20261020` | [Eight-arm training sensitivity](../results/experiment_exports/rene_cqc_training_fidelity_20261020/) | Computational experiment |

The human review rows all denote human judgments. Portable replay covers the supplied numeric/label inputs, as listed in `EXPECTED_RESULTS.md`; historical model pipelines require separately obtained upstream inputs.
