# SSQR Three alignments

Update: 2026 - 10-19 T07: 50: 08.028369 + 00: 00. Status: Full training and final acceptance pass.

| Item | Realization and acceptance | Status |
|---|---|---|
| CoSQA Input interface | Only doc for true query; code_tokens Strings keep words; Official ID; public Query/code separated from private qrels; 20,604 Matching, 10,020 Querying, 6,267 | Completed |
| T5 Negative entropy position rating | Original teacher-forcing logits 32,128 full word entropy; Including sentry/EOS, exclusion padding; ENTR / PROB / RAND shared caches; Check for actual weights | Completed |
| CodeBERT Training and reasoning | Retrain from base pooler, without unitation point accumulation, query → code full batch CE; 32 Matching, 256 length, two rounds 15,740 steps, linear10%warmup, weight_decay = 0 | Completed |

Final weight:`models/codebert-ssqr-pooler-2epoch-20261019/checkpoint-15740 `；SHA-256 `0d533c04602916c22d1f13d282056f081b6bd04f9866ccc1ded46a0516e13e3d `.

19 Contract test passed. Real T5 lot/bar Max deviation 2.8610e-6, independent FP64 entropy max variance 3.4758e-6. The main entropy contains all padding tag positions; span-only Entropy is a separate column, which cannot be mixed.

## Revert Settings

- Default Follow Open JSON Source 1 n Position, visible modern greed settings; Also supports the paper 0...n position. The publicly available author does not record the full decode option and cannot claim that the original decoder status has been restored.
- PROB is defined as the probability of teacher-forced average labels log; Author is not publicly available PROB source code, this operational definition needs to be frozen before the new experiment. The random position is not put back, the selection from the same cache is not sufficient and the original sentence is not filled.
- T5 Use local best92000 weights; Source family T5Tokenizer legacy = True and base SentencePiece, word by word check local fast tokenizer.
- CodeBERT Sovereignty reverted to the scheduled final two rounds checkpoint; Save only best selected by CodeXGLUE valid. Modern PyTorch, BF16 and activated check points are operating environment differences; Negative reservation in full. The last batch that is insufficient 32 achieves the reservation by source.
- The new weights are kept separately in models/codebert-ssqr-pooler-2epoch-20261019; Old weight and five places CURRENT/STABLE are checked to be the same.

## One correction from the old review

The previous freeze review incorrectly wrote the local complete CodeBERT learning rate schedule as constant; The original local script is already in use linear+10%warmup. What needs to be corrected is CLS / Normalized training target. The evidence of the old freeze is retained and has been further corrected in this report.

## Evidence of acceptance and next steps

- IMPLEMENTATION_VERIFICATION.json、TEST_RESULTS.json、T5_REAL_WEIGHT_SMOKE.json、TRAINING_BENCHMARK.json。
- FINAL_ACCEPTANCE.json: The actual final weight passes the entire chain ranking for author vectors/losses, independent reloading, synthesis input.
- CoSQA Method effects have not been calculated. The next step is to freeze the development/confirmation of clusters, set-up and statistical comparisons in accordance with the established plan, before developing test runs and efficiency estimates; This task does not generate new 4/8 seed candidates. risk_certified = false, NoRewrite Maintain.
