## Table S7. Full-H opportunity-loss and verification accounting

| Retriever / stage | Mean available gain | Mean unretained gain | Mean ranking loss | Verification pairs | Input tokens |
| --- | --- | --- | --- | --- | --- |
| CodeBERT / RIDGE_gain | 0.06609266 | 0.05086288 | 0.01515114 | 0 | 0 |
| CodeBERT / RIDGE_verified | 0.06609266 | 0.05000006 | 0.01524555 | 17,056 | 1,164,129 |
| CodeBERT / RIDGE_risk | 0.06609266 | 0.06558435 | 0.00068608 | 17,056 | 1,164,129 |
| BM25 / RIDGE_gain | 0.01941318 | 0.01832012 | 0.00317249 | 0 | 0 |
| BM25 / RIDGE_verified | 0.01941318 | 0.01855408 | 0.00257590 | 17,056 | 1,394,014 |
| BM25 / RIDGE_risk | 0.01941318 | 0.01941318 | 0.00000000 | 17,056 | 1,394,014 |

Each retriever uses all 8,530 queries in complete H, across 1,963 connected groups. Available gain is max(best3 ENTR RR@100 minus original-query RR@100, 0), retaining the original query as a safety option. For a stage change D and available gain U, unretained positive gain is U−max(D,0), and ranking loss is max(−D,0). The displayed means satisfy mean(U)−mean(D)=mean(unretained gain)+mean(ranking loss); Table S5 reports the corresponding net changes and positive-gain retention.
These are descriptive arithmetic summaries of the frozen outcomes, not a new comparison family or confidence interval. Verification pairs and input tokens are cumulative charged workload. Later risk fallback does not refund verification already performed. The verified-stage policy also includes queries not selected for cross-encoder scoring; their gain-stage predictions proceed to risk assessment.
Source: artifact/numerical/data/full_H_stages_CodeBERT.jsonl.gz and full_H_stages_BM25.jsonl.gz. Run tools/rebuild_s7.py --check to recompute all six rows.
