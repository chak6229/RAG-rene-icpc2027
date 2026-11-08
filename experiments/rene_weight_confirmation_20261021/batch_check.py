"""Use already consumed source queries to check both-weight batch32 fidelity."""
from __future__ import annotations

import gc
import time

from .common import MODEL_PATHS, OUT, ROOT, WEIGHTS, digest, now, rows, write


def main():
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import load_tokenizer
    from experiments.rene_factor_confirmation_20261020.inference import generate_query_block
    from experiments.rene_t5_weight_factors_20261020.metrics import order_candidates
    from .inference import setup
    setup()
    queries=list(rows(ROOT/'results/rene_ssqr_bridge_20261019/data/source/queries.jsonl'))
    base_ids={r['query_id'] for r in rows(ROOT/'results/rene_ssqr_rewriter_envelope_20261020/candidates/source/t5base.jsonl')}
    chosen=sorted((q for q in queries if q['query_id'] in base_ids),key=lambda q:digest('weight-confirm-batch/'+q['query_id']))[:32]
    selected={q['query_id'] for q in chosen}
    checks={}
    for arm in WEIGHTS:
        path=ROOT/('results/rene_ssqr_bridge_20261019/candidates/source/baseline.jsonl' if arm=='local' else 'results/rene_ssqr_rewriter_envelope_20261020/candidates/source/t5base.jsonl')
        old={(r['query_id'],r['position']):r for r in rows(path) if r['query_id'] in selected}
        checkpoint=ROOT/MODEL_PATHS[arm]
        tok=load_tokenizer(checkpoint);model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).float().cuda().eval()
        start=time.perf_counter();new=generate_query_block(model,tok,chosen,32)
        mismatch=[(r['query_id'],r['position']) for r in new if r['span']!=old[r['query_id'],r['position']]['span']]
        eh=max(abs(r['full_vocab_entropy']-old[r['query_id'],r['position']]['full_vocab_entropy']) for r in new)
        ep=max(abs(r['label_mean_log_probability']-old[r['query_id'],r['position']]['label_mean_log_probability']) for r in new)
        orders=[]
        for q in chosen:
            a=[r for r in new if r['query_id']==q['query_id']];b=[r for (qid,p),r in old.items() if qid==q['query_id']]
            for policy in ('ENTR','PROB'):
                if order_candidates(a,policy)!=order_candidates(b,policy):orders.append((q['query_id'],policy))
        passed=not mismatch and not orders and max(eh,ep)<5e-5
        checks[arm]={'passed':passed,'queries':len(chosen),'positions':len(new),'span_mismatches':mismatch,
                    'full_order_mismatches':orders,'maximum_entropy_error':eh,'maximum_logp_error':ep,'seconds':time.perf_counter()-start}
        print({arm:checks[arm]},flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    ok=all(c['passed'] for c in checks.values())
    write(OUT/'BATCHING_DEVELOPMENT_CHECK.json',{'created_utc':now(),'passed':ok,'checks':checks,
          'scope':'consumed source queries only, spans and full ENTR/PROB order, no new retrieval effects'})
    if not ok:raise ValueError('Development batch fidelity failure; do not freeze inference')


if __name__=='__main__':main()
