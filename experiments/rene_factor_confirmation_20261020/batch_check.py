"""Development-only batching fidelity/throughput check before final freeze."""
import time

from .common import ROOT,OUT,digest,now,rows,write


def main():
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import load_tokenizer
    from .inference import generate_query_block
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    queries=list(rows(ROOT/'results/rene_ssqr_bridge_20261019/data/source/queries.jsonl'))
    chosen=sorted(queries,key=lambda q:digest('factor-batch-audit/'+q['query_id']))[:32]
    ids={q['query_id'] for q in chosen}
    old={(r['query_id'],r['position']):r for r in rows(ROOT/'results/rene_ssqr_bridge_20261019/candidates/source/baseline.jsonl') if r['query_id'] in ids}
    checkpoint=ROOT/'models/t5-cqc-codenn-1m/best';tok=load_tokenizer(checkpoint)
    model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).float().cuda().eval()
    started=time.perf_counter();new=generate_query_block(model,tok,chosen,32);elapsed=time.perf_counter()-started
    mismatch=[];eh=0.;ep=0.;orders=0
    for r in new:
        o=old[r['query_id'],r['position']]
        if r['span']!=o['span']:mismatch.append((r['query_id'],r['position']))
        eh=max(eh,abs(r['full_vocab_entropy']-o['full_vocab_entropy']))
        ep=max(ep,abs(r['label_mean_log_probability']-o['label_mean_log_probability']))
    for qid in ids:
        a=[r for r in new if r['query_id']==qid];b=[r for key,r in old.items() if key[0]==qid]
        orders+=sorted(a,key=lambda r:(r['full_vocab_entropy'],r['position']))[0]['position']!=sorted(b,key=lambda r:(r['full_vocab_entropy'],r['position']))[0]['position']
    ok=not mismatch and eh<5e-5 and ep<5e-5 and not orders
    write(OUT/'BATCHING_DEVELOPMENT_CHECK.json',{'created_utc':now(),'passed':ok,'queries':len(chosen),'candidate_positions':len(new),
          'span_mismatches':mismatch,'maximum_entropy_error':eh,'maximum_log_probability_error':ep,'ENTR_top1_order_changes':orders,
          'seconds':elapsed,'new_confirmation_outcomes_read':False,'scope':'old consumed source queries only, performance/floating-point fidelity, no batch selected by retrieval effects'})
    print({'passed':ok,'positions':len(new),'seconds':elapsed,'entropy_error':eh,'logp_error':ep,'mismatches':mismatch},flush=True)
    if not ok:raise ValueError('Batch32 does not meet development fidelity contract')


if __name__=='__main__':main()
