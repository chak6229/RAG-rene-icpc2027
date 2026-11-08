from __future__ import annotations

import builtins
import contextlib
import gc
import io
import json
import time
from pathlib import Path

import numpy as np

from .common import ROOT, OUT, now, read, record, rows, sha, write, writel


def verify_design(*, private=False):
    d = read(OUT / 'DESIGN.json')
    for r in d['bound_files'] + d['protected_files']:
        if not private and '/private/' in r['path']:
            continue
        if sha(ROOT / r['path']) != r['sha256']:
            raise ValueError('Frozen input changed: ' + r['path'])
    return d


def verify_seal(name):
    seal = read(OUT / name)
    if seal['design_sha256'] != sha(OUT / 'DESIGN.json'):
        raise ValueError('Design seal mismatch')
    for r in seal['files']:
        if sha(ROOT / r['path']) != r['sha256']:
            raise ValueError('Sealed file changed: ' + r['path'])
    return seal


def progress(stage, **kwargs):
    v = {'updated_utc': now(), 'stage': stage, 'completed': False, **kwargs}
    (OUT / 'RUN_STATUS.json').write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(v, flush=True)


@contextlib.contextmanager
def deny_labels():
    old_builtin, old_io, old_path = builtins.open, io.open, Path.open
    blocked_names = {'STAQC_ELIGIBLE_PAIRS.jsonl', 'STAQC_PAIR_METADATA.jsonl', 'reference_fingerprints.pkl', 'RESULTS.json', 'per_query.jsonl'}

    def check(file, mode='r'):
        if isinstance(file, (str, Path)) and ('r' in mode or '+' in mode):
            p = Path(file).resolve()
            if 'private' in p.parts or (OUT / 'sources') in p.parents or p.name in blocked_names:
                raise PermissionError('Label-bearing source/outcomes blocked during inference: ' + p.name)

    def b(file, mode='r', *a, **kw):
        check(file, mode); return old_builtin(file, mode, *a, **kw)

    def i(file, mode='r', *a, **kw):
        check(file, mode); return old_io(file, mode, *a, **kw)

    def p(file, mode='r', *a, **kw):
        check(file, mode); return old_path(file, mode, *a, **kw)

    builtins.open, io.open, Path.open = b, i, p
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = old_builtin, old_io, old_path


def generate_query_block(model,tokenizer,queries,batch_size=32):
    """Same position/greedy/teacher-forcing contract, batching across queries."""
    import torch
    from experiments.rene_ssqr_alignment_20261019.t5 import insertion_inputs,_inputs,teacher_forced_scores
    records=[]
    by={q['query_id']:q['query'].split() for q in queries}
    for q in queries:
        for r in insertion_inputs(q['query'],position_policy='source_1_to_n'):
            records.append({'query_id':q['query_id'],**r})
    model.eval();device=next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0,len(records),batch_size):
            batch=records[start:start+batch_size]
            inputs=_inputs(tokenizer,[r['masked_query'] for r in batch],device)
            generated=model.generate(**inputs,do_sample=False,num_beams=1,num_return_sequences=1,
                 max_new_tokens=19,output_scores=False,return_dict_in_generate=False,use_cache=True,
                 decoder_start_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
            for r,ids in zip(batch,generated):
                span=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()
                words=by[r['query_id']];p=r['position'];rewrite=' '.join(words[:p]+([span] if span else [])+words[p:])
                r.update(span=span,rewrite=rewrite,is_identity=rewrite==' '.join(words))
    metrics=teacher_forced_scores(model,tokenizer,[r['masked_query'] for r in records],[r['span'] for r in records],batch_size=batch_size)
    for r,m in zip(records,metrics):r.update(m)
    return records


def generate():
    d = verify_design()
    if (OUT / 'CANDIDATES_SEALED.json').exists():
        return verify_seal('CANDIDATES_SEALED.json')
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import GenerationSettings, generate_candidates, load_tokenizer
    from experiments.rene_budget_aggregation_20261020.metrics import ordered_positions
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    queries = list(rows(OUT / 'public/queries.jsonl'))
    model_path = ROOT / d['generation']['checkpoint']
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).float().cuda().eval()
    tokenizer = load_tokenizer(model_path)
    path = OUT / 'candidates.jsonl'
    partial = OUT / 'candidates.partial.jsonl'
    old = list(rows(partial)) if partial.exists() else []
    by_q = {}
    for r in old:
        by_q.setdefault(r['query_id'], []).append(r)
    for q in queries:
        if q['query_id'] in by_q:
            if {r['position'] for r in by_q[q['query_id']]} != set(range(1,len(q['query'].split())+1)):
                raise ValueError('Incomplete previously committed query; refusing silent continuation')
    if set(by_q) - {q['query_id'] for q in queries}:
        raise ValueError('Foreign query in partial candidate bank')
    started = time.perf_counter()
    remaining=[q for q in queries if q['query_id'] not in by_q]
    previous=len(by_q)
    with deny_labels(), partial.open('a', encoding='utf-8') as h:
        for offset in range(0,len(remaining),32):
            block=remaining[offset:offset+32]
            current = generate_query_block(model,tokenizer,block,batch_size=32)
            h.write(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in current)); h.flush()
            done=previous+min(offset+32,len(remaining))
            if done%128 == 0 or done == len(queries):
                progress('generation',queries_done=done,queries_total=len(queries),seconds=round(time.perf_counter()-started,1))
    partial.rename(path)
    del model; gc.collect(); torch.cuda.empty_cache()
    by_q = {}
    for r in rows(path):
        by_q.setdefault(r['query_id'], []).append(r)
    selections = []
    for q in queries:
        pool = by_q[q['query_id']]
        selections.append({'query_id':q['query_id'], 'positions':len(pool),
                           'orders':{p:ordered_positions(pool,p) for p in ('ENTR','PROB')}})
    writel(OUT / 'actions.jsonl',selections)
    write(OUT / 'CANDIDATES_SEALED.json',{'created_utc':now(),'design_sha256':sha(OUT/'DESIGN.json'),
          'queries':len(queries),'candidates':sum(len(v) for v in by_q.values()),
          'files':[record(path),record(OUT/'actions.jsonl')], 'qrels_read':False,
          'new_effects_computed':False,'extra_seed_generations':0,'seconds_this_session':time.perf_counter()-started})


def stable_order(values,ids,depth=100):
    return np.lexsort((ids,-np.asarray(values)))[:depth]


def rank():
    d = verify_design(); verify_seal('CANDIDATES_SEALED.json')
    if (OUT/'RANKINGS_SEALED.json').exists():
        return verify_seal('RANKINGS_SEALED.json')
    import torch
    from threadpoolctl import threadpool_limits
    from experiments.rene_ssqr_alignment_20261019.pipeline import load_trained_retriever
    from experiments.rene_ssqr_alignment_20261019.retriever import encode_texts
    from experiments.rene_cosqa_pilot_20261019.ranking import BM25
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    corpus=list(rows(OUT/'public/corpus.jsonl'))
    texts=list(dict.fromkeys([q['query'] for q in rows(OUT/'public/queries.jsonl')]+[r['rewrite'] for r in rows(OUT/'candidates.jsonl')]))
    ids=np.array([r['doc_id'] for r in corpus],np.int64)
    outputs=[]
    for ret in ('CodeBERT','BM25'):
        path=OUT/f'rankings/{ret}.jsonl'
        manifest=path.with_suffix('.manifest.json')
        if manifest.exists():
            if sha(path)!=read(manifest)['ranking_sha256']:
                raise ValueError('Completed ranking changed')
            outputs += [record(path),record(manifest)]
            continue
        path.parent.mkdir(parents=True,exist_ok=True)
        artifacts=OUT/f'retrieval_cache/{ret}';artifacts.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        with deny_labels():
            if ret=='CodeBERT':
                model,tok=load_trained_retriever(ROOT/'models/codebert-ssqr-pooler-2epoch-20261019','cuda')
                for label,current in (('code',[r['code_text'] for r in corpus]),('query',texts)):
                    vp=artifacts/f'{label}_vectors.npy'
                    if not vp.exists():
                        vv=[]
                        for offset in range(0,len(current),512):
                            vv.append(encode_texts(model,tok,current[offset:offset+512],batch_size=32,device='cuda'))
                            progress('encoding',kind=label,rows_done=min(offset+512,len(current)),rows_total=len(current))
                        np.save(vp,np.concatenate(vv),allow_pickle=False)
                cv=np.load(artifacts/'code_vectors.npy');qv=np.load(artifacts/'query_vectors.npy')
                if cv.shape!=(len(corpus),768) or qv.shape!=(len(texts),768):
                    raise ValueError('Embedding coverage mismatch')
                if not (artifacts/'texts.json').exists():
                    write(artifacts/'texts.json',texts)
                elif read(artifacts/'texts.json')!=texts:
                    raise ValueError('Text order mismatch')
                del model;gc.collect();torch.cuda.empty_cache()
            else:
                bm=BM25([r['code_text'] for r in corpus],k1=1.2,b=.75)
            partial=path.with_suffix('.partial.jsonl')
            old=list(rows(partial)) if partial.exists() else []
            if any(r['text']!=texts[i] for i,r in enumerate(old)):
                raise ValueError('Partial ranking order mismatch')
            with partial.open('a',encoding='utf-8') as h,threadpool_limits(limits=4):
                for offset in range(len(old),len(texts),128):
                    current=texts[offset:offset+128]
                    scores=qv[offset:offset+len(current)]@cv.T if ret=='CodeBERT' else np.stack([bm.score(t) for t in current])
                    for text,values in zip(current,scores):
                        order=stable_order(values,ids)
                        h.write(json.dumps({'text':text,'ranked_doc_ids':ids[order].tolist(),'scores':values[order].astype(float).tolist()},ensure_ascii=False,allow_nan=False)+'\n')
                    h.flush()
                    if offset%1024==0 or offset+len(current)==len(texts):
                        progress('ranking',retriever=ret,texts_done=offset+len(current),texts_total=len(texts))
            partial.rename(path)
        write(manifest,{'created_utc':now(),'retriever':ret,'ranking_sha256':sha(path),'design_sha256':sha(OUT/'DESIGN.json'),
              'corpus_sha256':sha(OUT/'public/corpus.jsonl'),'unique_texts':len(texts),'qrels_read':False,
              'files':[record(p) for p in sorted(artifacts.iterdir()) if p.is_file()],
              'seconds':time.perf_counter()-started})
        outputs += [record(path),record(manifest)]
    write(OUT/'RANKINGS_SEALED.json',{'created_utc':now(),'design_sha256':sha(OUT/'DESIGN.json'),
          'files':outputs,'qrels_read':False,'new_effects_computed':False})


if __name__=='__main__':
    generate(); rank()
