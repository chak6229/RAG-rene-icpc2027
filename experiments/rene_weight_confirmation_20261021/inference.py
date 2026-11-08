"""Paired weights, label-blocked candidate generation and deterministic retrieval."""
from __future__ import annotations

import argparse
import builtins
import contextlib
import gc
import io
import json
import random
import time
from pathlib import Path

import numpy as np

from .common import MODEL_PATHS, OLD, OUT, ROOT, WEIGHTS, now, progress, read, record, rows, sha, verify_design, verify_seal, write, writel
from experiments.rene_factor_confirmation_20261020.inference import generate_query_block
from experiments.rene_t5_weight_factors_20261020.metrics import order_candidates


@contextlib.contextmanager
def deny_labels():
    original=(builtins.open,io.open,Path.open)
    blocked={'ELIGIBLE_PAIRS.jsonl','EXPOSURE_MANIFEST.json','SAMPLE_PLAN.json','STAQC_ELIGIBLE_PAIRS.jsonl',
             'STAQC_PAIR_METADATA.jsonl','reference_fingerprints.pkl','RESULTS.json','per_query.jsonl','paired_per_query.jsonl'}
    def check(file,mode):
        if isinstance(file,(str,Path)) and ('r' in mode or '+' in mode):
            p=Path(file).resolve()
            if 'private' in p.parts or p.name in blocked or (OLD/'sources') in p.parents:
                raise PermissionError('Label-bearing source/outcomes blocked during inference: '+p.name)
    def b(file,mode='r',*args,**kwargs):check(file,mode);return original[0](file,mode,*args,**kwargs)
    def i(file,mode='r',*args,**kwargs):check(file,mode);return original[1](file,mode,*args,**kwargs)
    def p(file,mode='r',*args,**kwargs):check(file,mode);return original[2](file,mode,*args,**kwargs)
    builtins.open,io.open,Path.open=b,i,p
    try:yield
    finally:builtins.open,io.open,Path.open=original


def setup():
    import torch
    random.seed(101);np.random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch


def generate():
    verify_design()
    if (OUT/'CANDIDATES_SEALED.json').exists():return verify_seal('CANDIDATES_SEALED.json')
    torch=setup()
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import load_tokenizer
    queries=list(rows(OUT/'public/queries.jsonl'))
    outputs=[]
    for arm in WEIGHTS:
        path=OUT/f'candidates/{arm}.jsonl';manifest=OUT/f'candidates/{arm}.manifest.json'
        actions=OUT/f'actions/{arm}.jsonl'
        if manifest.exists():
            for item in read(manifest)['files']:
                if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Completed candidates changed')
            outputs.extend(read(manifest)['files']);outputs.append(record(manifest));continue
        path.parent.mkdir(parents=True,exist_ok=True)
        partial=path.with_suffix('.partial.jsonl')
        counts={};previous_ids=[]
        if partial.exists():
            for r in rows(partial):
                if r['query_id'] not in counts:previous_ids.append(r['query_id'])
                counts.setdefault(r['query_id'],set()).add(r['position'])
        if previous_ids!=[q['query_id'] for q in queries[:len(previous_ids)]]:raise ValueError('Partial query order mismatch')
        for q in queries[:len(previous_ids)]:
            if counts[q['query_id']]!=set(range(1,len(q['query'].split())+1)):raise ValueError('Incomplete committed query')
        checkpoint=ROOT/MODEL_PATHS[arm]
        with deny_labels():
            tok=load_tokenizer(checkpoint)
            model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint,local_files_only=True).float().cuda().eval()
            started=time.perf_counter();previous=len(previous_ids)
            with partial.open('a',encoding='utf-8') as h:
                for offset in range(previous,len(queries),32):
                    current=generate_query_block(model,tok,queries[offset:offset+32],32)
                    h.write(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in current));h.flush()
                    done=min(offset+32,len(queries))
                    if done%256==0 or done==len(queries):progress('generation',weight=arm,queries_done=done,queries_total=len(queries),seconds=round(time.perf_counter()-started,1))
            del model;gc.collect();torch.cuda.empty_cache()
        partial.rename(path)
        by={}
        for r in rows(path):by.setdefault(r['query_id'],[]).append(r)
        writel(actions,[{'query_id':q['query_id'],'positions':len(by[q['query_id']]),
                        'orders':{p:order_candidates(by[q['query_id']],p) for p in ('ENTR','PROB')}} for q in queries])
        write(manifest,{'created_utc':now(),'weight':arm,'design_sha256':sha(OUT/'DESIGN.json'),
            'queries':len(queries),'candidates':sum(len(p) for p in by.values()),'files':[record(path),record(actions)],
            'qrels_read':False,'new_effects_computed':False,'seconds_this_session':time.perf_counter()-started})
        outputs.extend(read(manifest)['files']);outputs.append(record(manifest));del by
    write(OUT/'CANDIDATES_SEALED.json',{'created_utc':now(),'design_sha256':sha(OUT/'DESIGN.json'),
        'files':outputs,'qrels_read':False,'new_effects_computed':False,'extra_seed_generations':0})


def stable_top(values,ids,depth=100):
    values=np.asarray(values)
    if len(values)!=len(ids) or not np.isfinite(values).all():raise ValueError('Invalid retrieval scores')
    k=min(depth,len(values))
    if k==len(values):return np.lexsort((ids,-values))[:k]
    ix=np.argpartition(-values,k-1)[:k]
    boundary=np.min(values[ix])
    better=np.flatnonzero(values>boundary)
    tied=np.flatnonzero(values==boundary)
    tied=tied[np.argsort(ids[tied],kind='stable')][:k-len(better)]
    chosen=np.concatenate([better,tied])
    return chosen[np.lexsort((ids[chosen],-values[chosen]))]


def rank():
    verify_design();verify_seal('CANDIDATES_SEALED.json')
    if (OUT/'RANKINGS_SEALED.json').exists():return verify_seal('RANKINGS_SEALED.json')
    torch=setup()
    from threadpoolctl import threadpool_limits
    from experiments.rene_ssqr_alignment_20261019.pipeline import load_trained_retriever
    from experiments.rene_ssqr_alignment_20261019.retriever import encode_texts
    corpus=list(rows(OUT/'public/corpus.jsonl'))
    texts=list(dict.fromkeys([q['query'] for q in rows(OUT/'public/queries.jsonl')]+
          [r['rewrite'] for arm in WEIGHTS for r in rows(OUT/f'candidates/{arm}.jsonl')]))
    ids=np.array([r['doc_id'] for r in corpus],np.int64)
    cache=OUT/'retrieval_cache';cache.mkdir(parents=True,exist_ok=True)
    if not (cache/'texts.json').exists():write(cache/'texts.json',texts)
    elif read(cache/'texts.json')!=texts:raise ValueError('Retrieval text order changed')
    started=time.perf_counter()
    with deny_labels():
        model,tok=load_trained_retriever(ROOT/'models/codebert-ssqr-pooler-2epoch-20261019','cuda')
        for kind,current in [('code',[r['code_text'] for r in corpus]),('query',texts)]:
            target=cache/f'{kind}_vectors.npy'
            completion=cache/f'{kind}_vectors.manifest.json'
            if completion.exists():
                if sha(target)!=read(completion)['sha256']:raise ValueError('Embedding cache changed')
                continue
            partial=cache/f'{kind}_vectors.partial.npy';state=cache/f'{kind}_encoding_state.json'
            start=read(state)['rows_done'] if state.exists() else 0
            if partial.exists():vectors=np.lib.format.open_memmap(partial,mode='r+')
            else:vectors=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(len(current),768))
            if vectors.shape!=(len(current),768):raise ValueError('Embedding shape mismatch')
            for offset in range(start,len(current),256):
                last=min(offset+256,len(current))
                vectors[offset:last]=encode_texts(model,tok,current[offset:last],batch_size=32,device='cuda')
                vectors.flush()
                state.write_text(json.dumps({'rows_done':last,'rows_total':len(current),'kind':kind}),encoding='utf-8')
                if last%1024==0 or last==len(current):progress('encoding',kind=kind,rows_done=last,rows_total=len(current),seconds=round(time.perf_counter()-started,1))
            del vectors;partial.rename(target)
            write(completion,{'created_utc':now(),'sha256':sha(target),'rows':len(current),'shape':[len(current),768],
                  'qrels_read':False,'batch_size':32,'precision':'float32'})
        del model;gc.collect();torch.cuda.empty_cache()
        cv=np.load(cache/'code_vectors.npy',mmap_mode='r');qv=np.load(cache/'query_vectors.npy',mmap_mode='r')
        path=OUT/'rankings/CodeBERT.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        partial=path.with_suffix('.partial.jsonl')
        count=0
        if partial.exists():
            for i,r in enumerate(rows(partial)):
                if i>=len(texts) or r['text']!=texts[i]:raise ValueError('Partial rankings out of order')
                count+=1
        with partial.open('a',encoding='utf-8') as h,threadpool_limits(limits=4):
            for offset in range(count,len(texts),128):
                scores=qv[offset:offset+128]@cv.T
                for text,values in zip(texts[offset:offset+128],scores):
                    order=stable_top(values,ids)
                    h.write(json.dumps({'text':text,'ranked_doc_ids':ids[order].tolist(),'scores':values[order].astype(float).tolist()},ensure_ascii=False,allow_nan=False)+'\n')
                h.flush()
                done=min(offset+128,len(texts))
                if done%4096==0 or done==len(texts):progress('ranking',texts_done=done,texts_total=len(texts),seconds=round(time.perf_counter()-started,1))
        partial.rename(path)
    outputs=[record(path)]+[record(p) for p in sorted(cache.iterdir()) if p.is_file() and 'state' not in p.name]
    write(OUT/'RANKINGS_SEALED.json',{'created_utc':now(),'design_sha256':sha(OUT/'DESIGN.json'),
        'files':outputs,'unique_texts':len(texts),'qrels_read':False,'new_effects_computed':False,'seconds_this_session':time.perf_counter()-started})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['generate','rank','all'])
    args=parser.parse_args()
    if args.mode in ('generate','all'):generate()
    if args.mode in ('rank','all'):rank()
