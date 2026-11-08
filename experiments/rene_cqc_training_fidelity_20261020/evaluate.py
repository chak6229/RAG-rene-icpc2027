"""Outcome-blind generation, fixed-pool rescoring and frozen CodeBERT retrieval."""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from .common import ROOT, PACKAGE, OUT, STORE, BRIDGE, ARM_SPECS, BASE, PROTECTED, now, read, rows, write, writel, record, sha, require, verify, text_id


def paths(domain):
    return dict(queries=OUT/f'data/{domain}_queries.jsonl',
        corpus=BRIDGE/'data/source/corpus.jsonl' if domain=='source' else ROOT/'results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl',
        qrels=BRIDGE/f'data/private/{domain}_qrels.jsonl',
        fixed=BRIDGE/'candidates/source/baseline.jsonl' if domain=='source' else ROOT/'results/rene_cosqa_development_20261019/candidates.jsonl',
        old_rankings=BRIDGE/f'rankings/{domain}/CodeBERT.jsonl',
        codes=BRIDGE/'retrieval_cache/source/CodeBERT/code_vectors.npy' if domain=='source' else ROOT/'results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy')


def freeze():
    verify()
    bound=list(PACKAGE.glob('*.py'))+[PACKAGE/'DESIGN.md',OUT/'TRAIN_PROTOCOL.json',OUT/'RUNTIME_DETERMINISM.json',OUT/'EVAL_TESTS.json']
    for domain in ('source','cosqa'):
        bound += [p for key,p in paths(domain).items()]
    bound += [ROOT/'experiments/rene_ssqr_alignment_20261019/t5.py',ROOT/'experiments/rene_ssqr_alignment_20261019/retriever.py',
        ROOT/'experiments/rene_ssqr_alignment_20261019/pipeline.py',ROOT/'experiments/rene_envelope_completion_20261020/common.py',
        ROOT/'models/codebert-ssqr-pooler-2epoch-20261019/model.safetensors',
        ROOT/'models/codebert-ssqr-pooler-2epoch-20261019/ssqr_representation_contract.json',
        ROOT/'models/codebert-ssqr-pooler-2epoch-20261019/TRAINING_CONTRACT.json',
        ROOT/'models/codebert-ssqr-pooler-2epoch-20261019/training_complete.json',
        ROOT/'results/rene_ssqr_bridge_20261019/PROTOCOL.json']
    write(OUT/'EVAL_PROTOCOL.json',dict(created_utc=now(),status='FROZEN_BEFORE_NEW_GENERATION_AND_OUTCOMES',
        training_protocol_sha256=sha(OUT/'TRAIN_PROTOCOL.json'),new_confirmation=False,risk_certified=False,
        code_and_input_files=[record(p) for p in sorted(set(bound))],
        candidate_views=['own_generated_pool','fixed_historical_local_pool_rescored'],
        score_precision='FP32 raw full vocabulary entropy; independent FP64 verification',
        generation='same1..n positions, greedy19 new tokens, batch8, no dropout',
        corpus_scope='original bridge corpora19210/6267',retrieval='FP32 pooler dot, TF32 off, max_length256, batch16',
        tie_rule='score descending then integer doc_id ascending',stored_full_new_scores=True,
        exact_random='original positions uniformly without replacement, preserve duplicate text probability',
        primary_family_size=28,statistics='10000 original-group shared bootstrap and cluster-t, Bonferroni',
        no_H_outcomes=True,no_new_policy_selection=True))
    print(dict(stage='evaluation_frozen'),flush=True)


def check_eval():
    p=read(OUT/'EVAL_PROTOCOL.json')
    require(p['training_protocol_sha256']==sha(OUT/'TRAIN_PROTOCOL.json'),'Training protocol binding')
    for r in p['code_and_input_files']:
        require(sha(ROOT/r['path'])==r['sha256'],'Evaluation input changed: '+r['path'])
    return p


def order(pool,policy):
    return [r['position'] for r in sorted(pool,key=(lambda r:(r['full_vocab_entropy'],r['position'])) if policy=='ENTR'
        else (lambda r:(-r['label_mean_log_probability'],r['position'])))]


def generate():
    check_eval();verify()
    require((OUT/'TRAINING_COMPLETE.json').exists(),'All fixed-budget training must complete before evaluation')
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import load_tokenizer,generate_candidates,GenerationSettings,teacher_forced_scores
    from experiments.rene_envelope_completion_20261020.common import deny_labels
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    queries={d:rows(paths(d)['queries']) for d in ('source','cosqa')}
    fixed={}
    for d in queries:
        ids={q['query_id'] for q in queries[d]};fixed[d]=[r for r in rows(paths(d)['fixed']) if r['query_id'] in ids]
    files=[]
    with deny_labels():
        for arm in ARM_SPECS:
            report=read(OUT/f'training/{arm}.json')
            require(report['weights_sha256']==sha(STORE/arm/'model.safetensors'),'Trained weights changed')
            model=AutoModelForSeq2SeqLM.from_pretrained(STORE/arm,local_files_only=True).float().cuda().eval()
            tokenizer=load_tokenizer(STORE/arm)
            for domain in queries:
                target=OUT/f'candidates/{arm}/{domain}_own.jsonl';rescored=OUT/f'candidates/{arm}/{domain}_fixed.jsonl'
                manifest_path=OUT/f'candidates/{arm}/{domain}_MANIFEST.json'
                if manifest_path.exists():
                    manifest=read(manifest_path)
                    for item in manifest['files']:require(sha(ROOT/item['path'])==item['sha256'],'Existing candidates changed')
                    files += [target,rescored,manifest_path];continue
                require(not target.exists() and not rescored.exists(),'Unsealed candidate bank; investigate before continuation')
                started=time.perf_counter();partial=target.with_suffix('.partial.jsonl')
                existing=rows(partial) if partial.exists() else []
                done={r['query_id'] for r in existing}
                require(done<={q['query_id'] for q in queries[domain]},'Foreign partial query')
                with partial.open('a',encoding='utf-8') as stream:
                    for i,q in enumerate(queries[domain]):
                        if q['query_id'] in done:continue
                        candidates=generate_candidates(model,tokenizer,q['query_id'],q['query'],GenerationSettings())
                        for r in candidates:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
                        stream.flush()
                        if (i+1)%64==0:
                            print(dict(stage='generation',arm=arm,domain=domain,queries=i+1,total=512,seconds=round(time.perf_counter()-started,1)),flush=True)
                partial.rename(target)
                old=fixed[domain]
                scores=teacher_forced_scores(model,tokenizer,[r['masked_query'] for r in old],[r['span'] for r in old],batch_size=8)
                updated=[]
                for r,score in zip(old,scores):
                    updated.append({**r,**score,'rescored_arm':arm})
                writel(rescored,updated)
                write(manifest_path,dict(created_utc=now(),arm=arm,domain=domain,queries=512,own_candidates=len(rows(target)),
                    fixed_candidates=len(updated),qrels_read=False,weights_sha256=report['weights_sha256'],
                    seconds=time.perf_counter()-started,files=[record(target),record(rescored)]))
                files += [target,rescored,manifest_path]
            del model;gc.collect();torch.cuda.empty_cache()
    selections=[]
    for arm in ARM_SPECS:
        for domain in queries:
            for view in ('own','fixed'):
                grouped=defaultdict(list)
                for r in rows(OUT/f'candidates/{arm}/{domain}_{view}.jsonl'):grouped[r['query_id']].append(r)
                for q in queries[domain]:
                    qid=q['query_id'];pool=grouped[qid]
                    require(sorted(r['position'] for r in pool)==list(range(1,len(q['query'].split())+1)),'Incomplete position coverage')
                    selections.append(dict(arm=arm,domain=domain,view=view,query_id=qid,ENTR=order(pool,'ENTR'),PROB=order(pool,'PROB')))
    writel(OUT/'selections.jsonl',selections);files.append(OUT/'selections.jsonl')
    check_eval();verify()
    write(OUT/'CANDIDATES_SEALED.json',dict(created_utc=now(),eval_protocol_sha256=sha(OUT/'EVAL_PROTOCOL.json'),
        files=[record(p) for p in files],qrels_read=False,selection_rows=len(selections)))


def retrieve():
    check_eval();verify()
    seal=read(OUT/'CANDIDATES_SEALED.json')
    for r in seal['files']:require(sha(ROOT/r['path'])==r['sha256'],'Candidate seal')
    import torch
    from experiments.rene_envelope_completion_20261020.common import deny_labels
    from experiments.rene_ssqr_alignment_20261019.pipeline import load_trained_retriever
    from experiments.rene_ssqr_alignment_20261019.retriever import encode_texts
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    artifacts=[]
    with deny_labels():
        for domain in ('source','cosqa'):
            corpus=rows(paths(domain)['corpus']);doc_ids=np.array([r['doc_id'] for r in corpus],np.int64)
            codes=np.load(paths(domain)['codes'],allow_pickle=False)
            require(codes.shape==(len(corpus),768),'Code-vector corpus mapping')
            texts=[r['query'] for r in rows(paths(domain)['queries'])]
            for arm in ARM_SPECS:
                for view in ('own','fixed'):texts.extend(r['rewrite'] for r in rows(OUT/f'candidates/{arm}/{domain}_{view}.jsonl'))
            texts=list(dict.fromkeys(texts));wanted={text_id(t) for t in texts}
            old={r['text_id']:r for r in rows(paths(domain)['old_rankings']) if r['text_id'] in wanted}
            new=[t for t in texts if text_id(t) not in old]
            dest=OUT/f'rankings/{domain}';dest.mkdir(parents=True,exist_ok=True)
            manifest_path=dest/'MANIFEST.json'
            if manifest_path.exists():
                manifest=read(manifest_path)
                for item in manifest['files']:
                    from pathlib import Path
                    path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
                    require(sha(path)==item['sha256'],'Existing rankings changed')
                artifacts += manifest['files']+[record(manifest_path)];continue
            started=time.perf_counter();model,tokenizer=load_trained_retriever(ROOT/'models/codebert-ssqr-pooler-2epoch-20261019','cuda')
            vectors=[]
            for offset in range(0,len(new),256):
                vectors.append(encode_texts(model,tokenizer,new[offset:offset+256],batch_size=16,device='cuda'))
                print(dict(stage='retrieve_encode',domain=domain,completed=min(offset+256,len(new)),total=len(new)),flush=True)
            vectors=np.concatenate(vectors) if vectors else np.empty((0,768),np.float32)
            del model;gc.collect();torch.cuda.empty_cache()
            vector_path=dest/'new_vectors.npy';np.save(vector_path,vectors,allow_pickle=False)
            write(dest/'new_texts.json',new)
            dense_path=STORE/f'{domain}_retrieval_scores.npy'
            require(not dense_path.exists(),'Unsealed dense score file')
            dense=np.lib.format.open_memmap(dense_path,mode='w+',dtype=np.float32,shape=(len(new),len(corpus)))
            merged=dict(old)
            with threadpool_limits(limits=4):
                for offset in range(0,len(new),128):
                    scores=vectors[offset:offset+128]@codes.T;dense[offset:offset+len(scores)]=scores
                    for text,value in zip(new[offset:offset+len(scores)],scores):
                        ordered=np.lexsort((doc_ids,-value))[:100]
                        merged[text_id(text)]=dict(text_id=text_id(text),text=text,ranked_doc_ids=doc_ids[ordered].tolist(),
                            scores=value[ordered].astype(float).tolist(),origin='new_frozen_retrieval')
            dense.flush();del dense
            target=dest/'Top100.jsonl';writel(target,[merged[text_id(t)] for t in texts])
            files=[target,vector_path,dest/'new_texts.json',dense_path]
            manifest=dict(created_utc=now(),domain=domain,texts=len(texts),new_texts=len(new),reused_texts=len(texts)-len(new),
                documents=len(corpus),qrels_read=False,seconds=time.perf_counter()-started,files=[record(p) for p in files])
            write(manifest_path,manifest);artifacts+=manifest['files']+[record(manifest_path)]
    check_eval();verify()
    write(OUT/'RANKINGS_SEALED.json',dict(created_utc=now(),eval_protocol_sha256=sha(OUT/'EVAL_PROTOCOL.json'),
        candidate_seal_sha256=sha(OUT/'CANDIDATES_SEALED.json'),files=artifacts,qrels_read=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','generate','retrieve'])
    {'freeze':freeze,'generate':generate,'retrieve':retrieve}[parser.parse_args().mode]()
