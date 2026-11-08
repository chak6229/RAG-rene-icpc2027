"""Source, exposure, factorial and numerical replay without production metric imports."""
from __future__ import annotations

import argparse
import collections
import gc
import math
import pickle
from fractions import Fraction

import numpy as np
from scipy.stats import t

from .common import OLD, OUT, PRIMARY, REPETITIONS, ROOT, SEED, WEIGHTS, digest, now, read, record, rows, sha, verify_design, verify_seal, write


def require(condition,message):
    if not condition:raise ValueError(message)


def preopen():
    d=verify_design(private=True);verify_seal('CANDIDATES_SEALED.json');verify_seal('RANKINGS_SEALED.json')
    require(not (OUT/'OUTCOME_OPENING.json').exists(),'Audit must precede new method effects')
    queries=list(rows(OUT/'public/queries.jsonl'));corpus=list(rows(OUT/'public/corpus.jsonl'))
    groups=list(rows(OUT/'public/groups.jsonl'));sample=read(OUT/'SAMPLE_PLAN.json')
    sources={r['unit_id']:r for r in rows(OUT/'ELIGIBLE_PAIRS.jsonl')}
    consumed=set(read(OUT/'EXPOSURE_MANIFEST.json')['consumed_group_ids'])
    require(len(queries)==d['queries']==len(groups)==len({g['group_id'] for g in groups}),'Sample/group denominator')
    require(len(corpus)==d['corpus_documents']==len(sample['corpus_source_unit_ids']),'Corpus denominator')
    with (OLD/'reference_fingerprints.pkl').open('rb') as h:seen=pickle.load(h)
    corpus_groups=set();positions={}
    for i,(doc,unit) in enumerate(zip(corpus,sample['corpus_source_unit_ids'])):
        source=sources[unit]
        require(doc=={'doc_id':i,'code_text':source['code_text']},'Corpus source identity')
        require(source['group_id'] not in consumed,'Previously consumed component in corpus')
        require(source['group_id'] not in corpus_groups,'Corpus components repeated')
        corpus_groups.add(source['group_id']);positions[unit]=i
        require(not any(set(v)&seen[k] for k,v in source['fingerprints'].items()),'Recovered training/reference overlap')
    del seen;gc.collect()
    qrels={r['query_id']:r['relevant_doc_ids'] for r in rows(OUT/'private/qrels.jsonl')}
    require(set(qrels)=={q['query_id'] for q in queries},'Qrels coverage')
    expected=sorted({r['group_id'] for r in sources.values()},key=lambda g:digest('weight-confirm-query/'+g))[:len(queries)]
    require([r['group_id'] for r in groups]==expected==sample['query_group_ids'],'Outcome-blind component sample replay')
    for q,g in zip(queries,groups):
        source=sources[q['query_id']]
        require(q=={'query_id':source['unit_id'],'query':source['query']},'Public query/source identity')
        require(g['group_id']==source['group_id'] and g['query_id']==q['query_id'],'Matched source/group identity')
        require(qrels[q['query_id']]==[positions[q['query_id']]],'Accepted snippet mapping')
    del sources;gc.collect()
    requested={q['query'] for q in queries};candidate_counts={}
    query_map={q['query_id']:q['query'] for q in queries}
    for arm in WEIGHTS:
        by=collections.defaultdict(list)
        for r in rows(OUT/f'candidates/{arm}.jsonl'):by[r['query_id']].append(r)
        actions={r['query_id']:r for r in rows(OUT/f'actions/{arm}.jsonl')}
        require(set(by)==set(query_map)==set(actions),'Candidate/action coverage')
        for qid,pool in by.items():
            words=query_map[qid].split()
            require(sorted(r['position'] for r in pool)==list(range(1,len(words)+1)),'Candidate positions')
            for r in pool:
                p=r['position'];span=r['span']
                require(r['masked_query']==' '.join(words[:p]+['<extra_id_0>']+words[p:]),'Masked input contract')
                require(r['rewrite']==' '.join(words[:p]+([span] if span else [])+words[p:]),'Rewrite contract')
                require(math.isfinite(r['full_vocab_entropy']) and math.isfinite(r['label_mean_log_probability']),'Finite selector scores')
                requested.add(r['rewrite'])
            for policy,key in [('ENTR',lambda r:(r['full_vocab_entropy'],r['position'])),('PROB',lambda r:(-r['label_mean_log_probability'],r['position']))]:
                require(actions[qid]['orders'][policy]==[r['position'] for r in sorted(pool,key=key)],'Label-free score/position ordering')
        candidate_counts[arm]=sum(map(len,by.values()));del by,actions;gc.collect()
    ranked=set()
    for r in rows(OUT/'rankings/CodeBERT.jsonl'):
        require(r['text'] not in ranked,'Repeated ranking text');ranked.add(r['text'])
        ids=r['ranked_doc_ids'];scores=r['scores']
        require(len(ids)==len(scores)==100 and len(set(ids))==100,'RR@100 depth/uniqueness')
        require(all(0<=i<len(corpus) for i in ids),'Ranking IDs')
        require(all(math.isfinite(v) for v in scores),'Ranking finite')
        require(all(scores[i]>scores[i+1] or scores[i]==scores[i+1] and ids[i]<ids[i+1] for i in range(99)),'Ranking score/tie order')
    require(ranked==requested,'Ranking coverage')
    write(OUT/'PREOPEN_AUDIT.json',{'created_utc':now(),'passed':True,'queries':len(queries),'independent_components':len(groups),
          'corpus_documents':len(corpus),'candidate_positions':candidate_counts,'ranked_texts':len(ranked),
          'all_corpus_components_unconsumed':True,'source_mapping_and_reference_overlap_checked':True,
          'sample_hash_selection_replayed':True,'new_effects_computed':False,'design_sha256':sha(OUT/'DESIGN.json')})
    print({'preopen_passed':True,'queries':len(queries),'candidates':candidate_counts},flush=True)


def cdf_max(values,k):
    k=min(k,len(values));den=math.comb(len(values),k)
    previous=Fraction(0);total=Fraction(0)
    for level in sorted(set(values)):
        count=sum(v<=level for v in values)
        cumulative=Fraction(math.comb(count,k),den) if count>=k else Fraction(0)
        total+=level*(cumulative-previous);previous=cumulative
    return total


def effect():
    verify_design(private=True);result=read(OUT/'RESULTS.json')
    require(result['design_sha256']==sha(OUT/'DESIGN.json'),'Result design hash')
    queries=list(rows(OUT/'public/queries.jsonl'));qrels={r['query_id']:set(r['relevant_doc_ids']) for r in rows(OUT/'private/qrels.jsonl')}
    requested=collections.defaultdict(set)
    for q in queries:requested[q['query']].update(qrels[q['query_id']])
    for arm in WEIGHTS:
        for r in rows(OUT/f'candidates/{arm}.jsonl'):requested[r['rewrite']].update(qrels[r['query_id']])
    lookup={}
    for r in rows(OUT/'rankings/CodeBERT.jsonl'):
        ranks={v:Fraction(1,i+1) for i,v in enumerate(r['ranked_doc_ids'])}
        lookup[r['text']]={v:ranks.get(v,Fraction(0)) for v in requested[r['text']]}
    del requested;gc.collect()
    exported={(r['view'],r['query_id']):r for r in rows(OUT/'per_query.jsonl')}
    independently={};max_error=0.;cells=0
    for arm in WEIGHTS:
        by=collections.defaultdict(list)
        for r in rows(OUT/f'candidates/{arm}.jsonl'):by[r['query_id']].append(r)
        for q in queries:
            qid=q['query_id'];target=next(iter(qrels[qid]));pool=by[qid]
            rr={r['position']:lookup[r['rewrite']][target] for r in pool}
            uniform=sum(rr.values(),Fraction(0))/len(rr)
            values={'NoRewrite':lookup[q['query']][target]}
            for k in (1,2,3):
                values[f'RAND_exact_k{k}_mean']=uniform;values[f'RAND_exact_k{k}_max']=cdf_max(list(rr.values()),k)
                for policy,key in [('ENTR',lambda r:(r['full_vocab_entropy'],r['position'])),('PROB',lambda r:(-r['label_mean_log_probability'],r['position']))]:
                    chosen=[rr[r['position']] for r in sorted(pool,key=key)[:min(k,len(pool))]]
                    values[f'{policy}_k{k}_mean']=sum(chosen,Fraction(0))/len(chosen)
                    values[f'{policy}_k{k}_max']=max(chosen)
            contrasts={}
            for p in ('ENTR','PROB'):
                for k in (1,2,3):
                    for a in ('mean','max'):contrasts[f'S_{p}_k{k}_{a}']=values[f'{p}_k{k}_{a}']-values[f'RAND_exact_k{k}_{a}']
                contrasts[p+'_B']=contrasts[f'S_{p}_k1_mean']-contrasts[f'S_{p}_k3_mean']
                contrasts[p+'_G']=contrasts[f'S_{p}_k3_mean']-contrasts[f'S_{p}_k3_max']
                contrasts[p+'_J']=contrasts[p+'_B']+contrasts[p+'_G']
            out=exported[arm,qid]
            for section,v in [('values',values),('contrasts',contrasts)]:
                for key,value in v.items():
                    error=abs(float(value)-out[section][key]);max_error=max(max_error,error);cells+=1
                    require(error<2e-12,'Independent rational factorial mismatch')
            independently[arm,qid]={k:float(v) for k,v in contrasts.items()}
        del by;gc.collect()
    paired={r['query_id']:r for r in rows(OUT/'paired_per_query.jsonl')}
    matrix=[]
    for q in queries:
        qid=q['query_id'];row=[]
        for name in PRIMARY:
            view,key=name.split('::')
            if view=='base_minus_local':
                value=independently['t5base',qid][key]-independently['local',qid][key]
                require(abs(value-paired[qid]['contrasts'][key])<2e-12,'Paired weight difference identity')
            else:value=independently[view,qid][key]
            row.append(value)
        matrix.append(row)
    x=np.asarray(matrix);n=len(x);mu=x.mean(0);se=np.std(x,axis=0,ddof=1)/math.sqrt(n)
    critical=t.ppf(1-.05/12,n-1)
    boot=np.empty((REPETITIONS,6));rng=np.random.default_rng(SEED)
    for start in range(0,REPETITIONS,16):
        ix=rng.integers(0,n,(min(16,REPETITIONS-start),n));boot[start:start+len(ix)]=x[ix].mean(1)
    quant=np.quantile(boot,[.05/12,1-.05/12],axis=0)
    for i,name in enumerate(PRIMARY):
        out=result['primary'][name]
        independent_t=[float(mu[i]-critical*se[i]),float(mu[i]+critical*se[i])]
        require(abs(out['mean']-mu[i])<2e-12 and abs(out['se']-se[i])<2e-12,'Independent mean/SE')
        require(np.max(np.abs(np.array(out['family6_ci_t'])-independent_t))<2e-12,'Independent corrected t interval')
        require(np.max(np.abs(np.array(out['family6_ci_bootstrap'])-quant[:,i]))<2e-12,'Independent paired bootstrap interval')
        lo=min(independent_t[0],quant[0,i]);hi=max(independent_t[1],quant[1,i])
        nonzero=int(np.count_nonzero(np.abs(x[:,i])>1e-12))
        direction='positive' if lo>0 and nonzero>=20 else 'negative' if hi<0 and nonzero>=20 else 'unresolved'
        require(direction==out['direction'] and nonzero==out['groups_with_nonzero'],'Independent decision/nonzero group count')
    write(OUT/'NUMERICAL_AUDIT.json',{'created_utc':now(),'passed':True,'independent_fraction_cells':cells,
          'maximum_float_error':max_error,'paired_queries':len(queries),'six_primary_t_and_full_bootstrap_replayed':True,
          'bootstrap_repetitions':REPETITIONS,'results':record(OUT/'RESULTS.json')})
    print({'numerical_audit_passed':True,'cells':cells,'max_error':max_error},flush=True)


def model():
    verify_design();verify_seal('RANKINGS_SEALED.json')
    from .inference import setup
    from experiments.rene_ssqr_alignment_20261019.pipeline import load_trained_retriever
    from experiments.rene_ssqr_alignment_20261019.retriever import encode_texts
    torch=setup()
    texts=read(OUT/'retrieval_cache/texts.json')
    selected=sorted(range(len(texts)),key=lambda i:digest('weight-rank-model-audit/'+texts[i]))[:32]
    cv=np.load(OUT/'retrieval_cache/code_vectors.npy',mmap_mode='r');qv=np.load(OUT/'retrieval_cache/query_vectors.npy',mmap_mode='r')
    model,tok=load_trained_retriever(ROOT/'models/codebert-ssqr-pooler-2epoch-20261019','cuda')
    corpus=list(rows(OUT/'public/corpus.jsonl'))
    docs=sorted(range(len(corpus)),key=lambda i:digest('weight-code-model-audit/'+str(i)))[:32]
    newq=encode_texts(model,tok,[texts[i] for i in selected],batch_size=32)
    newc=encode_texts(model,tok,[corpus[i]['code_text'] for i in docs],batch_size=32)
    qerror=float(np.max(np.abs(newq-qv[selected])));cerror=float(np.max(np.abs(newc-cv[docs])))
    require(qerror<5e-5 and cerror<5e-5,'Retriever reload vector discrepancy')
    del model;gc.collect();torch.cuda.empty_cache()
    requested={texts[i] for i in selected};ranks={r['text']:r for r in rows(OUT/'rankings/CodeBERT.jsonl') if r['text'] in requested}
    ids=np.arange(len(corpus));max_score_error=0.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        # Replay exact cached batch shape. A different BLAS output width can change
        # FP32 last bits and ties; this reproduces the prespecified batch128 contract.
        for index in selected:
            start=(index//128)*128;score=(qv[start:start+128]@cv.T)[index-start]
            order=np.lexsort((ids,-score))[:100];saved=ranks[texts[index]]
            require(order.tolist()==saved['ranked_doc_ids'],'Full stable-sort independent ranking replay')
            max_score_error=max(max_score_error,float(np.max(np.abs(score[order]-saved['scores']))))
    require(max_score_error<2e-5,'Cached full-corpus score discrepancy')
    write(OUT/'MODEL_AUDIT.json',{'created_utc':now(),'passed':True,'query_vectors_reloaded':32,'code_vectors_reloaded':32,
          'max_query_vector_error':qerror,'max_code_vector_error':cerror,'full_corpus_ranking_replays':32,
          'max_score_error':max_score_error,'ranking_ties_replayed_with_full_lexsort':True})
    print({'model_audit_passed':True,'qerror':qerror,'cerror':cerror},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preopen','effect','model']);args=parser.parse_args()
    {'preopen':preopen,'effect':effect,'model':model}[args.mode]()
