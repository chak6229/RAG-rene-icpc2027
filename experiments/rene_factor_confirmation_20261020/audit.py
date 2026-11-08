"""Independent source/contract and numerical checks, no production metric import."""
from __future__ import annotations

import argparse
import collections
import itertools
import math
import pickle
from fractions import Fraction

import numpy as np
from scipy.stats import t

from .common import ROOT, OUT, SOURCES, digest, fingerprints, now, read, record, rows, sha, write
from .inference import verify_design, verify_seal
from .precision import CONTRASTS


def preopen():
    d=verify_design(private=True)
    verify_seal('CANDIDATES_SEALED.json');verify_seal('RANKINGS_SEALED.json')
    if (OUT/'OUTCOME_OPENING.json').exists():
        raise ValueError('Preopening audit must precede effects')
    queries=list(rows(OUT/'public/queries.jsonl'));corpus=list(rows(OUT/'public/corpus.jsonl'))
    groups=list(rows(OUT/'public/groups.jsonl'));sample=read(OUT/'SAMPLE_PLAN.json')
    sources={r['unit_id']:r for r in rows(OUT/'STAQC_ELIGIBLE_PAIRS.jsonl')}
    with (OUT/'reference_fingerprints.pkl').open('rb') as h:seen=pickle.load(h)
    def require(condition,message):
        if not condition:
            raise ValueError(message)
    require(len(queries)==d['queries']==len(groups),'Sample denominator')
    require(len({r['group_id'] for r in groups})==len(groups),'Connected group reuse')
    for q,g in zip(queries,groups):
        source=sources[q['query_id']]
        require(set(q)=={'query_id','query'} and q['query']==source['query'],'Public query-source mismatch')
        require(g['query_id']==q['query_id'] and g['group_id']==source['group_id'],'Group mismatch')
    for doc,unit in zip(corpus,sample['corpus_source_unit_ids']):
        source=sources[unit]
        require(doc['code_text']==source['code_text'],'Corpus-source mismatch')
        require(not any(set(v)&seen[k] for k,v in source['fingerprints'].items()),'Known reference overlap in corpus')
    qrels={r['query_id']:r['relevant_doc_ids'] for r in rows(OUT/'private/qrels.jsonl')}
    require(set(qrels)=={q['query_id'] for q in queries},'Qrels denominator')
    units=sample['corpus_source_unit_ids']
    for q in queries:require(qrels[q['query_id']]==[units.index(q['query_id'])],'Accepted answer mapping')
    candidates=collections.defaultdict(list)
    for r in rows(OUT/'candidates.jsonl'):candidates[r['query_id']].append(r)
    actions={r['query_id']:r for r in rows(OUT/'actions.jsonl')}
    require(set(candidates)==set(qrels)==set(actions),'Candidate/action coverage')
    for q in queries:
        pool=candidates[q['query_id']]
        require({r['position'] for r in pool}==set(range(1,len(q['query'].split())+1)),'Missing position')
        require(len(pool)==len({r['position'] for r in pool}),'Duplicated position')
        for r in pool:
            words=q['query'].split();p=r['position']
            require(r['rewrite']==' '.join(words[:p]+([r['span']] if r['span'] else [])+words[p:]),'Rewrite reconstruction')
        for policy,key in (('ENTR',lambda r:(r['full_vocab_entropy'],r['position'])),('PROB',lambda r:(-r['label_mean_log_probability'],r['position']))):
            require(actions[q['query_id']]['orders'][policy]==[r['position'] for r in sorted(pool,key=key)],'Selection order')
    requested={q['query'] for q in queries}|{r['rewrite'] for pool in candidates.values() for r in pool}
    rank_counts={}
    for ret in ('CodeBERT','BM25'):
        ranked=list(rows(OUT/f'rankings/{ret}.jsonl'))
        require({r['text'] for r in ranked}==requested and len(ranked)==len(requested),'Ranking text coverage')
        for r in ranked:
            ids=r['ranked_doc_ids'];scores=r['scores']
            require(len(ids)==100 and len(set(ids))==100 and len(scores)==100,'Ranking depth/uniqueness')
            require(all(0<=i<len(corpus) for i in ids),'Ranking doc ID outside corpus')
            require(all((scores[i]>scores[i+1] or scores[i]==scores[i+1] and ids[i]<ids[i+1]) for i in range(99)),'Ranking score/tie order')
        rank_counts[ret]=len(ranked)
    for r in read(SOURCES/'staqc_DATA_RECEIPTS.json')['files']:
        path=ROOT/r['path'];data=path.read_bytes()
        import hashlib
        require(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==r['git_blob_sha1'],'Author source Git identity')
    write(OUT/'PREOPEN_AUDIT.json',{'created_utc':now(),'passed':True,'new_effects_computed':False,
          'queries':len(queries),'connected_groups':len(groups),'corpus':len(corpus),
          'candidates':sum(map(len,candidates.values())),'ranked_texts':rank_counts,
          'source_mapping_and_known_exposure_checked':True,'author_git_blobs_checked':True,
          'design_sha256':sha(OUT/'DESIGN.json'),'scope':'full query/corpus identity, group independence, known overlap, labels mapping, candidate/action and exported ranking structural audit; not new RR effects or all full-corpus scores'})
    print({'preopen_passed':True,'queries':len(queries)},flush=True)


def exact_expected_max(values,k):
    # Independent CDF identity. Unlike the production order-statistic mass formula,
    # sum differences of CDFs at each distinct observed reciprocal-rank value.
    k=min(k,len(values));den=math.comb(len(values),k)
    levels=sorted(set(values));previous=Fraction(0);total=Fraction(0)
    for level in levels:
        n_le=sum(v<=level for v in values)
        cdf=Fraction(math.comb(n_le,k),den) if n_le>=k else Fraction(0)
        total+=level*(cdf-previous);previous=cdf
    return total


def effect():
    d=verify_design(private=True);result=read(OUT/'RESULTS.json')
    exported={(r['retriever'],r['query_id']):r for r in rows(OUT/'per_query.jsonl')}
    queries=list(rows(OUT/'public/queries.jsonl'));qrels={r['query_id']:set(r['relevant_doc_ids']) for r in rows(OUT/'private/qrels.jsonl')}
    candidates=collections.defaultdict(list)
    for r in rows(OUT/'candidates.jsonl'):candidates[r['query_id']].append(r)
    maximum=0.;rr_count=0;cells=0;independent=[]
    for ret in ('CodeBERT','BM25'):
        ranks={r['text']:r['ranked_doc_ids'] for r in rows(OUT/f'rankings/{ret}.jsonl')}
        for q in queries:
            qid=q['query_id'];pool=candidates[qid];rel=qrels[qid]
            def rank_rr(text):
                nonlocal rr_count
                rr_count+=1
                return next((Fraction(1,i+1) for i,v in enumerate(ranks[text][:100]) if v in rel),Fraction(0))
            rrs={r['position']:rank_rr(r['rewrite']) for r in pool}
            base=rank_rr(q['query']);uniform=sum(rrs.values(),Fraction(0))/len(rrs)
            vals={'NoRewrite':base}
            for k in (1,2,3):
                m=min(k,len(pool));random_max=exact_expected_max(list(rrs.values()),m)
                vals.update({f'RAND_exact_k{k}_first':uniform,f'RAND_exact_k{k}_mean':uniform,f'RAND_exact_k{k}_max':random_max})
                for policy,key in (('ENTR',lambda r:(r['full_vocab_entropy'],r['position'])),('PROB',lambda r:(-r['label_mean_log_probability'],r['position']))):
                    selected=[rrs[r['position']] for r in sorted(pool,key=key)[:m]]
                    vals.update({f'{policy}_k{k}_first':selected[0],f'{policy}_k{k}_mean':sum(selected,Fraction(0))/m,f'{policy}_k{k}_max':max(selected)})
            out=exported[ret,qid]
            for key,v in vals.items():
                err=abs(float(v)-out['values'][key]);maximum=max(maximum,err);cells+=1
                if err>2e-13:raise ValueError('Independent factorial mismatch')
            factors={}
            for p in ('ENTR','PROB'):
                s1=vals[f'{p}_k1_mean']-vals['RAND_exact_k1_mean']
                s3m=vals[f'{p}_k3_mean']-vals['RAND_exact_k3_mean']
                s3x=vals[f'{p}_k3_max']-vals['RAND_exact_k3_max']
                factors[p+'_B_budget_mean_1_to_3']=s1-s3m
                factors[p+'_G_aggregation_at3_mean_to_max']=s3m-s3x
                factors[p+'_J_exact']=s1-s3x
            for c in ('B_budget_mean_1_to_3','G_aggregation_at3_mean_to_max'):
                factors['ENTR_minus_PROB_'+c]=factors['ENTR_'+c]-factors['PROB_'+c]
            for c,v in factors.items():
                err=abs(float(v)-out['contrasts'][c]);maximum=max(maximum,err);cells+=1
                if err>2e-13:raise ValueError('Independent B/G mismatch')
            independent.append({'retriever':ret,'query_id':qid,'group':out['group'],'factors':{k:float(v) for k,v in factors.items()}})
        del ranks
    # Exactly one question per component makes the independent cluster-t check an
    # ordinary sample-variance calculation; do not import production summary code.
    for ret in ('CodeBERT','BM25'):
        current=sorted((r for r in independent if r['retriever']==ret),key=lambda r:r['group'])
        if len({r['group'] for r in current})!=len(current):raise ValueError('Dependent components reused')
        names=list(current[0]['factors'])
        x=np.array([[r['factors'][c] for c in names] for r in current])
        mu=x.mean(0);se=x.std(0,ddof=1)/math.sqrt(len(x))
        for j,c in enumerate(names):
            target=result['views'][ret]['summaries'][c]
            for a,b in ((mu[j],target['mean']),(se[j],target['se_cluster'])):
                maximum=max(maximum,abs(float(a)-b))
                if abs(float(a)-b)>1e-12:raise ValueError('Independent summary mismatch')
            ci=mu[j]+np.array([-1,1])*t.ppf(.9875,len(x)-1)*se[j]
            if ret=='CodeBERT' and c in CONTRASTS:
                if np.max(np.abs(ci-target['family2_ci95_cluster_t']))>1e-12:raise ValueError('Independent corrected t interval mismatch')
        # Separate bootstrap implementation: same group draws, direct sample means.
        # Restrict to the two primaries to keep the independent audit bounded.
        xp=np.array([[r['factors'][c] for c in CONTRASTS] for r in current])
        rng=np.random.default_rng(202610202);boot=np.empty((10000,2))
        for start in range(0,10000,32):
            ix=rng.integers(0,len(xp),(min(32,10000-start),len(xp)))
            boot[start:start+len(ix)]=xp[ix].mean(axis=1)
        if ret=='CodeBERT':
            quant=np.quantile(boot,[.0125,.9875],axis=0)
            for j,c in enumerate(CONTRASTS):
                err=float(np.max(np.abs(quant[:,j]-result['views'][ret]['summaries'][c]['family2_ci95_group_bootstrap'])))
                maximum=max(maximum,err)
                if err>1e-12:raise ValueError('Independent corrected bootstrap interval mismatch')
    write(OUT/'EFFECT_AUDIT.json',{'created_utc':now(),'passed':True,'independent_no_production_metrics':True,
          'Fraction_RR_checks':rr_count,'independent_factorial_and_factor_fields':cells,
          'corrected_primary_intervals_recomputed':True,'maximum_absolute_error':maximum,
          'results':record(OUT/'RESULTS.json'),'per_query':record(OUT/'per_query.jsonl')})
    print({'independent_effect_audit':True,'RR_checks':rr_count,'max_error':maximum},flush=True)


def models():
    d=verify_design()
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import load_tokenizer
    from experiments.rene_ssqr_alignment_20261019.pipeline import load_trained_retriever
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    candidates=list(rows(OUT/'candidates.jsonl'))
    chosen=sorted(candidates,key=lambda r:digest('factor-model-audit/'+r['query_id']+'/'+str(r['position'])))[:16]
    path=ROOT/d['generation']['checkpoint'];tok=load_tokenizer(path)
    model=AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=True).float().cuda().eval()
    err_h=0.;err_lp=0.;recreated=0
    with torch.inference_mode():
        for r in chosen:
            ids=tok([r['masked_query']],return_tensors='pt');ids={k:v.cuda() for k,v in ids.items()}
            label=tok([f'<extra_id_0> {r["span"]} <extra_id_1>'],return_tensors='pt')
            labels=label.input_ids.cuda();attention=label.attention_mask.cuda().bool()
            labels=labels.masked_fill(~attention,-100)
            logit=model(**ids,labels=labels,decoder_attention_mask=attention).logits[0].double()
            logp=logit-torch.logsumexp(logit,dim=-1,keepdim=True)
            entropy=float((-(torch.exp(logp)*logp).sum(-1)).mean().cpu())
            lp=float(logp.gather(1,labels[0,:,None]).mean().cpu())
            err_h=max(err_h,abs(entropy-r['full_vocab_entropy']))
            err_lp=max(err_lp,abs(lp-r['label_mean_log_probability']))
            out=model.generate(**ids,do_sample=False,num_beams=1,num_return_sequences=1,max_new_tokens=19,
                decoder_start_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True)
            span=tok.decode(out[0],skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()
            if span!=r['span']:raise ValueError('Independent generation changed')
            recreated+=1
    del model;torch.cuda.empty_cache()
    if err_h>3e-5 or err_lp>5e-5:raise ValueError('T5 FP64 audit tolerance exceeded')
    model,tok=load_trained_retriever(ROOT/'models/codebert-ssqr-pooler-2epoch-20261019','cuda')
    texts=read(OUT/'retrieval_cache/CodeBERT/texts.json');qv=np.load(OUT/'retrieval_cache/CodeBERT/query_vectors.npy')
    cv=np.load(OUT/'retrieval_cache/CodeBERT/code_vectors.npy');corpus=list(rows(OUT/'public/corpus.jsonl'))
    vector_max=0.;checked=0
    for values,cached in ((texts,qv),([r['code_text'] for r in corpus],cv)):
        indices=sorted(range(len(values)),key=lambda i:digest('factor-cb-audit/'+str(i)))[:16]
        for i in indices:
            tokens=tok([values[i]],padding='max_length',truncation=True,max_length=256,return_tensors='pt')
            with torch.inference_mode():
                v=model(**{k:val.cuda() for k,val in tokens.items()}).pooler_output[0].float().cpu().numpy()
            error=float(np.max(np.abs(v-cached[i])));vector_max=max(vector_max,error);checked+=1
    del model;torch.cuda.empty_cache()
    if vector_max>5e-5:raise ValueError('CodeBERT vector audit tolerance exceeded')
    # Recompute exported dense scores against every corpus document for a fixed
    # outcome-blind text sample, independently of target labels and ranking code.
    from threadpoolctl import threadpool_limits
    ranks={r['text']:r for r in rows(OUT/'rankings/CodeBERT.jsonl')}
    selected=sorted(range(len(texts)),key=lambda i:digest('factor-dense-audit/'+str(i)))[:64]
    score_max=0.;ambiguous_rows=0
    with threadpool_limits(limits=4):
        full=qv[selected]@cv.T
    for i,values in zip(selected,full):
        row=ranks[texts[i]];ids=np.asarray(row['ranked_doc_ids'])
        err=float(np.max(np.abs(values[ids]-row['scores'])));score_max=max(score_max,err)
        if err>3e-4:raise ValueError('Dense score export mismatch')
        cutoff=float(values[ids[-1]])
        omitted=np.ones(len(cv),dtype=bool);omitted[ids]=False
        if float(values[omitted].max())>cutoff+3e-4:raise ValueError('Strictly better document omitted from Top100')
        exact=sorted(range(len(values)),key=lambda j:(-float(values[j]),j))[:100]
        if exact!=ids.tolist():
            # FP32 GEMM batch shape may change near ties. Count those explicitly;
            # all deviations must be within the stated score tolerance.
            ambiguous_rows+=1
            for a,b in zip(exact,ids):
                if abs(float(values[a])-float(values[b]))>3e-4:raise ValueError('Dense ordering mismatch beyond FP32 ties')
    write(OUT/'MODEL_AUDIT.json',{'created_utc':now(),'passed':True,'t5_actual_generation_cases':recreated,
          't5_entropy_independent_FP64_cases':len(chosen),'maximum_entropy_error':err_h,'maximum_log_probability_error':err_lp,
          'actual_CodeBERT_vectors':checked,'maximum_vector_error':vector_max,
          'dense_full_corpus_score_rows':len(selected),'dense_score_comparisons':len(selected)*len(cv),
          'maximum_dense_export_error':score_max,'rows_with_FP32_near_tie_order_differences':ambiguous_rows,
          'scope':'bounded real-model samples and64full-corpus score rows, not full regeneration/re-encoding; full RR/factor replay independently audited'})
    print({'real_model_audit':True,'T5':recreated,'CodeBERT':checked},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preopen','effect','models']);a=p.parse_args()
    {'preopen':preopen,'effect':effect,'models':models}[a.stage]()


if __name__=='__main__':main()
