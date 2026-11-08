"""Actual extra BM25 evidence, fixed dense retrieval actions and labels.

BM25 is evidence only: never feeds the gold function/rank into inference features.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import joblib
import numpy as np

from build_rewrite_utility_dataset import code_text
from run_uq_pilot import (Hurdle, booster, dump, flatten, grouped_folds,
                          sample_fold, decision_metrics)


def tokens(text):
    text=re.sub(r'([a-z0-9])([A-Z])',r'\1 \2',text)
    return re.findall(r'[a-z][a-z0-9]*',text.lower())


class BM25:
    def __init__(self,documents):
        posting=defaultdict(list);length=[]
        for index,text in enumerate(documents):
            terms=tokens(text);length.append(len(terms))
            for word,count in Counter(terms).items():posting[word].append((index,count))
        self.n=len(documents);length=np.array(length);avg=max(length.mean(),1)
        self.posting={}
        for word,items in posting.items():
            ids,tf=np.array(items).T
            idf=np.log1p((self.n-len(ids)+.5)/(len(ids)+.5))
            contribution=idf*(tf*2.2)/(tf+1.2*(.25+.75*length[ids]/avg))
            self.posting[word]=(ids,contribution)

    def retrieve(self,text):
        scores=np.zeros(self.n)
        for word in set(tokens(text)):
            if word in self.posting:
                ids,weight=self.posting[word];scores[ids]+=weight
        top=np.argpartition(-scores,9)[:10]
        top=top[np.lexsort((top,-scores[top]))]
        return top,scores[top]


def build_extra(rows,index,cache):
    values=[]
    def retrieve(text):
        if text not in cache:cache[text]=index.retrieve(text)
        return cache[text]
    for r in rows:
        oi,os=retrieve(r['query'])
        for c in r['candidates']:
            if not c['quality_pass']:continue
            ci,cs=retrieve(c['text'])
            values.append([os[0],os[0]-os[1],os.mean(),os.std(),
                           cs[0],cs[0]-cs[1],cs.mean(),cs.std(),cs[0]-os[0],
                           len(set(oi)&set(ci))/10,float(oi[0]==ci[0])])
    return np.array(values)


def query_rows(data,pred):
    rows=[];features=[]
    for q in np.unique(data['q']):
        ix=np.flatnonzero(data['q']==q);order=ix[np.argsort(-pred['mu'][ix])];best=order[0]
        features.append(np.r_[data['x'][best],pred['mu'][best],pred['s'][best],
                              pred['mu'][best]-(pred['mu'][order[1]] if len(order)>1 else 0)])
        rows.append((q,best,ix))
    return np.array(features),rows


def run(sampled,extra,action,seed,directory):
    train=flatten(sampled['utility_train'],action);ev=flatten(sampled['new_holdout'],action)
    x,y,repo=train['x'],train['y'],train['repo'];full_x=np.c_[x,extra['utility_train']]
    b_oof=np.zeros(len(y));s_oof=np.zeros(len(y));f_oof=np.zeros(len(y))
    for val in grouped_folds(repo,seed=seed):
        tr=np.setdiff1d(np.arange(len(y)),val)
        b=Hurdle().fit(x[tr],y[tr],seed);f=Hurdle().fit(full_x[tr],y[tr],seed)
        pb=b.predict(x[val]);pf=f.predict(full_x[val])
        b_oof[val]=pb['mu'];s_oof[val]=pb['s'];f_oof[val]=pf['mu']
    meta_x,indices=query_rows(train,dict(mu=b_oof,s=s_oof));target=[]
    for _,best,ix in indices:
        fb=ix[np.argmax(f_oof[ix])]
        target.append((y[fb] if f_oof[fb]>0 else 0)-(y[best] if b_oof[best]>0 else 0))
    controller=booster(meta_x,np.array(target),seed)
    base=Hurdle().fit(x,y,seed);full=Hurdle().fit(full_x,y,seed)
    pb=base.predict(ev['x']);pf=full.predict(np.c_[ev['x'],extra['new_holdout']])
    meta_ev,indices=query_rows(ev,pb);acquisition_score=controller.predict(meta_ev)
    uncertainty=np.array([pb['s'][best] for _,best,_ in indices]);rng=np.random.default_rng(seed)
    rankings=dict(learned_expected_gain=np.argsort(-acquisition_score),
                  uncertainty=np.argsort(-uncertainty),random=rng.permutation(len(indices)))
    result={}
    for mode in ('never','always',*rankings):
        result[mode]={}
        fractions=(0.,) if mode=='never' else (1.,) if mode=='always' else (.2,.5)
        for fraction in fractions:
            acquire=set(range(len(indices))) if mode=='always' else set()
            if mode in rankings:acquire=set(rankings[mode][:round(len(indices)*fraction)].tolist())
            chosen=[]
            for local,(_,best,ix) in enumerate(indices):
                if local in acquire:best=ix[np.argmax(pf['mu'][ix])];mu=pf['mu'][best]
                else:mu=pb['mu'][best]
                if mu>0:chosen.append(best)
            result[mode][str(fraction)]=dict(acquisition_fraction=len(acquire)/ev['query_count'],
                                              **decision_metrics(ev,chosen,seed))
    joblib.dump(dict(base=base,full=full,controller=controller),directory/f'{action}.joblib')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();out=a.directory/'bm25_evidence';out.mkdir(exist_ok=True)
    source=Path('data/codexglue/advtest/valid.jsonl')
    docs=[code_text(json.loads(s)) for s in source.open(encoding='utf-8') if s.strip()]
    started=time.perf_counter();index=BM25(docs);build_seconds=time.perf_counter()-started
    cache={};results={};timing=[]
    for fold,seed in ((0,101),(1,202),(2,303)):
        print(f'BM25 fold {fold}',flush=True)
        sampled=sample_fold(Path(f'results/vac_mqr_v5/enhanced_cv_folds/fold_{fold}.jsonl'),seed)
        t=time.perf_counter();before=len(cache)
        extra={role:build_extra(sampled[role],index,cache) for role in ('utility_train','new_holdout')}
        timing.append(dict(fold=fold,unique_queries=len(cache)-before,
                           evidence_seconds=time.perf_counter()-t))
        results[str(fold)]={action:run(sampled,extra,action,seed,out) for action in ('rewrite','fusion')}
        dump(out/'checkpoint.json',results)
    dump(out/'results.json',dict(test_used=False,development_only=True,results=results,
        bm25_corpus_documents=len(docs),code_only=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        bm25_parameters=dict(k1=1.2,b=.75,tokenization='alphabetic camel/snake fragments'),
        index_build_seconds=build_seconds,evidence_timing=timing,
        caveat='Evidence extracted in batches for evaluation; measured time is not end-to-end serving latency. No risk guarantee.'))
    print('BM25 completed',flush=True)


if __name__=='__main__':main()
