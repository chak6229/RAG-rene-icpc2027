from __future__ import annotations

import collections
import json
import math
import time

import numpy as np
from scipy.stats import t

from .common import ROOT, OUT, now, read, record, rows, sha, write, writel
from .inference import verify_design, verify_seal, progress
from .precision import CONTRASTS


def summary(matrix, names, groups, repetitions=10000):
    x=np.asarray(matrix,dtype=float)
    labels=sorted(set(groups));lookup={g:i for i,g in enumerate(labels)}
    codes=np.array([lookup[g] for g in groups]);sizes=np.bincount(codes).astype(float)
    sums=np.zeros((len(labels),len(names)));np.add.at(sums,codes,x)
    mu=x.mean(0);se=np.sqrt(len(labels)/(len(labels)-1)*np.square(sums-sizes[:,None]*mu).sum(0)/len(x)**2)
    rng=np.random.default_rng(202610202);boot=np.empty((repetitions,len(names)))
    for start in range(0,repetitions,32):
        ix=rng.integers(0,len(labels),(min(32,repetitions-start),len(labels)))
        boot[start:start+len(ix)]=sums[ix].sum(1)/sizes[ix].sum(1)[:,None]
    quant=np.quantile(boot,[.025,.975,.0125,.9875],axis=0)
    out={}
    changed=np.zeros((len(labels),len(names)),dtype=bool);np.logical_or.at(changed,codes,np.abs(x)>1e-12)
    for i,name in enumerate(names):
        c=float(t.ppf(.975,len(labels)-1));f=float(t.ppf(.9875,len(labels)-1))
        r={'mean':float(mu[i]),'se_cluster':float(se[i]),'ci95_cluster_t':[float(mu[i]-c*se[i]),float(mu[i]+c*se[i])],
           'ci95_group_bootstrap':quant[:2,i].tolist(),'queries':len(x),'groups':len(labels),
           'changed_queries':int(np.count_nonzero(np.abs(x[:,i])>1e-12)), 'groups_with_nonzero':int(changed[:,i].sum())}
        if name in CONTRASTS:
            r.update(family2_ci95_cluster_t=[float(mu[i]-f*se[i]),float(mu[i]+f*se[i])],
                     family2_ci95_group_bootstrap=quant[2:,i].tolist())
            lo=min(r['family2_ci95_cluster_t'][0],r['family2_ci95_group_bootstrap'][0])
            hi=max(r['family2_ci95_cluster_t'][1],r['family2_ci95_group_bootstrap'][1])
            r['direction']='positive' if lo>0 and r['groups_with_nonzero']>=20 else 'negative' if hi<0 and r['groups_with_nonzero']>=20 else 'unresolved'
        out[name]=r
    return out


def rr(order,relevant):
    return next((1./(i+1) for i,d in enumerate(order[:100]) if d in relevant),0.)


def main():
    d=verify_design(private=True);verify_seal('CANDIDATES_SEALED.json');verify_seal('RANKINGS_SEALED.json')
    pre=read(OUT/'PREOPEN_AUDIT.json')
    if not pre['passed']:
        raise ValueError('Independent preopening audit did not pass')
    if (OUT/'RESULTS.json').exists():
        raise FileExistsError('No repeated outcome analysis')
    if not (OUT/'OUTCOME_OPENING.json').exists():
        write(OUT/'OUTCOME_OPENING.json',{'created_utc':now(),'design_sha256':sha(OUT/'DESIGN.json'),
              'candidate_seal_sha256':sha(OUT/'CANDIDATES_SEALED.json'), 'ranking_seal_sha256':sha(OUT/'RANKINGS_SEALED.json'),
              'preopening_audit':record(OUT/'PREOPEN_AUDIT.json'),'scope':'one opening of prespecified B/G and descriptive secondary effects; no fitting/policy search'})
    started=time.perf_counter()
    from experiments.rene_budget_aggregation_20261020.metrics import factorial_values, contrasts, exact_random_max
    queries=list(rows(OUT/'public/queries.jsonl'))
    groups={r['query_id']:r['group_id'] for r in rows(OUT/'public/groups.jsonl')}
    qrels={r['query_id']:set(r['relevant_doc_ids']) for r in rows(OUT/'private/qrels.jsonl')}
    actions={r['query_id']:r for r in rows(OUT/'actions.jsonl')}
    candidates=collections.defaultdict(list)
    for r in rows(OUT/'candidates.jsonl'):candidates[r['query_id']].append(r)
    output=[];views={}
    for ret in ('CodeBERT','BM25'):
        ranking={r['text']:r['ranked_doc_ids'] for r in rows(OUT/f'rankings/{ret}.jsonl')}
        current=[]
        for q in queries:
            qid=q['query_id'];rel=qrels[qid];pool=candidates[qid]
            scores={r['position']:rr(ranking[r['rewrite']],rel) for r in pool}
            base=rr(ranking[q['query']],rel)
            v=factorial_values(scores,actions[qid]['orders'],base,exact_random_max(list(scores.values()),3))
            c=contrasts(v)
            # No Monte-Carlo RAND20 is run in this study. The compatibility argument
            # above equals exact3 and its legacy alias is removed from every export.
            v={k:val for k,val in v.items() if 'RAND20' not in k}
            c={k:val for k,val in c.items() if 'RAND20' not in k}
            r={'retriever':ret,'query_id':qid,'group':groups[qid],'values':v,'contrasts':c,
               'position_rr':{str(k):val for k,val in scores.items()}}
            current.append(r);output.append(r)
        names=list(CONTRASTS)+['PROB_B_budget_mean_1_to_3','PROB_G_aggregation_at3_mean_to_max',
             'ENTR_minus_PROB_B_budget_mean_1_to_3','ENTR_minus_PROB_G_aggregation_at3_mean_to_max',
             'ENTR_J_exact','PROB_J_exact']
        names += [f'S_{p}_k{k}_{a}' for p in ('ENTR','PROB') for k in (1,2,3) for a in ('mean','max')]
        names += [f'delta_{p}_k{k}_{a}' for p in ('ENTR','PROB','RAND_exact') for k in (1,2,3) for a in ('mean','max')]
        matrix=[[r['contrasts'][n] for n in names] for r in current]
        progress('statistics',retriever=ret,queries=len(current),groups=len(set(groups.values())))
        estimates=summary(matrix,names,[r['group'] for r in current])
        views[ret]={'summaries':estimates, 'raw_means':{k:float(np.mean([r['values'][k] for r in current])) for k in current[0]['values']},
                    'primary_role':ret=='CodeBERT'}
        if ret=='BM25':
            for c in CONTRASTS:
                # This retriever is descriptive; do not accidentally confer extra
                # primary status merely because it uses the same contrast names.
                for k in ('family2_ci95_cluster_t','family2_ci95_group_bootstrap','direction'):
                    estimates[c].pop(k,None)
        del ranking
    by={(r['retriever'],r['query_id']):r for r in output}
    diff=summary([[by['CodeBERT',q['query_id']]['contrasts'][c]-by['BM25',q['query_id']]['contrasts'][c] for c in CONTRASTS] for q in queries],
                 ['CodeBERT_minus_BM25_'+c for c in CONTRASTS],[groups[q['query_id']] for q in queries])
    writel(OUT/'per_query.jsonl',output)
    write(OUT/'RESULTS.json',{'created_utc':now(),'design_sha256':sha(OUT/'DESIGN.json'), 'opening':record(OUT/'OUTCOME_OPENING.json'),
          'queries':len(queries),'groups':len(set(groups.values())),'corpus_documents':d['corpus_documents'],
          'views':views,'retriever_differences_descriptive':diff,
          'primary_directions':{c:views['CodeBERT']['summaries'][c]['direction'] for c in CONTRASTS},
          'new_setting_confirmation':True,'external_preregistration':False,'CoSQA_confirmation':False,
          'risk_certified':False,'seconds':time.perf_counter()-started})
    print({'primary_directions':{c:views['CodeBERT']['summaries'][c]['direction'] for c in CONTRASTS}},flush=True)


if __name__=='__main__':main()
