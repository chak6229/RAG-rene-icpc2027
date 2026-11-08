"""One outcome opening; six-comparison paired confirmation and complete curves."""
from __future__ import annotations

import collections
import gc
import math
import time

import numpy as np
from scipy.stats import t

from .common import FAMILY, OUT, PRIMARY, REPETITIONS, ROOT, SEED, WEIGHTS, now, progress, read, record, rows, sha, verify_design, verify_seal, write, writel
from experiments.rene_t5_weight_factors_20261020.metrics import factorial, paired_difference


def summarize(matrix,names,primary=(),repetitions=REPETITIONS):
    x=np.asarray(matrix,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(names) or not np.isfinite(x).all() or len(x)<2:raise ValueError('Invalid statistics input')
    n=len(x);mu=x.mean(0);se=x.std(0,ddof=1)/math.sqrt(n)
    c=float(t.ppf(.975,n-1));fc=float(t.ppf(1-.025/FAMILY,n-1))
    # Shared component draws across weights and differences; one query per group.
    # Fast multinomial block multiplication is algebraically the ordinary group bootstrap.
    rng=np.random.default_rng(SEED)
    boot=np.empty((repetitions,len(names)),dtype=np.float64)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        for start in range(0,repetitions,16):
            ix=rng.integers(0,n,(min(16,repetitions-start),n))
            counts=np.stack([np.bincount(draw,minlength=n) for draw in ix]).astype(np.float64)
            boot[start:start+len(ix)]=counts@x/n
    q=np.quantile(boot,[.025,.975,.025/FAMILY,1-.025/FAMILY],axis=0)
    out={}
    for i,name in enumerate(names):
        r={'mean':float(mu[i]),'se':float(se[i]),'queries':n,'groups':n,
           'groups_with_nonzero':int(np.count_nonzero(np.abs(x[:,i])>1e-12)),
           'ci95_t':[float(mu[i]-c*se[i]),float(mu[i]+c*se[i])], 'ci95_bootstrap':q[:2,i].tolist()}
        if name in primary:
            r['family6_ci_t']=[float(mu[i]-fc*se[i]),float(mu[i]+fc*se[i])]
            r['family6_ci_bootstrap']=q[2:,i].tolist()
            lo=min(r['family6_ci_t'][0],r['family6_ci_bootstrap'][0]);hi=max(r['family6_ci_t'][1],r['family6_ci_bootstrap'][1])
            r['direction']='positive' if lo>0 and r['groups_with_nonzero']>=20 else 'negative' if hi<0 and r['groups_with_nonzero']>=20 else 'unresolved'
        out[name]=r
    return out


def reciprocal(order,relevant):
    return next((1./(i+1) for i,doc in enumerate(order[:100]) if doc in relevant),0.)


def main():
    d=verify_design(private=True);verify_seal('CANDIDATES_SEALED.json');verify_seal('RANKINGS_SEALED.json')
    if not read(OUT/'PREOPEN_AUDIT.json')['passed']:raise ValueError('Preopen audit did not pass')
    if (OUT/'RESULTS.json').exists():raise FileExistsError('Outcome analysis already completed')
    if not (OUT/'OUTCOME_OPENING.json').exists():
        write(OUT/'OUTCOME_OPENING.json',{'created_utc':now(),'design_sha256':sha(OUT/'DESIGN.json'),
              'candidate_seal':record(OUT/'CANDIDATES_SEALED.json'),'ranking_seal':record(OUT/'RANKINGS_SEALED.json'),
              'preopen_audit':record(OUT/'PREOPEN_AUDIT.json'),'scope':'single prespecified paired six-contrast analysis; no tuning, sample extension or optional stopping'})
    started=time.perf_counter()
    queries=list(rows(OUT/'public/queries.jsonl'));groups={r['query_id']:r['group_id'] for r in rows(OUT/'public/groups.jsonl')}
    if len(groups)!=len(set(groups.values())):raise ValueError('Not one query per group')
    qrels={r['query_id']:set(r['relevant_doc_ids']) for r in rows(OUT/'private/qrels.jsonl')}
    requested={q['query'] for q in queries}
    for arm in WEIGHTS:
        for r in rows(OUT/f'candidates/{arm}.jsonl'):requested.add(r['rewrite'])
    # Store only RR lookup, not every ranking for both weights; labels first accessed here.
    text_rr={}
    # Rankings are common, but RR depends on the paired query's target.
    text_targets=collections.defaultdict(set)
    for q in queries:text_targets[q['query']].update(qrels[q['query_id']])
    for arm in WEIGHTS:
        for r in rows(OUT/f'candidates/{arm}.jsonl'):text_targets[r['rewrite']].update(qrels[r['query_id']])
    for r in rows(OUT/'rankings/CodeBERT.jsonl'):
        ranks={doc:1./(i+1) for i,doc in enumerate(r['ranked_doc_ids'][:100])}
        text_rr[r['text']]={target:ranks.get(target,0.) for target in text_targets[r['text']]}
    del text_targets,requested;gc.collect()
    arms={}
    for arm in WEIGHTS:
        by=collections.defaultdict(list)
        for r in rows(OUT/f'candidates/{arm}.jsonl'):by[r['query_id']].append(r)
        actions={r['query_id']:r for r in rows(OUT/f'actions/{arm}.jsonl')}
        out=[]
        for q in queries:
            qid=q['query_id'];pool=by[qid];target=next(iter(qrels[qid]))
            scores={r['position']:text_rr[r['rewrite']][target] for r in pool}
            baseline=text_rr[q['query']][target]
            vals,contrasts=factorial(scores,actions[qid]['orders'],baseline)
            out.append({'domain':'staqc_unused','view':arm,'query_id':qid,'group':groups[qid],
                        'positions':len(pool),'orders':actions[qid]['orders'],'values':vals,'contrasts':contrasts,
                        'position_rr':{str(k):v for k,v in scores.items()}})
        arms[arm]=out;del by,actions
    arms['base_minus_local']=[paired_difference(a,b) for a,b in zip(arms['local'],arms['t5base'])]
    # Full utility and contrast curves are predefined descriptive outputs. Compute all
    # in the same draw stream so paired relationships are retained in every sample.
    names=[f'{view}::{section}::{key}' for view in arms for section in ('values','contrasts') for key in arms[view][0][section]]
    specs=[name.split('::') for name in names]
    matrix=np.array([[arms[view][i][section][key] for view,section,key in specs] for i in range(len(queries))])
    pnames={f'{view}::contrasts::{key}' for view,key in (n.split('::') for n in PRIMARY)}
    progress('statistics',queries=len(queries),groups=len(groups),metrics=len(names),bootstrap_repetitions=REPETITIONS)
    summary=summarize(matrix,names,pnames)
    views={view:{section:{key:summary[f'{view}::{section}::{key}'] for key in arms[view][0][section]} for section in ('values','contrasts')} for view in arms}
    primary={name:views[view]['contrasts'][key] for name in PRIMARY for view,key in [name.split('::')]}
    writel(OUT/'per_query.jsonl',(r for arm in WEIGHTS for r in arms[arm]))
    writel(OUT/'paired_per_query.jsonl',arms['base_minus_local'])
    result={'created_utc':now(),'design_sha256':sha(OUT/'DESIGN.json'),'opening':record(OUT/'OUTCOME_OPENING.json'),
            'queries':len(queries),'groups':len(groups),'corpus_documents':d['corpus_documents'],'views':views,'primary':primary,
            'primary_directions':{n:r['direction'] for n,r in primary.items()},
            'both_weights_BG_positive':all(primary[f'{w}::ENTR_{c}']['direction']=='positive' for w in WEIGHTS for c in ('B','G')),
            'new_setting_confirmation':True,'same_T5_family':True,'external_preregistration':False,
            'equivalence_tested':False,'risk_certified':False,'no_model_or_policy_promotion':True,'seconds':time.perf_counter()-started}
    write(OUT/'RESULTS.json',result)
    print({'primary_directions':result['primary_directions'],'both_weights_BG_positive':result['both_weights_BG_positive']},flush=True)


if __name__=='__main__':main()
