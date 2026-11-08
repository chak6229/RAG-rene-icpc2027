from __future__ import annotations

import math

import numpy as np
from scipy.stats import t

POLICIES=('ENTR','PROB','RAND_exact')
AGGREGATIONS=('first','mean','max')
BUDGETS=(1,2,3)
PRIMARY_NAMES=tuple(f'{p}_{c}' for p in ('ENTR','PROB','ENTR_minus_PROB')
                    for c in ('B_budget_mean_1_to_3','G_aggregation_at3_mean_to_max'))


def ordered_positions(candidates,policy):
    if not candidates or len({r['position'] for r in candidates})!=len(candidates):
        raise ValueError('Nonempty unique positions required')
    if policy=='ENTR':
        key=lambda r:(r['full_vocab_entropy'],r['position'])
    elif policy=='PROB':
        key=lambda r:(-r['label_mean_log_probability'],r['position'])
    else:
        raise ValueError(policy)
    if not all(math.isfinite(r['full_vocab_entropy']) and math.isfinite(r['label_mean_log_probability']) for r in candidates):
        raise ValueError('Nonfinite generation score')
    return [r['position'] for r in sorted(candidates,key=key)]


def exact_random_max(values,k):
    n=len(values)
    if n<1 or k<1:
        raise ValueError('Empty pool or nonpositive budget')
    k=min(k,n)
    ordered=sorted(values)
    denominator=math.comb(n,k)
    return math.fsum(v*math.comb(i,k-1)/denominator for i,v in enumerate(ordered) if i>=k-1)


def factorial_values(scores,orders,baseline,rand20):
    if not scores or any(set(order)!=set(scores) or len(order)!=len(scores) for order in orders.values()):
        raise ValueError('Position/order mismatch')
    if set(orders)!={'ENTR','PROB'}:
        raise ValueError('Missing selector')
    out={'NoRewrite':float(baseline),'RAND20_best3':float(rand20)}
    uniform=math.fsum(scores.values())/len(scores)
    for k in BUDGETS:
        for policy in POLICIES:
            if policy=='RAND_exact':
                values={'first':uniform,'mean':uniform,'max':exact_random_max(list(scores.values()),k)}
            else:
                selected=[scores[p] for p in orders[policy][:min(k,len(scores))]]
                values={'first':selected[0],'mean':math.fsum(selected)/len(selected),'max':max(selected)}
            out.update({f'{policy}_k{k}_{aggregation}':float(value) for aggregation,value in values.items()})
    return out


def contrasts(v):
    out={}
    for p in POLICIES:
        for k in BUDGETS:
            for a in AGGREGATIONS:
                out[f'delta_{p}_k{k}_{a}']=v[f'{p}_k{k}_{a}']-v['NoRewrite']
    for p in ('ENTR','PROB'):
        s={(k,a):v[f'{p}_k{k}_{a}']-v[f'RAND_exact_k{k}_{a}'] for k in BUDGETS for a in AGGREGATIONS}
        for (k,a),value in s.items():
            out[f'S_{p}_k{k}_{a}']=value
        for a in AGGREGATIONS:
            for before,after in ((1,2),(2,3),(1,3)):
                out[f'change_{p}_{a}_k{before}_to_k{after}']=s[after,a]-s[before,a]
        for k in BUDGETS:
            for before,after in (('first','mean'),('mean','max'),('first','max')):
                out[f'change_{p}_k{k}_{before}_to_{after}']=s[k,after]-s[k,before]
            out[f'oracle_lift_{p}_k{k}']=v[f'{p}_k{k}_max']-v[f'{p}_k{k}_mean']
        out[f'{p}_B_budget_mean_1_to_3']=s[1,'mean']-s[3,'mean']
        out[f'{p}_G_aggregation_at3_mean_to_max']=s[3,'mean']-s[3,'max']
        out[f'{p}_J_exact']=s[1,'mean']-s[3,'max']
        out[f'{p}_J_RAND20']=s[1,'mean']-(v[f'{p}_k3_max']-v['RAND20_best3'])
    for k in BUDGETS:
        out[f'oracle_lift_RAND_exact_k{k}']=v[f'RAND_exact_k{k}_max']-v[f'RAND_exact_k{k}_mean']
        for a in AGGREGATIONS:
            out[f'ENTR_minus_PROB_k{k}_{a}']=v[f'ENTR_k{k}_{a}']-v[f'PROB_k{k}_{a}']
    for c in ('B_budget_mean_1_to_3','G_aggregation_at3_mean_to_max'):
        out['ENTR_minus_PROB_'+c]=out['ENTR_'+c]-out['PROB_'+c]
    out['RAND20_minus_exact3']=v['RAND20_best3']-v['RAND_exact_k3_max']
    for p in ('ENTR','PROB'):
        if abs(out[f'{p}_J_exact']-out[f'{p}_B_budget_mean_1_to_3']-out[f'{p}_G_aggregation_at3_mean_to_max'])>1e-13:
            raise AssertionError('Decomposition identity')
    return out


def clustered_summary(matrix,groups,names,*,repetitions=10000,seed=20261020,family_size=24):
    x=np.asarray(matrix,dtype=np.float64)
    labels=sorted(set(groups))
    if x.ndim!=2 or x.shape!=(len(groups),len(names)) or len(labels)<2 or not np.isfinite(x).all():
        raise ValueError('Invalid clustered matrix')
    codes=np.array([labels.index(g) for g in groups])
    sizes=np.bincount(codes,minlength=len(labels)).astype(float)
    sums=np.zeros((len(labels),len(names)))
    np.add.at(sums,codes,x)
    mean=x.mean(0)
    se=np.sqrt(len(labels)/(len(labels)-1)*np.square(sums-sizes[:,None]*mean).sum(0)/len(x)**2)
    boot=np.empty((repetitions,len(names)))
    rng=np.random.default_rng(seed)
    for offset in range(0,repetitions,64):
        count=min(64,repetitions-offset)
        draw=rng.integers(0,len(labels),(count,len(labels)))
        boot[offset:offset+count]=sums[draw].sum(1)/sizes[draw].sum(1)[:,None]
    quantiles=np.quantile(boot,[.025,.975,.05/(2*family_size),1-.05/(2*family_size)],axis=0)
    critical=float(t.ppf(.975,len(labels)-1))
    family_critical=float(t.ppf(1-.05/(2*family_size),len(labels)-1))
    changed=np.zeros((len(labels),len(names)),dtype=bool)
    np.logical_or.at(changed,codes,np.abs(x)>1e-12)
    out={}
    for i,name in enumerate(names):
        record=dict(mean=float(mean[i]),se_cluster=float(se[i]),
                    ci95_group_bootstrap=quantiles[:2,i].tolist(),
                    ci95_cluster_t=[float(mean[i]-critical*se[i]),float(mean[i]+critical*se[i])],
                    changed_queries=int(np.count_nonzero(np.abs(x[:,i])>1e-12)),
                    groups_with_nonzero=int(changed[:,i].sum()))
        if name in PRIMARY_NAMES:
            record.update(family24_ci95_group_bootstrap=quantiles[2:,i].tolist(),
                          family24_ci95_cluster_t=[float(mean[i]-family_critical*se[i]),float(mean[i]+family_critical*se[i])])
        out[name]=record
    return out
