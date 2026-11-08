"""Cluster-paired inference and batch-ranking bootstrap (no stdlib shadowing)."""
from __future__ import annotations

import numpy as np
from scipy import stats

from experiments.rene_uq_independent_proof_20261019.study import SEED,TOL


def block_bootstrap_indices(n,repeats=10000):
    if n % 8:
        raise ValueError("Confirmation sample must contain complete generation blocks")
    return np.random.default_rng(SEED).integers(n//8,size=(repeats,n//8),dtype=np.int32)


def intervals(values,draws):
    x = np.asarray(values,float)
    if x.ndim != 1 or len(x)%8:
        raise ValueError("Expected query vector with complete blocks")
    blocks = x.reshape(-1,8).mean(1)
    bootstrap = blocks[draws].mean(1)
    b95 = np.quantile(bootstrap,[.025,.975]).tolist()
    b975 = np.quantile(bootstrap,[.0125,.9875]).tolist()
    sd = float(blocks.std(ddof=1))
    half = stats.t.ppf(.9875,len(blocks)-1)*sd/np.sqrt(len(blocks))
    center = float(x.mean())
    t975 = [float(center-half),float(center+half)]
    return dict(mean=center,descriptive_ci95=b95,simultaneous_bootstrap_ci97_5=b975,
                sensitivity_cluster_t_ci97_5=t975,
                simultaneous_envelope=[min(b975[0],t975[0]),max(b975[1],t975[1])],
                nonzero_queries=int(np.count_nonzero(abs(x)>TOL)),
                nonzero_generation_blocks=int(np.count_nonzero(np.any(abs(x.reshape(-1,8))>TOL,axis=1))),
                query_sd=float(x.std(ddof=1)),generation_block_sd=sd)


def verdict(comparison,margin=.005):
    lo,hi = comparison["simultaneous_envelope"]
    direction = "positive" if lo>0 else "negative" if hi<0 else "unresolved"
    sufficient_events = comparison["nonzero_queries"]>=20 and comparison["nonzero_generation_blocks"]>=10
    equivalent = bool(-margin<lo and hi<margin and sufficient_events)
    return dict(direction=direction,equivalence_established=equivalent,
                equivalence_margin=margin,nonzero_event_guard_met=sufficient_events)


def score_difference(values,eligible,draws):
    """Proper-score difference among same pre-gate eligible actions."""
    values,eligible = np.asarray(values,float),np.asarray(eligible,bool)
    sums = np.where(eligible,values,0.).reshape(-1,8).sum(1)
    counts = eligible.reshape(-1,8).sum(1)
    denominator = counts[draws].sum(1)
    boot = sums[draws].sum(1)/np.maximum(denominator,1)
    return dict(mean=float(sums.sum()/max(int(counts.sum()),1)),
                descriptive_ci95=np.quantile(boot,[.025,.975]).tolist(),
                eligible=int(counts.sum()))


def fixed_half_contrast(delta,eligible,prob_before,prob_after,queries,draws):
    """Re-select within every bootstrap sample; fixed50 is a batch policy."""
    delta=np.asarray(delta,float); eligible=np.asarray(eligible,bool)
    queries=np.asarray(queries,int)
    eligible_global=np.flatnonzero(eligible)
    def value(indices,p):
        current=indices[eligible[indices]]
        count=len(current)//2
        order=np.lexsort((queries[current],p[current]))
        return float(delta[current[order[:count]]].sum()/len(indices))
    before=np.asarray(prob_before,float);after=np.asarray(prob_after,float)
    indices=np.arange(len(delta))
    point=value(indices,after)-value(indices,before)
    boot=[]
    offsets=np.arange(8)
    for blocks in draws:
        ix=(blocks[:,None]*8+offsets[None,:]).ravel()
        boot.append(value(ix,after)-value(ix,before))
    return dict(mean=point,descriptive_ci95=np.quantile(boot,[.025,.975]).tolist(),
                resampled_adoption_recomputed=True,queries=len(delta),
                nominal_adoptions=int(len(eligible_global)//2),confirmatory=False)
