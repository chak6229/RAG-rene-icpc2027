"""Label-free rank fusion with a deterministic per-document RR-loss bound.

Bounds loss severity for any single relevant document. Does not bound how
often a loss occurs and does not estimate epistemic uncertainty.
"""
import math


def _scores(original,rewrite,weight,cutoff,offset):
    if not 0<=weight<=1 or cutoff<1 or offset<0:raise ValueError('invalid ranking parameters')
    original=tuple(map(int,original[:cutoff]));rewrite=tuple(map(int,rewrite[:cutoff]))
    if len(set(original))!=len(original) or len(set(rewrite))!=len(rewrite):raise ValueError('duplicate document')
    if len(original)!=cutoff:raise ValueError('full original ranking required')
    scores={}
    for ranking,w in ((original,1.),(rewrite,weight)):
        for rank,doc in enumerate(ranking,1):scores[doc]=scores.get(doc,0.)+w/(offset+rank)
    return original,scores


def weighted_rrf(original,rewrite,weight=1.,cutoff=100,offset=60):
    original,scores=_scores(original,rewrite,weight,cutoff,offset)
    if weight==0:return original
    return tuple(sorted(scores,key=lambda d:(-scores[d],d))[:cutoff])


def bounded_rrf(original,rewrite,max_loss=.1,weight=1.,cutoff=100,offset=60):
    """Schedule fusion priorities subject to reciprocal-rank deadlines.

For every originally retrieved document with 1/r > epsilon, require an
output rank <= floor(1/(1/r-epsilon)), capped at the evaluation cutoff to
prevent disappearance. Original order is always a feasible schedule.

At position t, a tight prefix of remaining earliest-deadline jobs must
contribute the next item. Choose the highest-priority item from its shortest
tight prefix; otherwise choose the highest-priority remaining item. This
maintains all remaining deadline feasibility by induction.
"""
    if not 0<=max_loss<=1:raise ValueError('invalid RR-loss budget')
    original,scores=_scores(original,rewrite,weight,cutoff,offset)
    deadlines={}
    for rank,doc in enumerate(original,1):
        retained=1/rank-max_loss
        if retained>1e-12:deadlines[doc]=min(cutoff,math.floor(1/retained+1e-10))
    priority=sorted(scores,key=lambda d:(-scores[d],d));order={d:i for i,d in enumerate(priority)}
    remaining=set(priority);out=[]
    for t in range(1,cutoff+1):
        constrained=sorted((d for d in remaining if d in deadlines),key=lambda d:(deadlines[d],d))
        tight=None
        for j,d in enumerate(constrained):
            slack=deadlines[d]-(t+j)
            if slack<0:raise AssertionError('infeasible deadline schedule')
            if slack==0:tight=j+1;break
        eligible=constrained[:tight] if tight is not None else remaining
        chosen=min(eligible,key=order.__getitem__)
        remaining.remove(chosen);out.append(chosen)
    ranks={d:i+1 for i,d in enumerate(out)}
    assert all(d in ranks and ranks[d]<=deadline for d,deadline in deadlines.items())
    assert max(1/r-(1/ranks[d] if d in ranks else 0.) for r,d in enumerate(original,1))<=max_loss+1e-10
    return tuple(out)


def worst_case_loss(original,action,cutoff=100):
    ranks={int(d):i+1 for i,d in enumerate(action[:cutoff])}
    return max([0.]+[1/r-(1/ranks[int(d)] if int(d) in ranks else 0.) for r,d in enumerate(original[:cutoff],1)])
