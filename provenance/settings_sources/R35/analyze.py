"""Paired effects conditional on fixed budget and declared training seeds."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from .common import ROOT, OUT, STORE, ARM_SPECS, now, read, rows, write, writel, sha, require, text_id, verify
from .evaluate import paths,check_eval


def exact_max(values,k=3):
    n=len(values);k=min(k,n)
    return math.fsum(value*math.comb(i,k-1)/math.comb(n,k) for i,value in enumerate(sorted(values)) if i>=k-1)


def rr(ranking,target):
    return 1./(ranking.index(target)+1) if target in ranking else 0.


def summarize(matrix,groups,names):
    x=np.asarray(matrix,dtype=float)
    labels=sorted(set(groups));index={g:i for i,g in enumerate(labels)}
    codes=[index[g] for g in groups]
    sums=np.zeros((len(labels),len(names)));np.add.at(sums,codes,x)
    sizes=np.bincount(codes,minlength=len(labels)).astype(float)
    mean=x.mean(0);se=np.sqrt(len(labels)/(len(labels)-1)*np.square(sums-sizes[:,None]*mean).sum(0)/len(x)**2)
    rng=np.random.default_rng(20261020);boot=np.empty((10000,len(names)))
    for offset in range(0,10000,64):
        draw=rng.integers(0,len(labels),(min(64,10000-offset),len(labels)))
        boot[offset:offset+len(draw)]=sums[draw].sum(1)/sizes[draw].sum(1)[:,None]
    quantiles=np.quantile(boot,[.025,.975,.05/56,1-.05/56],axis=0)
    output={}
    for i,name in enumerate(names):
        width=float(t.ppf(.975,len(labels)-1))*se[i];familywidth=float(t.ppf(1-.05/56,len(labels)-1))*se[i]
        output[name]=dict(mean=float(mean[i]),se_cluster=float(se[i]),ci95_group_bootstrap=quantiles[:2,i].tolist(),
            ci95_cluster_t=[float(mean[i]-width),float(mean[i]+width)],
            family28_ci95_group_bootstrap=quantiles[2:,i].tolist(),
            family28_ci95_cluster_t=[float(mean[i]-familywidth),float(mean[i]+familywidth)])
    return output


def values_for(pool,rankings,target,query,selection,random_seeds):
    scores={r['position']:rr(rankings[text_id(r['rewrite'])],target) for r in pool}
    result=dict(NoRewrite=rr(rankings[text_id(query)],target),RAND_exact_top1=math.fsum(scores.values())/len(scores),
        RAND_exact_best3=exact_max(list(scores.values())),Oracle_all=max(scores.values()))
    for policy in ('ENTR','PROB'):
        selected=selection[policy]
        result[policy+'_top1']=scores[selected[0]]
        result[policy+'_best3']=max(scores[p] for p in selected[:3])
        result[policy+'_mean3']=math.fsum(scores[p] for p in selected[:3])/min(3,len(selected))
    from experiments.rene_ssqr_alignment_20261019.t5 import select_candidates
    result['RAND20_best3']=math.fsum(max(scores[r['position']] for r in select_candidates(pool,policy='RAND',k=3,seed=s)) for s in random_seeds)/20
    return result,scores


def main():
    check_eval();p=verify()
    require(not (OUT/'RESULTS.json').exists(),'Results already exist')
    from pathlib import Path
    for seal_name in ('CANDIDATES_SEALED.json','RANKINGS_SEALED.json'):
        seal=read(OUT/seal_name)
        for item in seal['files']:
            path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
            require(sha(path)==item['sha256'],'Inference seal changed')
    write(OUT/'OUTCOME_ACCESS.json',dict(created_utc=now(),eval_protocol_sha256=sha(OUT/'EVAL_PROTOCOL.json'),
        rankings_seal_sha256=sha(OUT/'RANKINGS_SEALED.json'),previous_historical_outcomes_known=True,new_confirmation=False,no_H_read=True))
    choices={(r['arm'],r['domain'],r['view'],r['query_id']):r for r in rows(OUT/'selections.jsonl')}
    random_seeds=read(ROOT/'results/rene_ssqr_bridge_20261019/PROTOCOL.json')['rand_seeds']
    allquery=[];allcandidates=[];views=[];primary=[];descriptive=[]
    for domain in ('source','cosqa'):
        queries=rows(paths(domain)['queries']); ids=[q['query_id'] for q in queries]
        qrels={r['query_id']:r['doc_id'] for r in rows(paths(domain)['qrels']) if r['query_id'] in set(ids)}
        rankings={r['text_id']:r['ranked_doc_ids'] for r in rows(OUT/f'rankings/{domain}/Top100.jsonl')}
        groups=[p['samples'][domain]['groups'][qid] for qid in ids]
        for view in ('own','fixed'):
            byarm={}
            for arm in ARM_SPECS:
                pools=defaultdict(list)
                for r in rows(OUT/f'candidates/{arm}/{domain}_{view}.jsonl'):pools[r['query_id']].append(r)
                values=[];rates=[]
                for q in queries:
                    qid=q['query_id'];pool=pools[qid]
                    result,scores=values_for(pool,rankings,qrels[qid],q['query'],choices[arm,domain,view,qid],random_seeds)
                    effect=dict(S3=result['ENTR_best3']-result['RAND_exact_best3'],S1=result['ENTR_top1']-result['RAND_exact_top1'],
                        top1_gain=result['ENTR_top1']-result['NoRewrite'],best3_gain=result['ENTR_best3']-result['NoRewrite'],
                        prob_S3=result['PROB_best3']-result['RAND_exact_best3'],ENTR_minus_PROB3=result['ENTR_best3']-result['PROB_best3'],
                        J=(result['ENTR_top1']-result['RAND_exact_top1'])-(result['ENTR_best3']-result['RAND_exact_best3']),
                        S3_RAND20=result['ENTR_best3']-result['RAND20_best3'])
                    row=dict(domain=domain,view=view,arm=arm,query_id=qid,group=groups[len(values)],RR=result,effects=effect,positions=len(pool))
                    allquery.append(row);values.append(row)
                    allcandidates.extend(dict(domain=domain,view=view,arm=arm,query_id=qid,position=r['position'],
                        text_id=text_id(r['rewrite']),RR=scores[r['position']]) for r in pool)
                byarm[arm]=values
                names=list(values[0]['RR'])+list(values[0]['effects'])
                summaries=summarize([[*r['RR'].values(),*r['effects'].values()] for r in values],groups,names)
                # Family-adjusted bounds are only primary for between-arm changes below.
                for summary in summaries.values():
                    summary.pop('family28_ci95_group_bootstrap');summary.pop('family28_ci95_cluster_t')
                views.append(dict(domain=domain,view=view,arm=arm,queries=512,groups=len(set(groups)),
                    MRR={k:summaries[k] for k in values[0]['RR']},effects={k:summaries[k] for k in values[0]['effects']},
                    harmful_top1=sum(r['effects']['top1_gain']<0 for r in values),beneficial_top1=sum(r['effects']['top1_gain']>0 for r in values)))
            names=[f'{arm}:{metric}' for arm in ARM_SPECS if arm!='L' for metric in ('S3','top1_gain','S1','J','prob_S3')]
            matrix=[[byarm[arm][i]['effects'][metric]-byarm['L'][i]['effects'][metric] for arm in ARM_SPECS if arm!='L'
                for metric in ('S3','top1_gain','S1','J','prob_S3')] for i in range(512)]
            summaries=summarize(matrix,groups,names)
            for name,summary in summaries.items():
                arm,metric=name.split(':')
                item=dict(domain=domain,view=view,arm=arm,reference='L',metric=metric,**summary)
                if view=='own' and metric in ('S3','top1_gain'):
                    primary.append(item)
                else:
                    item.pop('family28_ci95_group_bootstrap');item.pop('family28_ci95_cluster_t');descriptive.append(item)
            seed_effect=summarize([[byarm['PS'][i]['effects'][m]-byarm['P'][i]['effects'][m] for m in ('S3','top1_gain')]
                for i in range(512)],groups,['S3','top1_gain'])
            for metric,summary in seed_effect.items():
                summary.pop('family28_ci95_group_bootstrap');summary.pop('family28_ci95_cluster_t')
                descriptive.append(dict(domain=domain,view=view,arm='PS',reference='P',metric=metric,**summary))
    require(len(primary)==28,'Primary family size')
    writel(OUT/'per_query.jsonl',allquery);writel(OUT/'candidate_RR.jsonl',allcandidates)
    result=dict(created_utc=now(),training_protocol_sha256=sha(OUT/'TRAIN_PROTOCOL.json'),eval_protocol_sha256=sha(OUT/'EVAL_PROTOCOL.json'),
        role=p['role'],new_confirmation=False,risk_certified=False,conditional_on_seeds=True,views=views,
        primary_changes=primary,descriptive_changes=descriptive,training=[read(OUT/f'training/{a}.json') for a in ARM_SPECS],
        no_H_read=True,historical93750_update_effect_not_identified=True)
    write(OUT/'RESULTS.json',result)
    for name,records in [('PRIMARY.csv',primary),('DESCRIPTIVE.csv',descriptive)]:
        columns=list(dict.fromkeys(k for r in records for k in r))
        with (OUT/name).open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for r in records:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()})
    print(dict(stage='effects_complete',primary_comparisons=len(primary),per_query_rows=len(allquery)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
