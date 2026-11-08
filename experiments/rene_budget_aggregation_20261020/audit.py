"""Standalone independent Fraction/CDF/statistical replay, no production imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter,defaultdict
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**23),b''):
            h.update(block)
    return h.hexdigest()


def require(ok,message):
    if not ok:
        raise AssertionError(message)


def exact_rr(order,target):
    require(len(order)==len(set(order))==100,'Top100 contract')
    return Fraction(1,order.index(target)+1) if target in order else Fraction(0)


def cdf_max(values,budget):
    """Integrate the discrete max tail; distinct from production order-statistic sum."""
    ordered=sorted(values);n=len(ordered);k=min(budget,n)
    total=Fraction(0);previous=Fraction(0)
    for index,value in enumerate(ordered):
        tail=1-Fraction(math.comb(index,k),math.comb(n,k))
        total+=(value-previous)*tail
        previous=value
    return total


def reference_metrics(score,order,original,rand20):
    v={'NoRewrite':original,'RAND20_best3':rand20}
    uniform=sum(score.values(),Fraction(0))/len(score)
    for k in (1,2,3):
        for p in ('ENTR','PROB'):
            chosen=[score[position] for position in order[p][:k]]
            v[f'{p}_k{k}_first']=chosen[0]
            v[f'{p}_k{k}_mean']=sum(chosen,Fraction(0))/len(chosen)
            v[f'{p}_k{k}_max']=max(chosen)
        v[f'RAND_exact_k{k}_first']=uniform
        v[f'RAND_exact_k{k}_mean']=uniform
        v[f'RAND_exact_k{k}_max']=cdf_max(list(score.values()),k)
    fx={f'delta_{p}_k{k}_{a}':v[f'{p}_k{k}_{a}']-original
        for p in ('ENTR','PROB','RAND_exact') for k in (1,2,3) for a in ('first','mean','max')}
    signals={p:{(k,a):v[f'{p}_k{k}_{a}']-v[f'RAND_exact_k{k}_{a}']
                for k in (1,2,3) for a in ('first','mean','max')} for p in ('ENTR','PROB')}
    for p,s in signals.items():
        fx.update({f'S_{p}_k{k}_{a}':value for (k,a),value in s.items()})
        for a in ('first','mean','max'):
            for lower,upper in ((1,2),(2,3),(1,3)):
                fx[f'change_{p}_{a}_k{lower}_to_k{upper}']=s[upper,a]-s[lower,a]
        for k in (1,2,3):
            for lower,upper in (('first','mean'),('mean','max'),('first','max')):
                fx[f'change_{p}_k{k}_{lower}_to_{upper}']=s[k,upper]-s[k,lower]
            fx[f'oracle_lift_{p}_k{k}']=v[f'{p}_k{k}_max']-v[f'{p}_k{k}_mean']
        fx[f'{p}_B_budget_mean_1_to_3']=v[f'{p}_k1_first']-v[f'{p}_k3_mean']
        fx[f'{p}_G_aggregation_at3_mean_to_max']=(v['RAND_exact_k3_max']-uniform)-(v[f'{p}_k3_max']-v[f'{p}_k3_mean'])
        fx[f'{p}_J_exact']=s[1,'first']-s[3,'max']
        fx[f'{p}_J_RAND20']=s[1,'first']-v[f'{p}_k3_max']+rand20
        require(fx[f'{p}_B_budget_mean_1_to_3']+fx[f'{p}_G_aggregation_at3_mean_to_max']==fx[f'{p}_J_exact'],'Exact reference-path identity')
    for k in (1,2,3):
        fx[f'oracle_lift_RAND_exact_k{k}']=v[f'RAND_exact_k{k}_max']-uniform
        for a in ('first','mean','max'):
            fx[f'ENTR_minus_PROB_k{k}_{a}']=v[f'ENTR_k{k}_{a}']-v[f'PROB_k{k}_{a}']
    for name in ('B_budget_mean_1_to_3','G_aggregation_at3_mean_to_max'):
        fx['ENTR_minus_PROB_'+name]=fx['ENTR_'+name]-fx['PROB_'+name]
    fx['RAND20_minus_exact3']=rand20-v['RAND_exact_k3_max']
    return v,fx


def independent_summary(matrix,groups,names,primary):
    """Group incidence bootstrap; production instead indexes/resums group arrays."""
    labels=sorted(set(groups));members={g:[] for g in labels}
    for row,g in zip(matrix,groups):
        members[g].append(row)
    counts=np.array([len(members[g]) for g in labels],dtype=float)
    totals=np.array([[math.fsum(row[j] for row in members[g]) for j in range(len(names))] for g in labels])
    means=np.array([math.fsum(row[j] for row in matrix)/len(matrix) for j in range(len(names))])
    se=np.array([math.sqrt(len(labels)/(len(labels)-1)*math.fsum((totals[i,j]-counts[i]*means[j])**2 for i in range(len(labels)))/len(matrix)**2) for j in range(len(names))])
    rng=np.random.default_rng(20261020)
    boot=np.empty((10000,len(names)))
    with threadpool_limits(limits=4):
        for offset in range(0,10000,250):
            draw=rng.integers(0,len(labels),(min(250,10000-offset),len(labels)))
            incidence=np.array([np.bincount(d,minlength=len(labels)) for d in draw],dtype=float)
            boot[offset:offset+len(draw)]=(incidence@totals)/(incidence@counts)[:,None]
    ordered=np.sort(boot,axis=0)

    def q(probability):
        index=(len(ordered)-1)*probability
        low=math.floor(index);fraction=index-low
        return ordered[low]*(1-fraction)+ordered[min(low+1,len(ordered)-1)]*fraction

    quantiles=[q(p) for p in (.025,.975,.05/48,1-.05/48)]
    critical=float(t.isf(.025,len(labels)-1));family=float(t.isf(.05/48,len(labels)-1))
    result={}
    for j,name in enumerate(names):
        r=dict(mean=float(means[j]),se_cluster=float(se[j]),
               ci95_group_bootstrap=[float(quantiles[0][j]),float(quantiles[1][j])],
               ci95_cluster_t=[float(means[j]-critical*se[j]),float(means[j]+critical*se[j])],
               changed_queries=sum(abs(row[j])>1e-12 for row in matrix),
               groups_with_nonzero=sum(any(abs(row[j])>1e-12 for row in members[g]) for g in labels))
        if name in primary:
            r.update(family24_ci95_group_bootstrap=[float(quantiles[2][j]),float(quantiles[3][j])],
                     family24_ci95_cluster_t=[float(means[j]-family*se[j]),float(means[j]+family*se[j])])
        result[name]=r
    return result


def audit(root):
    out=root/'results/rene_budget_aggregation_20261020'
    bridge=root/'results/rene_ssqr_bridge_20261019'
    A=root/'results/rene_core_step_a_20261020'
    protocol=read(out/'PROTOCOL.json');result=read(out/'RESULTS.json')
    counts=Counter();maximum=defaultdict(float)

    def near(actual,expected,label):
        if isinstance(expected,dict):
            require(set(actual)==set(expected),'Different summary keys: '+label)
            for key,value in expected.items():
                near(actual[key],value,label)
        elif isinstance(expected,(list,tuple)):
            require(len(actual)==len(expected),'Different vector sizes')
            for x,y in zip(actual,expected):
                near(x,y,label)
        else:
            error=abs(float(actual)-float(expected))
            maximum[label]=max(maximum[label],error)
            require(error<1e-12,f'{label}: {actual} differs from {expected}')
            counts[label]+=1

    require(protocol['role']=='post_hoc_paired_factorial_diagnostic_on_consumed_source_and_D','Wrong role')
    require(result['protocol_sha256']==sha(out/'PROTOCOL.json'),'Wrong protocol hash')
    for item in protocol['bound_files']+protocol['protected_files']:
        require(sha(root/item['path'])==item['sha256'],'Frozen drift: '+item['path'])
        counts['frozen_and_protected_files']+=1
    seal=read(out/'ACTIONS_SEALED.json');access=read(out/'OUTCOME_ACCESS.json')
    require(seal['protocol_sha256']==sha(out/'PROTOCOL.json'),'Action protocol differs')
    require(access['action_seal_sha256']==sha(out/'ACTIONS_SEALED.json'),'Action seal differs')
    require(protocol['created_utc']<seal['created_utc']<access['created_utc']<result['created_utc'],'Phase order differs')
    for item in seal['files']:
        require(sha(root/item['path'])==item['sha256'],'Action drift')
    action_rows=rows(out/'ACTIONS.jsonl')
    actions={(r['domain'],r['query_id']):r for r in action_rows}
    require(len(actions)==len(action_rows)==3538,'Wrong action population')
    exported_rows=rows(out/'per_query.jsonl')
    exported={(r['domain'],r['retriever'],r['query_id']):r for r in exported_rows}
    require(len(exported)==len(exported_rows)==7076,'Wrong factorial population')
    old={(r['domain'],r['retriever'],r['query_id']):r for r in rows(A/'per_query.jsonl')}
    seeds=read(A/'PROTOCOL.json')['rand_seeds']
    require(len(seeds)==20,'Wrong historical random repeats')
    view_map={(v['domain'],v['retriever']):v for v in result['views']}
    require(set(view_map)=={(d,r) for d in ('source','cosqa') for r in ('CodeBERT','BM25')},'Missing view')
    primary={f'{p}_{c}' for p in ('ENTR','PROB','ENTR_minus_PROB')
             for c in ('B_budget_mean_1_to_3','G_aggregation_at3_mean_to_max')}
    require(set(protocol['primary_comparisons']['per_view'])==primary and protocol['primary_comparisons']['family_size']==24,'Wrong planned family')
    for domain,sample in protocol['samples'].items():
        ids=sample['query_ids']
        queries={r['query_id']:r['query'] for r in rows(bridge/f'data/{domain}/queries.jsonl')}
        qrel=rows(bridge/f'data/private/{domain}_qrels.jsonl')
        target={r['query_id']:r['doc_id'] for r in qrel}
        require(len(target)==len(qrel)==len(ids) and all(r['relevance']==1 for r in qrel),'Single-positive target contract')
        pools=defaultdict(list)
        for row in rows(bridge/f'candidates/{domain}/baseline.jsonl'):
            pools[row['query_id']].append(row)
        require(set(pools)==set(queries)==set(target)==set(ids),'Unpaired population')
        sorted_orders={}
        for qid in ids:
            pool=sorted(pools[qid],key=lambda c:c['position'])
            require([c['position'] for c in pool]==list(range(1,len(queries[qid].split())+1)),'Position count drift')
            order={
                'ENTR':[c['position'] for c in sorted(pool,key=lambda c:(c['full_vocab_entropy'],c['position']))],
                'PROB':[c['position'] for c in sorted(pool,key=lambda c:(-c['label_mean_log_probability'],c['position']))],
            }
            require(actions[domain,qid]['orders']==order,'Label-free action prefix differs')
            require(actions[domain,qid]['effective_k']=={str(k):min(k,len(pool)) for k in (1,2,3)},'Short-pool cap differs')
            sorted_orders[qid]=order
            counts['position_orders']+=2
        for retriever in ('CodeBERT','BM25'):
            ranks={}
            for row in rows(bridge/f'rankings/{domain}/{retriever}.jsonl'):
                require(hashlib.sha256(row['text'].encode()).hexdigest()==row['text_id'],'Text hash drift')
                require(row['text_id'] not in ranks,'Duplicate ranking text')
                require(len(row['ranked_doc_ids'])==len(set(row['ranked_doc_ids']))==100,'Ranking depth differs')
                require(list(zip(row['ranked_doc_ids'],row['scores']))==sorted(zip(row['ranked_doc_ids'],row['scores']),key=lambda x:(-x[1],x[0])),'Rank score/tie order differs')
                ranks[row['text_id']]=row['ranked_doc_ids'];counts['ranking_texts_checked']+=1
            raw_records=[];fx_records=[]
            for qid in ids:
                saved=exported[domain,retriever,qid];previous=old[domain,retriever,qid]
                require(saved['group']==sample['groups'][qid] and saved['positions']==len(pools[qid]),'Group/pool mismatch')
                score={}
                for c in pools[qid]:
                    text_hash=hashlib.sha256(c['rewrite'].encode()).hexdigest()
                    score[c['position']]=exact_rr(ranks[text_hash],target[qid])
                original=exact_rr(ranks[hashlib.sha256(queries[qid].encode()).hexdigest()],target[qid])
                counts['Fraction_candidate_RR_recomputed']+=len(score)
                counts['Fraction_original_RR_recomputed']+=1
                require(set(score)==set(dict(previous['candidate_RR'])),'Old A missing position')
                near(dict(previous['candidate_RR']),score,'old_A_candidate_RR')
                shuffled=[]
                for seed in seeds:
                    digest=hashlib.sha256(f'{seed}\0{qid}'.encode()).digest()
                    positions=random.Random(int.from_bytes(digest,'big')).sample(sorted(score),len(score))
                    shuffled.append(max(score[pos] for pos in positions[:3]))
                rand20=sum(shuffled,Fraction(0))/20
                v,fx=reference_metrics(score,sorted_orders[qid],original,rand20)
                near(saved['values'],v,'factorial_values')
                near(saved['contrasts'],fx,'factorial_effects')
                for p in ('ENTR','PROB'):
                    near(previous['contrasts'][f'j_{p}_contract_difference_exact_random3'],fx[f'{p}_J_exact'],'old_A_J_exact')
                    near(previous['contrasts'][f'j_{p}_contract_difference'],fx[f'{p}_J_RAND20'],'old_A_J20')
                    require(fx[f'{p}_J_RAND20']==fx[f'{p}_J_exact']+fx['RAND20_minus_exact3'],'Fraction MC identity')
                    counts['exact_decomposition_identities']+=2
                raw_records.append({k:float(value) for k,value in v.items()})
                fx_records.append({k:float(value) for k,value in fx.items()})
            view=view_map[domain,retriever]
            require((view['queries'],view['groups'])==(len(ids),len(set(sample['groups'].values()))),'View denominator drift')
            names=list(raw_records[0])+list(fx_records[0])
            matrix=[[r[name] if name in r else f[name] for name in names] for r,f in zip(raw_records,fx_records)]
            reference=independent_summary(matrix,[sample['groups'][qid] for qid in ids],names,primary)
            near(view['raw_summaries'],{name:reference[name] for name in raw_records[0]},'raw_statistical_fields')
            near(view['effects'],{name:reference[name] for name in fx_records[0]},'effect_statistical_fields')
            near(view['primary_decomposition'],{name:reference[name] for name in primary},'primary_family_fields')
            expected_combos={(p,k,a) for p in ('ENTR','PROB','RAND_exact') for k in (1,2,3) for a in ('first','mean','max')}
            require(len(view['factorial'])==len(expected_combos)==27,'Factorial cell count')
            for cell in view['factorial']:
                p,k,a=cell['policy'],cell['k'],cell['aggregation']
                require((p,k,a) in expected_combos,'Factorial duplicate/missing')
                expected_combos.remove((p,k,a))
                near(cell['MRR'],reference[f'{p}_k{k}_{a}'],'factorial_table_MRR')
                near(cell['delta_NoRewrite'],reference[f'delta_{p}_k{k}_{a}'],'factorial_table_delta')
                if p=='RAND_exact':
                    require(cell['advantage_vs_random'] is None,'Random self-advantage')
                else:
                    near(cell['advantage_vs_random'],reference[f'S_{p}_k{k}_{a}'],'factorial_table_advantage')
                require(cell['label_assisted']==(a=='max'),'Opportunity label differs')
                counts['factorial_cells']+=1
            require(not expected_combos,'Factorial incomplete')
            counts['views_replayed']+=1
            print(json.dumps(dict(independent_view=f'{domain}/{retriever}',queries=len(ids)),ensure_ascii=False),flush=True)
    for name in ('risk_certified','new_confirmation'):
        require(result[name] is False,'Unexpected efficacy/certification identity')
    return dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),independent_no_production_imports=True,
                counts=dict(counts),max_errors=dict(maximum),new_confirmation=False,
                H_qrels_or_rankings_read=False,no_model_inference=True,risk_certified=False,
                protocol_sha256=sha(out/'PROTOCOL.json'),results_sha256=sha(out/'RESULTS.json'),
                per_query_sha256=sha(out/'per_query.jsonl'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit(args.root)
    path=args.output or args.root/'results/rene_budget_aggregation_20261020/AUDIT.json'
    with path.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
