"""Independent data/training/Fraction/statistical audit; no production imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/rene_cqc_training_fidelity_20261020'
STORE=Path('external_inputs/cqc_training_model_store')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as h:return [json.loads(x) for x in h if x.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**23),b''):h.update(b)
    return h.hexdigest()


def resolve(value):
    p=Path(value);return p if p.is_absolute() else ROOT/p


def require(ok,message):
    if not ok:raise AssertionError(message)


def write(name,value):
    with (OUT/name).open('x',encoding='utf-8') as h:json.dump(value,h,indent=2,ensure_ascii=False)


def training():
    p=read(OUT/'TRAIN_PROTOCOL.json');runtime=read(OUT/'RUNTIME_DETERMINISM.json')
    counts=defaultdict(int)
    for item in p['bound_files']+p['protected_files']:
        require(sha(resolve(item['path']))==item['sha256'],'Training-bound digest: '+item['path']);counts['frozen_training_files']+=1
    require(runtime['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and runtime['all_formal_runs_use_same_runtime'],'Runtime determinism')
    bank=rows(OUT/'data/train.jsonl');raw=(ROOT/'data/codenn_cqc/train.txt').read_text(encoding='utf-8').splitlines()
    expected=random.Random(20261020).sample(range(len(raw)),32768);rng=random.Random(20261021)
    for row,index in zip(bank,expected):
        words=raw[index].split();length=max(1,round(.15*len(words)));start=rng.randrange(len(words)-length+1)
        source=' '.join(words[:start]+['<extra_id_0>']+words[start+length:])
        target='<extra_id_0> '+' '.join(words[start:start+length])+' <extra_id_1>'
        require(row==dict(source_index=index,query=raw[index],source=source,target=target),'Shared training bank rederivation')
        counts['training_rows_rederived']+=1
    losses={};reports={}
    for arm,spec in p['arms'].items():
        report=read(OUT/f'training/{arm}.json');steps=rows(OUT/f'training/{arm}_steps.jsonl')
        require(report['spec']==spec and report['steps']==1024 and report['examples_seen']==32768,'Budget/spec agreement')
        require(len(steps)==1024 and [r['step'] for r in steps]==list(range(1,1025)),'No missing or duplicate training updates')
        require(report['weights_sha256']==sha(STORE/arm/'model.safetensors'),'Weights digest')
        require(report['training_bank_sha256']==sha(OUT/'data/train.jsonl'),'Training bank binding')
        for i,row in enumerate(steps):
            expected_lr=.001 if spec['schedule']=='constant' else .001*(1-i/spec['horizon'])
            require(row['bank_start']==i*32 and row['bank_stop']==(i+1)*32,'Common minibatch exposure')
            require(abs(row['learning_rate']-expected_lr)<1e-15,'Learning-rate schedule')
            require(math.isfinite(row['mean_microbatch_loss']) and math.isfinite(row['gradient_norm']),'Finite training')
            counts['update_records_checked']+=1
        losses[arm]=[(r['mean_microbatch_loss'],r['gradient_norm']) for r in steps];reports[arm]=report
    require(losses['L']==losses['N'],'Length-only negative control: full training loss/gradient trajectory')
    require(reports['L']['weights_sha256']==reports['N']['weights_sha256'],'Length-only negative control: exact weight bytes')
    require(len({r['weights_sha256'] for a,r in reports.items() if a!='N'})==7,'All other trained configurations differ')
    counts['length_control_identical_updates']=1024
    write('TRAIN_AUDIT.json',dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),independent_no_production_imports=True,
        counts=dict(counts),length_control_identical_weights=True,training_protocol_sha256=sha(OUT/'TRAIN_PROTOCOL.json'),
        weights={a:r['weights_sha256'] for a,r in reports.items()},new_confirmation=False))
    print(dict(stage='training_audit_passed',counts=dict(counts)),flush=True)


def cdf_max(values):
    values=sorted(values);n=len(values);k=min(3,n);out=Fraction(0);previous=Fraction(0)
    for i,value in enumerate(values):
        out+=(value-previous)*(1-Fraction(math.comb(i,k),math.comb(n,k)));previous=value
    return out


def exact_rr(order,target):
    require(len(order)==len(set(order))==100,'Unique fixed-depth ranking')
    return Fraction(1,order.index(target)+1) if target in order else Fraction(0)


def stat_reference(matrix,groups,names):
    # Independently aggregate Python lists before vectorized resampling.
    labels=sorted(set(groups));grouped={g:[] for g in labels}
    for row,g in zip(matrix,groups):grouped[g].append(row)
    sums=np.array([[math.fsum(r[i] for r in grouped[g]) for i in range(len(names))] for g in labels])
    sizes=np.array([len(grouped[g]) for g in labels],float)
    center=np.array([math.fsum(r[i] for r in matrix)/len(matrix) for i in range(len(names))])
    residual=sums-sizes[:,None]*center
    se=np.sqrt((len(labels)/(len(labels)-1))*np.sum(residual*residual,axis=0))/len(matrix)
    rng=np.random.default_rng(20261020);rep=[]
    for start in range(0,10000,64):
        draw=rng.integers(0,len(labels),(min(64,10000-start),len(labels)))
        rep.append(sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None])
    bounds=np.quantile(np.concatenate(rep),[.025,.975,.05/56,1-.05/56],axis=0)
    result={}
    for i,name in enumerate(names):
        z=float(t.isf(.025,len(labels)-1));familyz=float(t.isf(.05/56,len(labels)-1))
        result[name]=dict(mean=center[i],se_cluster=se[i],ci95_group_bootstrap=bounds[:2,i],
            ci95_cluster_t=[center[i]-z*se[i],center[i]+z*se[i]],family28_ci95_group_bootstrap=bounds[2:,i],
            family28_ci95_cluster_t=[center[i]-familyz*se[i],center[i]+familyz*se[i]])
    return result


def effects():
    protocol=read(OUT/'TRAIN_PROTOCOL.json');evaluation=read(OUT/'EVAL_PROTOCOL.json');result=read(OUT/'RESULTS.json')
    counts=defaultdict(int);errors=defaultdict(float)
    for item in evaluation['code_and_input_files']:
        require(sha(resolve(item['path']))==item['sha256'],'Evaluation-bound digest');counts['evaluation_bound_files']+=1
    for name in ('CANDIDATES_SEALED.json','RANKINGS_SEALED.json'):
        seal=read(OUT/name)
        for item in seal['files']:
            require(sha(resolve(item['path']))==item['sha256'],'Inference digest');counts['inference_files']+=1
    reported={(r['domain'],r['view'],r['arm'],r['query_id']):r for r in rows(OUT/'per_query.jsonl')}
    expected_rr={};independent={};seeds=read(ROOT/'results/rene_ssqr_bridge_20261019/PROTOCOL.json')['rand_seeds']
    selections={(r['domain'],r['view'],r['arm'],r['query_id']):r for r in rows(OUT/'selections.jsonl')}
    for domain in ('source','cosqa'):
        qrows=rows(OUT/f'data/{domain}_queries.jsonl');queries={r['query_id']:r['query'] for r in qrows}
        ids=list(queries);groupmap=protocol['samples'][domain]['groups']
        qrels={r['query_id']:r['doc_id'] for r in rows(ROOT/f'results/rene_ssqr_bridge_20261019/data/private/{domain}_qrels.jsonl')}
        rankings={r['text_id']:r for r in rows(OUT/f'rankings/{domain}/Top100.jsonl')}
        for key,row in rankings.items():
            require(hashlib.sha256(row['text'].encode()).hexdigest()==key,'Ranking text identity')
            require(all(a>=b for a,b in zip(row['scores'],row['scores'][1:])),'Sorted rankings')
            counts['ranking_texts']+=1
        for view in ('own','fixed'):
            for arm in protocol['arms']:
                pools=defaultdict(list)
                for r in rows(OUT/f'candidates/{arm}/{domain}_{view}.jsonl'):pools[r['query_id']].append(r)
                for qid in ids:
                    pool=pools[qid];bypos={r['position']:r for r in pool}
                    require(sorted(bypos)==list(range(1,len(queries[qid].split())+1)),'Complete same position scope')
                    entropy=sorted(bypos,key=lambda i:(bypos[i]['full_vocab_entropy'],i))
                    probability=sorted(bypos,key=lambda i:(-bypos[i]['label_mean_log_probability'],i))
                    choice=selections[domain,view,arm,qid]
                    require(choice['ENTR']==entropy and choice['PROB']==probability,'Outcome-blind selection rederivation')
                    scores={}
                    for pos,r in bypos.items():
                        words=queries[qid].split()
                        require(r['masked_query']==' '.join(words[:pos]+['<extra_id_0>']+words[pos:]),'Masked query identity')
                        require(r['rewrite']==' '.join(words[:pos]+([r['span']] if r['span'] else [])+words[pos:]),'Rewrite identity')
                        key=hashlib.sha256(r['rewrite'].encode()).hexdigest()
                        scores[pos]=exact_rr(rankings[key]['ranked_doc_ids'],qrels[qid])
                        expected_rr[domain,view,arm,qid,pos]=(key,float(scores[pos]));counts['Fraction_candidate_RR']+=1
                    original=exact_rr(rankings[hashlib.sha256(queries[qid].encode()).hexdigest()]['ranked_doc_ids'],qrels[qid])
                    metrics=dict(NoRewrite=original,RAND_exact_top1=sum(scores.values(),Fraction(0))/len(scores),
                        RAND_exact_best3=cdf_max(list(scores.values())),Oracle_all=max(scores.values()))
                    for policy,order_ in [('ENTR',entropy),('PROB',probability)]:
                        metrics[policy+'_top1']=scores[order_[0]];metrics[policy+'_best3']=max(scores[i] for i in order_[:3])
                        metrics[policy+'_mean3']=sum((scores[i] for i in order_[:3]),Fraction(0))/min(3,len(order_))
                    random_maxima=[]
                    for seed in seeds:
                        digest=hashlib.sha256(f'{seed}\0{qid}'.encode()).digest()
                        order_=random.Random(int.from_bytes(digest,'big')).sample(sorted(bypos),len(bypos))
                        random_maxima.append(max(scores[pos] for pos in order_[:3]))
                    metrics['RAND20_best3']=sum(random_maxima,Fraction(0))/20
                    fx=dict(S3=metrics['ENTR_best3']-metrics['RAND_exact_best3'],S1=metrics['ENTR_top1']-metrics['RAND_exact_top1'],
                        top1_gain=metrics['ENTR_top1']-metrics['NoRewrite'],best3_gain=metrics['ENTR_best3']-metrics['NoRewrite'],
                        prob_S3=metrics['PROB_best3']-metrics['RAND_exact_best3'],ENTR_minus_PROB3=metrics['ENTR_best3']-metrics['PROB_best3'],
                        J=(metrics['ENTR_top1']-metrics['RAND_exact_top1'])-(metrics['ENTR_best3']-metrics['RAND_exact_best3']),
                        S3_RAND20=metrics['ENTR_best3']-metrics['RAND20_best3'])
                    rec=reported[domain,view,arm,qid]
                    require(rec['group']==groupmap[qid] and rec['positions']==len(pool),'Query metadata')
                    for label,values in [('RR',metrics),('effects',fx)]:
                        require(set(rec[label])==set(values),'Metric names')
                        for name,value in values.items():
                            error=abs(rec[label][name]-float(value));errors[label]=max(errors[label],error)
                            require(error<1e-12,'Fraction metric mismatch');counts['query_metric_fields']+=1
                    independent[domain,view,arm,qid]=dict(RR={k:float(v) for k,v in metrics.items()},effects={k:float(v) for k,v in fx.items()})
        # Every new dense retrieval score, independently reproduce Top100 ordering.
        manifest=read(OUT/f'rankings/{domain}/MANIFEST.json')
        texts=read(OUT/f'rankings/{domain}/new_texts.json')
        dense=np.load(STORE/f'{domain}_retrieval_scores.npy',mmap_mode='r')
        corpus_path=ROOT/'results/rene_ssqr_bridge_20261019/data/source/corpus.jsonl' if domain=='source' else ROOT/'results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl'
        doc_ids=np.array([r['doc_id'] for r in rows(corpus_path)])
        require(dense.shape==(len(texts),len(doc_ids)),'Dense retrieval shape')
        for text,values in zip(texts,dense):
            key=hashlib.sha256(text.encode()).hexdigest();order_=np.lexsort((doc_ids,-values))[:100]
            require(rankings[key]['ranked_doc_ids']==doc_ids[order_].tolist(),'Independent full-score order')
            require(np.array_equal(np.array(rankings[key]['scores'],np.float32),values[order_]),'Persisted score mapping')
            counts['full_score_rows']+=1;counts['full_scores']+=len(doc_ids)
    exported=rows(OUT/'candidate_RR.jsonl')
    require(len(exported)==len(expected_rr),'Complete candidate export')
    require(len(reported)==len(independent),'Complete query export')
    seen=set()
    for r in exported:
        signature=(r['domain'],r['view'],r['arm'],r['query_id'],r['position'])
        require(signature not in seen,'Duplicate candidate export');seen.add(signature)
        key,value=expected_rr[r['domain'],r['view'],r['arm'],r['query_id'],r['position']]
        require(key==r['text_id'] and abs(value-r['RR'])<1e-12,'Candidate export');counts['candidate_export_rows']+=1
    for domain in ('source','cosqa'):
        ids=protocol['samples'][domain]['query_ids'];groups=[protocol['samples'][domain]['groups'][q] for q in ids]
        for view in ('own','fixed'):
            for arm in protocol['arms']:
                sample=[independent[domain,view,arm,q] for q in ids]
                names=list(sample[0]['RR'])+list(sample[0]['effects'])
                matrix=[[*(r['RR'][k] for k in sample[0]['RR']),*(r['effects'][k] for k in sample[0]['effects'])] for r in sample]
                statistics=stat_reference(matrix,groups,names)
                claimed=next(r for r in result['views'] if r['domain']==domain and r['view']==view and r['arm']==arm)
                for label in ('MRR','effects'):
                    for name,fields in claimed[label].items():
                        for field,value in fields.items():
                            error=float(np.max(np.abs(np.asarray(value)-np.asarray(statistics[name][field]))))
                            require(error<1e-11,'Independent absolute statistic');errors['statistics']=max(errors['statistics'],error);counts['statistical_fields']+=1
            names=[f'{a}:{m}' for a in protocol['arms'] if a!='L' for m in ('S3','top1_gain','S1','J','prob_S3')]
            matrix=[[independent[domain,view,a,q]['effects'][m]-independent[domain,view,'L',q]['effects'][m]
                for a in protocol['arms'] if a!='L' for m in ('S3','top1_gain','S1','J','prob_S3')] for q in ids]
            statistics=stat_reference(matrix,groups,names)
            records=[r for r in result['primary_changes']+result['descriptive_changes'] if r['domain']==domain and r['view']==view and r['reference']=='L']
            for rec in records:
                expected=statistics[rec['arm']+':'+rec['metric']]
                for field in expected:
                    if field not in rec:continue
                    error=float(np.max(np.abs(np.asarray(rec[field])-np.asarray(expected[field]))))
                    require(error<1e-11,'Independent between-arm statistic');errors['statistics']=max(errors['statistics'],error);counts['contrast_fields']+=1
            matrix=[[independent[domain,view,'PS',q]['effects'][m]-independent[domain,view,'P',q]['effects'][m] for m in ('S3','top1_gain')] for q in ids]
            seed_reference=stat_reference(matrix,groups,['S3','top1_gain'])
            for rec in result['descriptive_changes']:
                if rec['domain']!=domain or rec['view']!=view or rec['reference']!='P':continue
                for field,value in seed_reference[rec['metric']].items():
                    if field not in rec:continue
                    require(float(np.max(np.abs(np.asarray(rec[field])-np.asarray(value))))<1e-11,'Seed sensitivity statistic')
                    counts['contrast_fields']+=1
    require(len(result['primary_changes'])==28,'Primary comparison family')
    require(not result['new_confirmation'] and not result['risk_certified'],'Evidence role')
    write('EFFECT_AUDIT.json',dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),independent_no_production_imports=True,
        counts=dict(counts),max_errors=dict(errors),results_sha256=sha(OUT/'RESULTS.json'),
        per_query_sha256=sha(OUT/'per_query.jsonl'),new_confirmation=False,risk_certified=False))
    print(dict(stage='effect_audit_passed',counts=dict(counts),max_errors=dict(errors)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['training','effects'])
    with threadpool_limits(limits=4):{'training':training,'effects':effects}[parser.parse_args().mode]()
