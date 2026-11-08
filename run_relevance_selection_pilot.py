"""Fixed-pool, valid-only document relevance and utility-selection pilot.

This is a predeclared development experiment on a previously viewed valid pool.
Independent code-only lexical and frozen encoder evidence cannot access gold IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

import joblib
import lightgbm as lgb
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import torch

from build_rewrite_utility_dataset import code_text
from crsu.core import RankingStates, rrf
from run_bm25_acquisition_pilot import BM25, tokens
import run_t5_seed_uq_pilot as old
from run_uq_pilot import booster, dump, finite_quantile, grouped_folds


OUT = Path('results/uq_pilot_20261012/relevance_selection')
SOURCE = old.OUT
ENCODER = Path('models/codebert-base')
SEED = 11120
SIGNALS = ('token_maxsim', 'bm25', 'idf_overlap', 'pooled_cosine')
STATS = ('action_top1', 'action_discounted', 'original_discounted',
         'discounted_delta', 'promoted_mean', 'demoted_mean', 'shift_evidence')
FEATURES = [f'{s}_{t}' for s in SIGNALS for t in STATS]
DEPENDENCIES = [Path(p) for p in (
    'run_t5_seed_uq_pilot.py', 'run_uq_pilot.py', 'run_bm25_acquisition_pilot.py',
    'build_rewrite_utility_dataset.py', 'build_crsu_pilot.py',
    'build_enhanced_retrieval_feedback_dataset.py', 'prepare_vac_mqr_cv_folds.py',
    'crsu/core.py', 'vac_mqr.py')]


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {str(SOURCE/p): old.sha256(SOURCE/p) for p in
              ('PROTOCOL.json', 'evidence.pkl', 'generation.jsonl', 'summary.json', 'predictions.pkl')}
    assets = {str(p): old.sha256(p) for p in ENCODER.iterdir() if p.is_file()}
    if (OUT/'PROTOCOL.json').exists():
        p = json.loads((OUT/'PROTOCOL.json').read_text(encoding='utf-8'))
        assert p['inputs'] == inputs and p['encoder_assets'] == assets
        assert p['source_sha256'] == old.sha256(Path(__file__))
        assert p['stable_sha256'] == old.sha256(old.STABLE)
        assert p['valid_sha256'] == old.sha256(old.VALID)
        assert all(old.sha256(Path(f)) == h for f,h in p['dependencies'].items())
        return p
    old.prepare()
    p = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), development_only=True,
             exploratory=True, historically_viewed_valid=True, test_used=False,
             inputs=inputs, encoder_assets=assets, source_sha256=old.sha256(Path(__file__)),
             dependencies={str(f):old.sha256(f) for f in DEPENDENCIES},
             stable_sha256=old.sha256(old.STABLE), valid_sha256=old.sha256(old.VALID),
             candidate_pool='unchanged multi8; all 9761 existing unique candidates',
             roles=old.COUNTS, encoder='frozen codebert-base pretrained, eval, no fine-tuning',
             document_selection='union of actual original/rewrite/fusion Top10 only; never add gold',
             semantic=dict(max_query_tokens=64, max_code_tokens=128, projection_dim=64,
                           projection_seed=12012, token_mask='exclude special and padding',
                           score='mean over original query tokens of max projected cosine to code tokens'),
             lexical='original query BM25 and IDF-weighted token coverage; code_tokens only, no docstring',
             document_signals=list(SIGNALS), features=FEATURES,
             arms=['base_point','relevance_point','base_pair','relevance_pair','base_mean_relevance_scale'],
             point='same query-weighted LightGBM 7 leaves/min30/120 rounds/seed11120 as previous pilot',
             pair='point squared error + within-query pair-difference squared error, lambda=1',
             scale='4-fold repo OOF log(abs(residual)+.001), no held-out labels in fitting',
             primary='relevance_pair vs base_point, mean-ranked 10% adoption',
             ablations=['relevance_point vs base_point','base_pair vs base_point',
                        'relevance_pair vs relevance_point','fixed base mean + relevance scale'],
             lower='independent LCB role; same first-mean-action policy; alpha .1',
             risk='selection role chooses harm threshold from .05/.1/.2/.4/1, risk checked once',
             caveat='query-binomial descriptive risk, not repo-dependence/multiple-policy certification')
    dump(OUT/'PROTOCOL.json',p)
    for f in [Path(__file__),*DEPENDENCIES]:
        relative=Path(f.name) if f==Path(__file__) else f
        dest=OUT/'frozen_source'/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(f.read_bytes())
    return p


def action_evidence(original, action, document_ids, signals):
    """Pure label-free evidence from an original query and actual result lists."""
    original=np.asarray(original[:10],int); action=np.asarray(action[:10],int)
    lookup={int(d):i for i,d in enumerate(document_ids)}
    oi=np.array([lookup[int(d)] for d in original]); ai=np.array([lookup[int(d)] for d in action])
    shift=np.zeros(len(document_ids))
    discount=1/np.arange(1,11,dtype=float)
    shift[ai]+=discount; shift[oi]-=discount
    pos=shift>old.TOL; neg=shift<-old.TOL
    ow=discount@signals[oi]/discount.sum(); aw=discount@signals[ai]/discount.sum()
    promoted=(shift[pos]@signals[pos])/shift[pos].sum() if pos.any() else np.zeros(len(SIGNALS))
    demoted=(-shift[neg]@signals[neg])/(-shift[neg]).sum() if neg.any() else np.zeros(len(SIGNALS))
    out=np.stack((signals[ai[0]],aw,ow,aw-ow,promoted,demoted,shift@signals),axis=1).ravel()
    assert out.shape==(len(FEATURES),) and np.isfinite(out).all()
    return out


def project_encode(model, tokenizer, texts, length, projection):
    result=[]; mask=[]; means=[]
    for start in range(0,len(texts),32):
        batch=tokenizer(texts[start:start+32],max_length=length,padding='max_length',
                        truncation=True,return_tensors='pt',return_special_tokens_mask=True)
        valid=(batch.pop('special_tokens_mask')==0)&batch['attention_mask'].bool()
        encoded={k:v.to('cuda') for k,v in batch.items()}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=model(**encoded).last_hidden_state.float()
            projected=hidden@projection
            projected=torch.nn.functional.normalize(projected,dim=-1)
            validgpu=valid.to('cuda')
            pooled=(hidden*validgpu[:,:,None]).sum(1)/validgpu.sum(1).clamp_min(1)[:,None]
            pooled=torch.nn.functional.normalize(pooled,dim=-1)
        result.append(projected.half().cpu().numpy());mask.append(valid.numpy())
        means.append(pooled.cpu().numpy())
    return np.concatenate(result),np.concatenate(mask),np.concatenate(means)


def build_evidence(protocol):
    if (OUT/'relevance_evidence.pkl').exists():
        print('Relevance evidence already complete',flush=True);return
    started=time.perf_counter()
    items=pickle.load((SOURCE/'evidence.pkl').open('rb'))
    rows=old.read_jsonl(SOURCE/'generation.jsonl'); valid=old.read_jsonl(old.VALID)
    assert [x['query_index'] for x in items]==[x['query_index'] for x in rows]
    per_query=[]; all_docs=set()
    for e in items:
        docs=set(map(int,e['original_top100'][:10]))
        docs.update(int(d) for t in e['candidate_top100'] for d in t[:10])
        docs.update(int(d) for t in e['candidate_top100'] for d in rrf(e['original_top100'],t)[:10])
        ids=np.array(sorted(docs));per_query.append(ids);all_docs.update(docs)
    doc_ids=np.array(sorted(all_docs)); lookup={int(d):i for i,d in enumerate(doc_ids)}
    print(f'Encode independent evidence: {len(doc_ids)} code documents, {len(rows)} queries',flush=True)
    from transformers import AutoModel,AutoTokenizer
    model=AutoModel.from_pretrained(ENCODER).to('cuda').eval()
    tokenizer=AutoTokenizer.from_pretrained(ENCODER)
    rng=np.random.default_rng(12012)
    projection=torch.tensor(rng.normal(size=(model.config.hidden_size,64))/8,
                            dtype=torch.float32,device='cuda')
    qt,qm,qmean=project_encode(model,tokenizer,[r['query'] for r in rows],64,projection)
    dt,dm,dmean=project_encode(model,tokenizer,[code_text(valid[int(i)]) for i in doc_ids],128,projection)
    del model;torch.cuda.empty_cache()
    np.save(OUT/'projection.npy',projection.cpu().numpy())
    index=BM25([code_text(r) for r in valid])
    termsets=[set(tokens(code_text(valid[int(d)]))) for d in doc_ids]
    idf={word:float(np.log1p((index.n-len(ids)+.5)/(len(ids)+.5)))
         for word,(ids,_) in index.posting.items()}
    evidence=[]
    for local,(e,row,ids) in enumerate(zip(items,rows,per_query)):
        di=np.array([lookup[int(d)] for d in ids])
        scores=np.zeros(len(valid));terms=set(tokens(row['query']))
        for term in terms:
            if term in index.posting:
                idx,w=index.posting[term];scores[idx]+=w
        bm25=scores[ids]/max(scores.max(),1e-12)
        denom=sum(idf.get(t,0) for t in terms)
        lexical=np.array([sum(idf.get(t,0) for t in terms&termsets[i])/max(denom,1e-12) for i in di])
        q=torch.tensor(qt[local,qm[local]].astype(np.float32),device='cuda')
        d=torch.tensor(dt[di].astype(np.float32),device='cuda')
        dmask=torch.tensor(dm[di],device='cuda')
        with torch.inference_mode():
            sim=torch.einsum('qh,dlh->dql',q,d)
            sim=sim.masked_fill(~dmask[:,None,:],-1.)
            maxsim=(sim.max(-1).values.mean(-1).cpu().numpy() if len(q)
                    else np.zeros(len(di)))
        pooled=dmean[di]@qmean[local]
        signals=np.column_stack((maxsim,bm25,lexical,pooled))
        assert np.isfinite(signals).all()
        extras={f:np.array([action_evidence(e['original_top100'],t if f=='rewrite' else rrf(e['original_top100'],t),ids,signals)
                           for t in e['candidate_top100']]).reshape(len(e['x']),len(FEATURES))
                for f in ('rewrite','fusion')}
        # Label audit operates separately from the evidence input construction.
        for family in extras:
            lists={str(j):t if family=='rewrite' else rrf(e['original_top100'],t)
                   for j,t in enumerate(e['candidate_top100'])}
            st=RankingStates.from_rankings(e['original_top100'],lists)
            np.testing.assert_allclose(st.gains[st.target_index(e['query_index'])],e['y'][family],atol=old.TOL,rtol=0)
        evidence.append(dict(query_index=e['query_index'],doc_ids=ids,doc_signals=signals,x=extras))
        if (local+1)%80==0:
            print(f'Relevance {local+1}/{len(items)}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    with (OUT/'relevance_evidence.pkl').open('wb') as stream:pickle.dump(evidence,stream)
    dump(OUT/'evidence_status.json',dict(test_used=False,queries=len(items),documents=len(doc_ids),
         query_document_pairs=sum(len(d) for d in per_query),label_checks=2*sum(len(e['x']) for e in items),
         label_mismatches=0,elapsed_seconds=time.perf_counter()-started))


def data_for(items,evidence,family):
    roles={}
    for role in old.COUNTS:
        roleitems=[x for x in items if x['role']==role]
        roles[role]=old.flatten(roleitems,family)
        roles[role]['base']=np.c_[roles[role]['x'],roles[role]['seed_x']]
        roles[role]['full']=np.c_[roles[role]['base'],np.concatenate(
            [z['x'][family] for x,z in zip(items,evidence) if x['role']==role])]
        assert np.isfinite(roles[role]['full']).all()
    reps={r:set(d['query_repos']) for r,d in roles.items()}
    assert all(not reps[a]&reps[b] for a in reps for b in reps if a!=b)
    return roles


def pair_derivatives(pred,y,q,w):
    error=pred-y
    _,inverse,counts=np.unique(q,return_inverse=True,return_counts=True)
    mean=np.bincount(inverse,weights=error)/counts
    # Jq = mean(e^2)/2 + sum_{i<j}(ei-ej)^2/(2*nq^2).
    # Point term anchors absolute Delta-RR; pair term stresses candidate order.
    gradient=w*(2*error-mean[inverse])
    hessian=w*(2-1/counts[inverse])
    return gradient,hessian


def mean_fit(x,y,q,w,pair):
    if not pair:return dict(model=booster(x,y,SEED,weight=w),intercept=0.)
    def objective(pred,dataset):
        return pair_derivatives(pred,dataset.get_label(),q,w)
    params=dict(objective=objective,verbosity=-1,num_leaves=7,min_data_in_leaf=30,
                learning_rate=.04,lambda_l2=1,num_threads=2,seed=SEED,
                deterministic=True,force_col_wise=True)
    intercept=float(np.average(y,weights=w))
    model=lgb.train(params,lgb.Dataset(x,label=y,init_score=np.full(len(y),intercept)),num_boost_round=120)
    return dict(model=model,intercept=intercept)


def mean_predict(model,x):
    return model['model'].predict(x)+model['intercept']


def fit_models(data):
    models={};oofs={};details={}
    for feature in ('base','full'):
        for pair in (False,True):
            name=('base' if feature=='base' else 'relevance')+('_pair' if pair else '_point')
            x=data[feature];y=data['y'];q=data['q'];w=data['weights']
            oof=np.full(len(y),np.nan); folds=grouped_folds(data['repo'],4,SEED)
            print(f'Fit {name}: {len(data["query_ids"])} queries / {len(y)} actions, {x.shape[1]} features',flush=True)
            detail=[]
            for val in folds:
                tr=np.setdiff1d(np.arange(len(y)),val)
                assert not set(data['repo'][val])&set(data['repo'][tr])
                m=mean_fit(x[tr],y[tr],q[tr],w[tr],pair);oof[val]=mean_predict(m,x[val])
                detail.append(dict(fit_repos=sorted(set(data['repo'][tr])),held_repos=sorted(set(data['repo'][val]))))
            assert np.isfinite(oof).all()
            residual=np.log(abs(y-oof)+.001)
            models[name]=dict(mean=mean_fit(x,y,q,w,pair),scale=booster(x,residual,SEED,weight=w),
                              harm=booster(x,(y<-old.TOL).astype(int),SEED,'binary',w),
                              mean_feature=feature,scale_feature=feature,harm_feature=feature)
            oofs[name]=oof;details[name]=detail
    models['base_mean_relevance_scale']=dict(models['base_point'])
    models['base_mean_relevance_scale'].update(
        scale=booster(data['full'],np.log(abs(data['y']-oofs['base_point'])+.001),SEED,weight=data['weights']),
        scale_feature='full')
    return models,oofs,details


def predict(model,data):
    return dict(mu=mean_predict(model['mean'],data[model['mean_feature']]),
                s=np.maximum(np.exp(model['scale'].predict(data[model['scale_feature']])),.001),
                harm=np.clip(model['harm'].predict(data[model['harm_feature']]),0,1))


def selection_diagnostics(data,pred):
    chosen=old.first(data,pred,'multi8');positive=[];oracle=[];pair_acc=[];regrets=[]
    for q in data['query_ids']:
        ix=np.flatnonzero(data['q']==q)
        if not len(ix):continue
        yy=data['y'][ix];mu=pred['mu'][ix];best=ix[int(np.argmax(mu))]
        top=max(float(yy.max()),0.);oracle.append(top);positive.append(max(float(data['y'][best]),0.))
        regrets.append(top-max(float(data['y'][best]),0.))
        diff=yy[:,None]-yy[None,:];keep=diff>old.TOL
        if keep.any():
            score=mu[:,None]-mu[None,:]
            pair_acc.append(float(np.mean((score[keep]>old.TOL)+.5*(abs(score[keep])<=old.TOL))))
    y=data['y'];err=abs(y-pred['mu']);w=data['weights']
    def correlation(mask):
        if mask.sum()<3:return None
        z=stats.spearmanr(pred['s'][mask],err[mask]).statistic
        return float(z) if np.isfinite(z) else None
    rr=sum(oracle)
    return dict(query_equal_mae=float(np.average(err,weights=w)),
        query_equal_harm_brier=float(np.average((pred['harm']-(y<-old.TOL))**2,weights=w)),
        positive_auprc=float(average_precision_score(y>old.TOL,pred['mu'])) if np.any(y>old.TOL) else None,
        within_query_pair_accuracy=float(np.mean(pair_acc)) if pair_acc else None,
        positive_opportunity_retention=sum(positive)/rr if rr else None,
        first_choice_oracle_adoption_ceiling=old.decision(data,chosen[y[chosen]>old.TOL]),
        all_candidate_oracle_delta_mrr=rr/data['query_count'],
        mean_opportunity_regret=float(sum(regrets)/data['query_count']),
        scale_error_rho=correlation(np.ones(len(y),bool)),scale_error_rho_nonzero=correlation(abs(y)>old.TOL))


def evaluate(protocol):
    if (OUT/'summary.json').exists():print('Evaluation complete',flush=True);return
    items=pickle.load((SOURCE/'evidence.pkl').open('rb'))
    evidence=pickle.load((OUT/'relevance_evidence.pkl').open('rb'))
    assert [e['query_index'] for e in items]==[e['query_index'] for e in evidence]
    previous=pickle.load((SOURCE/'predictions.pkl').open('rb'))
    result=dict(test_used=False,development_only=True,exploratory=True,
                protocol_sha256=old.sha256(OUT/'PROTOCOL.json'),families={})
    saved={}
    for family in ('rewrite','fusion'):
        roles=data_for(items,evidence,family);train=roles['utility_train'];ev=roles['new_holdout']
        models,oofs,fit_roles=fit_models(train)
        joblib.dump(models,OUT/f'{family}_models.joblib');restored=joblib.load(OUT/f'{family}_models.joblib')
        predictions={name:{r:predict(m,d) for r,d in roles.items() if r!='utility_train'} for name,m in models.items()}
        # Baseline exactly replays yesterday's same-candidate seed-feature model.
        for k in ('mu','s','harm'):
            np.testing.assert_allclose(predictions['base_point']['new_holdout'][k],
                                       previous[f'{family}__multi8__seed_features'][k],atol=old.TOL,rtol=0)
        for r in predictions['base_point']:
            for k in ('mu','harm'):
                np.testing.assert_array_equal(predictions['base_point'][r][k],predictions['base_mean_relevance_scale'][r][k])
        out=dict(methods={},comparisons={},utility_train_oof={})
        for name,m in models.items():
            ps=predictions[name];cal=roles['lcb_calibration'];pc=ps['lcb_calibration']
            replay=predict(restored[name],cal)
            for k in replay:np.testing.assert_array_equal(replay[k],pc[k])
            ci=old.first(cal,pc,'multi8')
            correction=max(finite_quantile((pc['mu'][ci]-cal['y'][ci])/pc['s'][ci]),0.)
            constant=max(finite_quantile(pc['mu'][ci]-cal['y'][ci]),0.)
            assert np.isfinite(correction) and np.isfinite(constant)
            pred={r:dict(v,lower=v['mu']-correction*v['s'],constant_lower=v['mu']-constant) for r,v in ps.items()}
            pe=pred['new_holdout'];ix=old.first(ev,pe,'multi8')
            residual=ev['y'][ix]-pe['lower'][ix]
            metrics=selection_diagnostics(ev,pe)
            metrics.update(normalized_correction=correction,
                first_choice_lcb_violation=float(np.mean(residual<0)),
                first_choice_q10_pinball=float(np.mean(np.maximum(.1*residual,-.9*residual))),
                positive_lcb=old.decision(ev,ix[pe['lower'][ix]>0]),
                curves={s:{str(c):old.curve(ev,pe,'multi8',z,c) for c in (.05,.1,.2)}
                        for s,z in (('mean',pe['mu']),('lcb',pe['lower']),('constant_lcb',pe['constant_lower']),('negative_harm',-pe['harm']))},
                risk_mean=old.risk_policy(roles,pred,'multi8','mean'),
                risk_lcb=old.risk_policy(roles,pred,'multi8','lcb'))
            out['methods'][name]=metrics;saved[f'{family}__{name}']=pred
            if name in oofs:
                op=dict(mu=oofs[name],s=np.ones(len(train['y'])),harm=np.zeros(len(train['y'])))
                out['utility_train_oof'][name]=selection_diagnostics(train,op)
        comparisons=(('evidence_only','relevance_point','base_point','mean'),
                     ('pair_objective_only','base_pair','base_point','mean'),
                     ('joint_primary','relevance_pair','base_point','mean'),
                     ('pair_with_evidence','relevance_pair','relevance_point','mean'),
                     ('uq_same_mean','base_mean_relevance_scale','base_point','lcb'))
        for label,a,b,score in comparisons:
            da=out['methods'][a]['curves'][score]['0.1']['query_values']
            db=out['methods'][b]['curves'][score]['0.1']['query_values']
            difference={q:da.get(q,0)-db.get(q,0) for q in ev['query_ids']}
            out['comparisons'][label]=dict(delta_difference=sum(difference.values())/ev['query_count'],
                repo_ci=old.clustered_ci(ev,difference,SEED,1000),
                caveat='exploratory, fixed fitted models/acceptance sets, no multiple-comparison correction')
        out['fit_repository_roles']=fit_roles
        result['families'][family]=out
    with (OUT/'predictions.pkl').open('wb') as stream:pickle.dump(saved,stream)
    assert old.sha256(old.STABLE)==protocol['stable_sha256']
    dump(OUT/'summary.json',result)
    dump(OUT/'reproducibility.json',dict(test_used=False,stable_unchanged=True,
         prior_baseline_replayed=True,repository_roles_disjoint=True,model_roundtrip=True,
         fixed_mean_scale_ablation_exact=True,source_sha256=old.sha256(Path(__file__)),
         summary_sha256=old.sha256(OUT/'summary.json'),
         evidence_sha256=old.sha256(OUT/'relevance_evidence.pkl')))
    lines=['# 固定候选池：代码相关性证据与收益选择','',
           '480个历史valid开发query，评估128；不作新test结论，所有特征来自原查询和代码，无gold身份输入。','',
           '| 动作 | 模型 | mean ΔMRR@约10% | harm | LCB ΔMRR@约10% | harm | 正收益机会保留 | 正LCB数 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for f,o in result['families'].items():
        for name,m in o['methods'].items():
            a=m['curves']['mean']['0.1'];b=m['curves']['lcb']['0.1']
            lines.append(f"| {f} | {name} | {a['delta_mrr']:+.6f} | {a['harmful_rate']:.3f} | {b['delta_mrr']:+.6f} | {b['harmful_rate']:.3f} | {m['positive_opportunity_retention']:.3f} | {m['positive_lcb']['accepted']} |")
    lines+=['','## 配对比较','','| 动作 | 比较 | ΔMRR差 | repo 95% CI |','|---|---|---:|---|']
    for f,o in result['families'].items():
        for k,z in o['comparisons'].items():lines.append(f"| {f} | {k} | {z['delta_difference']:+.6f} | {z['repo_ci']} |")
    lines+=['','风险检查与proper score见summary.json；有限校准样本、历史valid开发复用、repository依赖与多策略比较限制仍适用。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({f:o['comparisons'] for f,o in result['families'].items()},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('prepare','evidence','evaluate','all'),default='all')
    args=parser.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        protocol=freeze()
        if args.stage in ('all','evidence'):build_evidence(protocol)
        if args.stage in ('all','evaluate'):evaluate(protocol)


if __name__=='__main__':main()
