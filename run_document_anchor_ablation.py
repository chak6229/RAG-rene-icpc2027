"""Complete the omitted document-score x success-anchor direct-head ablation.

Fixed old pool/folds/architecture. Original baseline and joint replay exactly;
fit only position/doc/anchor arms, with honest upstream and residual crossfit.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import pickle
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import run_nested_relevance_adoption_pilot as base
from run_uq_pilot import booster,clustered_ci,dump,grouped_folds

OUT=Path('results/uq_pilot_20261012/document_anchor_ablation')
ARMS=('baseline','position','document','anchor','joint')
MASKS=dict(position=(8,),document=tuple(range(7))+(8,),anchor=(7,8,9),joint=tuple(range(10)))
PAIRS=dict(document_without_anchor=('document','position'),
    document_with_anchor=('joint','anchor'),anchor_without_document=('anchor','position'),
    anchor_with_document=('joint','document'),position_vs_baseline=('position','baseline'),
    joint_vs_baseline=('joint','baseline'))
DEPENDENCIES=[Path(p) for p in ('run_nested_relevance_adoption_pilot.py','run_relevance_selection_pilot.py',
    'run_t5_seed_uq_pilot.py','run_uq_pilot.py','crsu/core.py','tests/test_document_anchor_ablation.py',
    'analyze_document_anchor_ablation.py')]


def freeze(items):
    OUT.mkdir(parents=True,exist_ok=True);base.freeze(items)
    paths=[base.SOURCE,base.EVIDENCE,base.OUT/'PROTOCOL.json',base.OUT/'summary.json',base.OUT/'predictions.pkl']
    paths+=[base.OUT/f'{family}_fold{f}_models.joblib' for family in ('rewrite','fusion') for f in range(5)]
    fixed=dict(inputs={str(p):base.old.sha256(p) for p in paths},
        dependencies={str(p):base.old.sha256(p) for p in DEPENDENCIES},
        source_sha256=base.old.sha256(Path(__file__)),stable_sha256=base.old.sha256(base.old.STABLE),
        fold_query_ids=[{r:[items[int(i)]['query_index'] for i in ix] for r,ix in plan.items()} for plan in base.splits(items)])
    if (OUT/'PROTOCOL.json').exists():
        p=json.loads((OUT/'PROTOCOL.json').read_text(encoding='utf8'))
        assert all(p[k]==v for k,v in fixed.items()),'frozen ablation changed';return p
    p=dict(fixed,frozen_utc=datetime.now(timezone.utc).isoformat(),test_used=False,exploratory=True,
        history='same repeatedly developed valid480 queries, not fresh confirmation',
        candidate_pool='same9761 candidates, original/rewrite/actual equal-weightRRF Top100 labels',
        upstream='same document logistic C1 and Top1-success logistic C.2, selector-fit only',
        document='seven learned document probability/shift stats; scorer contains original ranks and lexical/semantic signals, not pure semantic relevance',
        anchor='predicted original Top1 correctness plus probability x observable Top1 demotion',
        structural_control='raw Top1 demotion present in position/document/anchor/joint; position control needed to isolate supervised heads',
        feature_masks={k:list(v) for k,v in MASKS.items()},feature_names=base.BASE_NAMES+base.EXTRA_NAMES,
        capacity='all factorial arms84 columns with excluded blocks zero; same booster7leaves/min30/120rounds/seeds; effective signal dimensions differ',
        fitting='selector156 only, repo4-fold upstream crossfit and residual OOF that refits upstream in each fold',
        replay='baseline74 and joint84 previous models/predictions/selected-actions/risk results exact, no refit',
        primary='factorial incremental document/anchor effects on mean-ranked10/fold adoption plus interaction',
        budget='exact10/96 per fold, total50/480; same mean/LCB budgets, full query denominator',
        secondary='candidate sorting/opportunity, action-level MAE/Brier, LCB/constant ranking same candidate choice',
        risk='same independentLCB38/selection38/risk76/eval96; adoption76 unused; mean andLCB gate, prior grid no retuning',
        uncertainty='same selected-action normalized/constant conformal; does not estimate strict epistemic uncertainty',
        certification='query-iid CP95 descriptive only; repository dependency/multiplicity not controlled, never poolfolds',
        no_case_rules=True,no_hyperparameter_search=True)
    dump(OUT/'PROTOCOL.json',p)
    for source in [Path(__file__),*DEPENDENCIES]:
        dest=OUT/'frozen_source'/source;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    return p


def mask_extra(extra,arm):
    if arm not in MASKS:raise ValueError('unknown factor arm')
    keep=np.zeros(10,bool);keep[list(MASKS[arm])]=True
    a=np.asarray(extra,float)
    if a.ndim!=2 or a.shape[1]!=10:raise ValueError('expected10 upstream features')
    return np.where(keep[None,:],a,0.)


def data_for(items,evidence,family,extras,arm):
    return base.data_for(items,evidence,family,[mask_extra(x,arm) for x in extras])


def fit_mean(items,evidence,family,arm,seed):
    extras,records=base.crossfit_extra(items,evidence,family,seed)
    evidence_model=base.fit_evidence(items,evidence)
    d=data_for(items,evidence,family,extras,arm);x=d['augmented']
    return dict(mean=booster(x,d['y'],seed,weight=d['weights']),
        harm=booster(x,(d['y']<-base.old.TOL).astype(int),seed,'binary',d['weights']),
        evidence_model=evidence_model,arm=arm,fit_query_ids=d['query_ids'],
        fit_repos=sorted(set(d['query_repos'])),upstream_oof=records),d


def inference(model,items,evidence,family):
    extras=base.predict_extra(model['evidence_model'],items,evidence,family)
    d=data_for(items,evidence,family,extras,model['arm']);x=d['augmented']
    p=dict(mu=model['mean'].predict(x),harm=np.clip(model['harm'].predict(x),0,1))
    if 'scale' in model:p['s']=np.maximum(np.exp(model['scale'].predict(x)),.001)
    return d,p


def fit_selector(items,evidence,family,arm,seed):
    offsets=np.r_[0,np.cumsum([len(e['x']) for e in items])]
    mu=np.full(offsets[-1],np.nan);records=[];reps=np.array([e['repo'] for e in items],object)
    for held in grouped_folds(reps,4,seed):
        tr=np.setdiff1d(np.arange(len(items)),held)
        assert not set(reps[tr])&set(reps[held])
        model,_=fit_mean(base.subset(items,tr),base.subset(evidence,tr),family,arm,seed)
        _,p=inference(model,base.subset(items,held),base.subset(evidence,held),family)
        ii=np.concatenate([np.arange(offsets[int(i)],offsets[int(i)+1]) for i in held]);mu[ii]=p['mu']
        records.append(dict(fit_repos=sorted(set(reps[tr])),held_repos=sorted(set(reps[held])),upstream=model['upstream_oof']))
    assert np.isfinite(mu).all()
    model,d=fit_mean(items,evidence,family,arm,seed)
    model.update(scale=booster(d['augmented'],np.log(abs(d['y']-mu)+.001),seed,weight=d['weights']),residual_oof_roles=records)
    return model


def prediction(model,items,evidence,family,arm):
    if arm in ('baseline','joint'):return base.predict_selector(model,items,evidence,family)
    return inference(model,items,evidence,family)


def selected(data,pred,arm):
    # base.selected_rows only gathers metadata beyond this fixed selector;
    # those extra metadata are not supplied to any adoption head in this run.
    return base.selected_rows(data,pred,arm!='baseline')


def budget(rows,key):
    return base.decision(rows,np.argsort(-rows[key],kind='stable')[:10])


def candidate_metrics(d,p):
    y=d['y'];w=d['weights'];out=base.selector_metrics(d,p)
    out.update(mae=float(np.average(abs(p['mu']-y),weights=w)),
        harm_brier=float(np.average((p['harm']-(y<-base.old.TOL))**2,weights=w)))
    return out


def vector(rows,decisions):
    values={}
    for d in decisions:values.update({int(q):float(v) for q,v in d['query_values'].items()})
    return {q:values.get(q,0.) for q in rows['query_ids']}


def contrasts(rows,values):
    outputs={}
    for name,(a,b) in PAIRS.items():
        delta={q:values[a][q]-values[b][q] for q in rows['query_ids']}
        outputs[name]=dict(delta_difference=sum(delta.values())/rows['query_count'],
            repo_ci=clustered_ci(rows,delta,base.SEED,3000))
    delta={q:values['joint'][q]-values['document'][q]-values['anchor'][q]+values['position'][q] for q in rows['query_ids']}
    outputs['interaction']=dict(delta_difference=sum(delta.values())/rows['query_count'],
        repo_ci=clustered_ci(rows,delta,base.SEED,3000))
    return outputs


def run(protocol,items,evidence):
    if (OUT/'summary.json').exists():print('Complete; frozen protocol verified',flush=True);return
    start=time.perf_counter();checks=base.audit_labels(items)
    prior=pickle.load((base.OUT/'predictions.pkl').open('rb'))
    previous_summary=json.loads((base.OUT/'summary.json').read_text(encoding='utf8'))
    result={};saved={}
    for family in ('rewrite','fusion'):
        all_rows={a:[] for a in ARMS};folds=[]
        for f,plan in enumerate(base.splits(items)):
            base.assert_disjoint(items,plan);train=base.subset(items,plan['selector_fit']);ztrain=base.subset(evidence,plan['selector_fit'])
            old_models=joblib.load(base.OUT/f'{family}_fold{f}_models.joblib');models={};rowsets={};outputs={};choices={}
            for arm in ARMS:
                print(f'{family} fold{f} {arm}: '+('exact replay' if arm in ('baseline','joint') else 'nested fit'),flush=True)
                if arm in ('baseline','joint'):model=old_models['baseline' if arm=='baseline' else 'learned']['selector']
                else:model=fit_selector(train,ztrain,family,arm,base.SEED+f)
                models[arm]=model;raw={};candidate_saved={}
                for role,ix in plan.items():
                    if role in ('selector_fit','adoption_fit'):continue
                    d,p=prediction(model,base.subset(items,ix),base.subset(evidence,ix),family,arm)
                    raw[role]=selected(d,p,arm)
                    if role=='new_holdout':
                        choices[arm]=candidate_metrics(d,p)
                        candidate_saved=dict(q=d['q'],y=d['y'],weights=d['weights'],pred=p,
                            inputs=d['full' if arm=='baseline' else 'augmented'])
                rows={r:base.calibration(v,raw['lcb_calibration']) for r,v in raw.items()}
                if arm in ('baseline','joint'):
                    ref='baseline_direct' if arm=='baseline' else 'learned_direct'
                    for role,v in rows.items():
                        reference=prior[f'{family}_fold{f}']['predictions'][ref][role]
                        for k in ('q','y','mu','s','harm','lower','constant_lower'):np.testing.assert_array_equal(v[k],reference[k])
                    if arm=='joint':
                        reference=prior[f'{family}_fold{f}']['selection']['learned']
                        np.testing.assert_array_equal(candidate_saved['inputs'],reference['eval_features'])
                        for k,v in candidate_saved['pred'].items():np.testing.assert_array_equal(v,reference['candidate_pred'][k])
                ev=rows['new_holdout'];rowsets[arm]=rows;all_rows[arm].append(ev)
                outputs[arm]=dict(mean10=budget(ev,'mu'),lcb10=budget(ev,'lower'),constant10=budget(ev,'constant_lower'),
                    prediction=base.prediction_metrics(ev),risk_mean=base.risk_policy(rows,'mean'),risk_lcb=base.risk_policy(rows,'lcb'))
                if arm in ('baseline','joint'):
                    ref='baseline_direct' if arm=='baseline' else 'learned_direct'
                    expected=previous_summary['families'][family]['folds'][f]['methods'][ref]
                    for key in ('risk_mean','risk_lcb'):
                        assert json.loads(json.dumps(outputs[arm][key]))==expected[key]
                    assert json.loads(json.dumps(outputs[arm]['mean10']))==expected['curves']['mean']['0.1']
                candidate_saved['choice_metrics']=choices[arm]
                if arm=='baseline':cand={}
                cand[arm]=candidate_saved
            path=OUT/f'{family}_fold{f}_models.joblib';joblib.dump(models,path)
            for arm,model in joblib.load(path).items():
                for role,ix in plan.items():
                    if role in ('selector_fit','adoption_fit'):continue
                    d,p=prediction(model,base.subset(items,ix),base.subset(evidence,ix),family,arm)
                    replay=selected(d,p,arm)
                    for k in ('q','y','mu','s','harm'):np.testing.assert_array_equal(replay[k],rowsets[arm][role][k])
            folds.append(dict(fold=f,methods=outputs,selection=choices,roles={r:len(ix) for r,ix in plan.items()}))
            saved[f'{family}_fold{f}']=dict(roles=plan,rows=rowsets,candidates=cand)
            print(f'{family} fold{f} mean10 '+str({a:round(z['mean10']['delta_mrr'],6) for a,z in outputs.items()}),flush=True)
        merged={a:base.combine_rows(v) for a,v in all_rows.items()};methods={};values={}
        for arm in ARMS:
            curves={k:base.merge_decisions(all_rows[arm],[f['methods'][arm][k] for f in folds]) for k in ('mean10','lcb10','constant10')}
            assert curves['mean10']['accepted']==50
            risk={g:dict(passing_folds=sum(f['methods'][arm]['risk_'+g]['passes_descriptive_check'] for f in folds),
                before_risk=base.merge_decisions(all_rows[arm],[f['methods'][arm]['risk_'+g]['before_risk'] for f in folds]),
                after_risk=base.merge_decisions(all_rows[arm],[f['methods'][arm]['risk_'+g]['after_risk'] for f in folds])) for g in ('mean','lcb')}
            c=[f['selection'][arm] for f in folds];oracle=sum(v['oracle_sum'] for v in c);positive=sum(v['selected_positive_sum'] for v in c)
            # Candidate metrics were query-weighted; query counts are equal per fold.
            methods[arm]=dict(curves=curves,risk=risk,prediction=base.prediction_metrics(merged[arm]),
                selection=dict(positive_opportunity_retention=positive/oracle,oracle_delta_mrr=oracle/len(items),
                    candidate_mae=float(np.mean([v['mae'] for v in c])),harm_brier=float(np.mean([v['harm_brier'] for v in c])),
                    pair_accuracy=float(np.mean([v['within_query_pair_accuracy'] for v in c]))),
                fold_delta=[f['methods'][arm]['mean10']['delta_mrr'] for f in folds])
            values[arm]=vector(merged[arm],[f['methods'][arm]['mean10'] for f in folds])
        result[family]=dict(methods=methods,contrasts=contrasts(merged['baseline'],values),folds=folds)
    with (OUT/'predictions.pkl').open('wb') as stream:pickle.dump(saved,stream)
    assert base.old.sha256(base.old.STABLE)==protocol['stable_sha256']
    dump(OUT/'summary.json',dict(test_used=False,exploratory=True,queries=len(items),families=result,
        label_checks=checks,elapsed_seconds=time.perf_counter()-start,protocol_sha256=base.old.sha256(OUT/'PROTOCOL.json')))
    dump(OUT/'POSTRUN_AUDIT.json',dict(test_used=False,stable_unchanged=True,labels_reaudited=checks,
        baseline_joint_predictions_and_risk_results_exact=True,all_models_roundtrip=True,
        nested_upstream_and_residual_crossfit=True,all_factorial_arms_fixed50=True,
        predictions_sha256=base.old.sha256(OUT/'predictions.pkl'),summary_sha256=base.old.sha256(OUT/'summary.json')))
    print(f'Completed in {time.perf_counter()-start:.1f}s',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','run'),default='run');args=p.parse_args()
    items=pickle.load(base.SOURCE.open('rb'));evidence=pickle.load(base.EVIDENCE.open('rb'))
    assert [e['query_index'] for e in items]==[z['query_index'] for z in evidence]
    with threadpool_limits(limits=2):
        protocol=freeze(items)
        if args.stage=='run':run(protocol,items,evidence)


if __name__=='__main__':main()
