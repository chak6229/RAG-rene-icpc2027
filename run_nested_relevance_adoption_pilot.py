"""Exploratory, nested repository CV on the frozen 480-query valid pool.

Evidence supervision, candidate selection and query adoption have separate roles.
No gold identity, true original RR, repository identity or role enters inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

import joblib
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from crsu.core import RankingStates, rrf
import run_relevance_selection_pilot as rel
import run_t5_seed_uq_pilot as old
from run_uq_pilot import booster, clustered_ci, dump, finite_quantile, grouped_folds

OUT = Path('results/uq_pilot_20261012/nested_relevance_adoption')
SEED = 12120
ARMS = ('baseline_direct', 'learned_direct', 'baseline_adoption', 'learned_adoption')
SOURCE = old.OUT/'evidence.pkl'
EVIDENCE = rel.OUT/'relevance_evidence.pkl'
BASE_NAMES = old.BASE_FEATURES + old.SEED_FEATURES + rel.FEATURES
EXTRA_NAMES = [f'learned_{s}' for s in rel.STATS] + [
    'predicted_original_top1_correct', 'original_top1_demotion', 'expected_top1_destruction']
DOC_NAMES = [f'{s}_{t}' for t in ('value', 'query_z', 'delta_union_max') for s in rel.SIGNALS] + [
    'original_reciprocal_rank', 'original_top1_indicator', 'original_top10_indicator']
ANCHOR_BASE = ['original_top1', 'original_margin', 'original_top10_std',
               'original_top10_entropy', 'original_top100_mean', 'original_top100_std']
META_OBS = ['query_candidate_cosine', 'top100_overlap', 'top10_rank_stability',
            'top1_document_stable', 'cross_seed_top10_agreement_mean',
            'exact_seed_support', 'candidate_top10_document_support',
            'bm25_discounted_delta', 'idf_overlap_discounted_delta',
            'token_maxsim_discounted_delta']
META_NAMES = ['selector_mean', 'selector_scale', 'selector_harm',
              'selector_mean_margin', 'selector_mean_std', 'selector_mean_range',
              'log_candidate_count'] + META_OBS
DEPENDENCIES = [Path(p) for p in ('run_relevance_selection_pilot.py',
    'run_t5_seed_uq_pilot.py', 'run_uq_pilot.py', 'crsu/core.py',
    'tests/test_nested_relevance_adoption_pilot.py')]


def splits(items):
    repos = np.array([e['repo'] for e in items], object)
    result = []
    for f, ev in enumerate(grouped_folds(repos, 5, SEED)):
        tr = np.setdiff1d(np.arange(len(items)), ev)
        bins = [tr[b] for b in grouped_folds(repos[tr], 10, SEED+1+f)]
        roles = dict(selector_fit=np.concatenate(bins[:4]),
                     adoption_fit=np.concatenate(bins[4:6]),
                     lcb_calibration=bins[6], selection_valid=bins[7],
                     risk_calibration=np.concatenate(bins[8:]), new_holdout=ev)
        assert_disjoint(items, roles)
        result.append(roles)
    assert sorted(np.concatenate([s['new_holdout'] for s in result]).tolist()) == list(range(len(items)))
    return result


def assert_disjoint(items, roles):
    repos = {k: {items[int(i)]['repo'] for i in v} for k, v in roles.items()}
    for a in repos:
        for b in repos:
            if a != b and repos[a] & repos[b]:
                raise ValueError(f'repository overlap: {a}, {b}')
    flat = np.concatenate(list(roles.values()))
    if len(np.unique(flat)) != len(flat):
        raise ValueError('query overlap')


def freeze(items):
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {str(p): old.sha256(p) for p in (SOURCE, EVIDENCE, old.OUT/'PROTOCOL.json',
                                            rel.OUT/'PROTOCOL.json')}
    dependencies = {str(p): old.sha256(p) for p in DEPENDENCIES}
    role_plan = [{k: [items[int(i)]['query_index'] for i in v] for k, v in s.items()} for s in splits(items)]
    fixed = dict(inputs=inputs, dependencies=dependencies,
                 source_sha256=old.sha256(Path(__file__)), stable_sha256=old.sha256(old.STABLE),
                 fold_query_ids=role_plan)
    if (OUT/'PROTOCOL.json').exists():
        p = json.loads((OUT/'PROTOCOL.json').read_text(encoding='utf-8'))
        assert all(p[k] == v for k, v in fixed.items())
        return p
    p = dict(fixed, frozen_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
        test_used=False, exploratory=True, history='previously viewed valid-derived pool; no independent confirmation',
        roles='5 outer repo folds; remaining queries in 10 repo bins: selector4/adoption2/lcb1/select1/risk2',
        candidate_pool='unchanged multi8, same real Top100/RRF and code-only evidence',
        arms=list(ARMS), base_features=BASE_NAMES, extra_features=EXTRA_NAMES,
        doc_features=DOC_NAMES, meta_features=META_NAMES,
        document_model='query-weighted standardized logistic C=1; observable union only; no injected gold',
        anchor='standardized logistic C=.2; observed original Top1 evidence, retriever scores; RR=1 training label only',
        learned_features='7 learned relevance shift statistics + success probability + demotion + their product',
        selector='ordinary query-weighted LightGBM 7 leaves/min30/120 rounds, same architecture as frozen relevance_point',
        selector_crossfit='4 repo folds; learned selector inputs crossfitted; selector residual OOF refits all upstream learners',
        adoption='fit only selected actions on separate adoption_fit; standardized Ridge alpha20 and harm logistic C=.2',
        scale='OOF log(abs residual+.001); adoption Ridge alpha20, candidate LightGBM; floor .001',
        calibration='selected action frozen by selector; independent normalized and constant one-sided alpha .1 CP',
        primary='learned_adoption vs baseline_direct, within-fold mean-ranked rounded10% adoption',
        ablations=['learned_direct vs baseline_direct', 'baseline_adoption vs baseline_direct',
                   'learned_adoption vs learned_direct', 'normalized vs constant LCB with exact same chosen actions/mean'],
        risk='selection chooses .05/.1/.2/.4/1 harm threshold with mean or LCB>0, risk checked once; CP95 upper<=.1',
        risk_limit='query-binomial descriptive only; repo dependence/multiplicity not certified',
        no_search=True, no_case_rules=True)
    dump(OUT/'PROTOCOL.json', p)
    for source in [Path(__file__), *DEPENDENCIES]:
        dest = OUT/'frozen_source'/(Path(source.name) if source == Path(__file__) else source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    return p


def document_features(item, evidence):
    """ID-independent observable values; item labels are never consulted."""
    signals = np.asarray(evidence['doc_signals'], float)
    ids = evidence['doc_ids']
    rank = {int(d): i+1 for i, d in enumerate(item['original_top100'])}
    rr = np.array([1/rank[int(d)] if int(d) in rank else 0. for d in ids])
    std = np.maximum(signals.std(0), .001)
    return np.c_[signals, (signals-signals.mean(0))/std,
                 signals-signals.max(0), rr, rr == 1., rr >= .1]


def anchor_features(item, evidence):
    original = item['original_top100'][:10]
    lookup = {int(d): i for i, d in enumerate(evidence['doc_ids'])}
    sig = evidence['doc_signals'][[lookup[int(d)] for d in original]]
    observed = np.array([item['x'][0, old.BASE_FEATURES.index(n)] if len(item['x']) else 0.
                         for n in ANCHOR_BASE])
    return np.r_[sig[0], sig[0]-sig[1:].mean(0), sig[0]-sig[1:].max(0),
                 observed, float(len(item['x']) > 0)]


def fit_binary(x, y, c=1., weight=None):
    if weight is None:
        weight = np.ones(len(y))
    prior = float((np.dot(weight, y)+.5)/(weight.sum()+1))
    if len(np.unique(y)) < 2:
        return dict(prior=prior, model=None)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    model.fit(x, y, standardscaler__sample_weight=weight, logisticregression__sample_weight=weight)
    return dict(prior=prior, model=model)


def binary_predict(model, x):
    return np.full(len(x), model['prior']) if model['model'] is None else model['model'].predict_proba(x)[:, 1]


def fit_evidence(items, evidence):
    x = np.concatenate([document_features(e, z) for e, z in zip(items, evidence)])
    # Gold equality is isolated to training targets, including absent-gold queries.
    y = np.concatenate([z['doc_ids'] == e['query_index'] for e, z in zip(items, evidence)]).astype(int)
    w = np.concatenate([np.full(len(z['doc_ids']), 1/len(z['doc_ids'])) for z in evidence])
    anchor_x = np.array([anchor_features(e, z) for e, z in zip(items, evidence)])
    anchor_y = np.array([int(e['original_top100'][0]) == e['query_index'] for e in items], int)
    return dict(document=fit_binary(x, y, 1., w), anchor=fit_binary(anchor_x, anchor_y, .2),
                fit_query_ids=[e['query_index'] for e in items], fit_repos=sorted({e['repo'] for e in items}))


def action_features(original, action, ids, doc_probability, p0):
    lookup = {int(d): i for i, d in enumerate(ids)}
    o = np.array([lookup[int(d)] for d in original[:10]])
    a = np.array([lookup[int(d)] for d in action[:10]])
    discount = 1/np.arange(1, 11, dtype=float)
    shift = np.zeros(len(ids)); shift[a] += discount; shift[o] -= discount
    pos = shift > old.TOL; neg = shift < -old.TOL
    original_score = float(discount @ doc_probability[o]/discount.sum())
    action_score = float(discount @ doc_probability[a]/discount.sum())
    promoted = float(shift[pos] @ doc_probability[pos]/shift[pos].sum()) if pos.any() else 0.
    demoted = float(-shift[neg] @ doc_probability[neg]/(-shift[neg]).sum()) if neg.any() else 0.
    where = np.flatnonzero(np.asarray(action) == original[0])
    reciprocal = 1/(int(where[0])+1) if len(where) else 0.
    demotion = 1-reciprocal
    return np.array([doc_probability[a[0]], action_score, original_score, action_score-original_score,
                     promoted, demoted, shift @ doc_probability, p0, demotion, p0*demotion])


def predict_extra(model, items, evidence, family):
    out = []
    for e, z in zip(items, evidence):
        p = binary_predict(model['document'], document_features(e, z))
        p0 = float(binary_predict(model['anchor'], anchor_features(e, z)[None])[0])
        out.append(np.array([action_features(e['original_top100'],
                    top if family == 'rewrite' else rrf(e['original_top100'], top), z['doc_ids'], p, p0)
                    for top in e['candidate_top100']]).reshape(-1, len(EXTRA_NAMES)))
    return out


def subset(items, ix):
    return [items[int(i)] for i in ix]


def crossfit_extra(items, evidence, family, seed):
    output = [None]*len(items); records = []
    reps = np.array([e['repo'] for e in items], object)
    for val in grouped_folds(reps, 4, seed):
        tr = np.setdiff1d(np.arange(len(items)), val)
        assert not set(reps[tr]) & set(reps[val])
        m = fit_evidence(subset(items, tr), subset(evidence, tr))
        preds = predict_extra(m, subset(items, val), subset(evidence, val), family)
        for i, prediction in zip(val, preds): output[int(i)] = prediction
        records.append(dict(fit_repos=sorted(set(reps[tr])), held_repos=sorted(set(reps[val]))))
    assert all(x is not None and np.isfinite(x).all() for x in output)
    return output, records


def data_for(items, evidence, family, extras=None):
    data = old.flatten(items, family)
    data['full'] = np.c_[data['x'], data['seed_x'], np.concatenate([z['x'][family] for z in evidence])]
    assert data['full'].shape[1] == len(BASE_NAMES)
    if extras is not None:
        data['augmented'] = np.c_[data['full'], np.concatenate(extras)]
    return data


def fit_selector_mean(items, evidence, family, learned, seed):
    extra_model = None; records = []
    if learned:
        extras, records = crossfit_extra(items, evidence, family, seed)
        extra_model = fit_evidence(items, evidence)
    else:
        extras = None
    d = data_for(items, evidence, family, extras)
    x = d['augmented' if learned else 'full']
    mean = booster(x, d['y'], seed, weight=d['weights'])
    harm = booster(x, (d['y'] < -old.TOL).astype(int), seed, 'binary', d['weights'])
    return dict(mean=mean, harm=harm, evidence_model=extra_model, learned=learned,
                upstream_oof=records, fit_query_ids=d['query_ids'], fit_repos=sorted(set(d['query_repos']))), d


def selector_data(model, items, evidence, family):
    extras = predict_extra(model['evidence_model'], items, evidence, family) if model['learned'] else None
    d = data_for(items, evidence, family, extras)
    return d, d['augmented' if model['learned'] else 'full']


def fit_selector(items, evidence, family, learned, seed):
    mean_oof = np.full(sum(len(e['x']) for e in items), np.nan)
    offsets = np.r_[0, np.cumsum([len(e['x']) for e in items])]
    records = []; reps = np.array([e['repo'] for e in items], object)
    for val in grouped_folds(reps, 4, seed):
        tr = np.setdiff1d(np.arange(len(items)), val)
        m, _ = fit_selector_mean(subset(items, tr), subset(evidence, tr), family, learned, seed)
        _, x = selector_data(m, subset(items, val), subset(evidence, val), family)
        ii = np.concatenate([np.arange(offsets[int(i)], offsets[int(i)+1]) for i in val])
        mean_oof[ii] = m['mean'].predict(x)
        records.append(dict(fit_repos=sorted(set(reps[tr])), held_repos=sorted(set(reps[val])),
                            upstream=m['upstream_oof']))
    assert np.isfinite(mean_oof).all()
    m, d = fit_selector_mean(items, evidence, family, learned, seed)
    x = d['augmented' if learned else 'full']
    m.update(scale=booster(x, np.log(abs(d['y']-mean_oof)+.001), seed, weight=d['weights']),
             residual_oof_roles=records)
    return m


def predict_selector(model, items, evidence, family):
    d, x = selector_data(model, items, evidence, family)
    pred = dict(mu=model['mean'].predict(x),
                s=np.maximum(np.exp(model['scale'].predict(x)), .001),
                harm=np.clip(model['harm'].predict(x), 0, 1))
    return d, pred


def selected_rows(data, pred, learned):
    chosen = old.first(data, pred, 'multi8')
    z = []; names = BASE_NAMES + EXTRA_NAMES if learned else BASE_NAMES
    observed = data['augmented' if learned else 'full']
    for i in chosen:
        ix = np.flatnonzero(data['q'] == data['q'][i]); mu = pred['mu'][ix]
        sorted_mu = np.sort(mu)
        margin = sorted_mu[-1]-sorted_mu[-2] if len(mu) > 1 else 0.
        selected = observed[i]
        row = [pred['mu'][i], pred['s'][i], pred['harm'][i], margin,
               mu.std(), np.ptp(mu), np.log1p(len(mu))]
        row.extend(selected[names.index(name)] for name in META_OBS)
        if learned: row.extend(selected[len(BASE_NAMES):])
        z.append(row)
    n = len(META_NAMES)+(len(EXTRA_NAMES) if learned else 0)
    return dict(x=np.array(z).reshape(-1, n), chosen=chosen,
                q=data['q'][chosen], repo=data['repo'][chosen], y=data['y'][chosen],
                mu=pred['mu'][chosen], s=pred['s'][chosen], harm=pred['harm'][chosen],
                query_ids=data['query_ids'], query_repos=data['query_repos'], query_count=data['query_count'])


def fit_ridge(x, y):
    return make_pipeline(StandardScaler(), Ridge(alpha=20.)).fit(x, y)


def fit_adoption(rows, seed):
    x = rows['x']; y = rows['y']; reps = rows['repo']; oof = np.full(len(y), np.nan)
    records = []
    for held in grouped_folds(reps, 4, seed):
        tr = np.setdiff1d(np.arange(len(y)), held)
        assert not set(reps[tr]) & set(reps[held])
        model = fit_ridge(x[tr], y[tr]); oof[held] = np.clip(model.predict(x[held]), -1, 1)
        records.append(dict(fit_repos=sorted(set(reps[tr])), held_repos=sorted(set(reps[held]))))
    assert np.isfinite(oof).all()
    return dict(mean=fit_ridge(x, y), scale=fit_ridge(x, np.log(abs(y-oof)+.001)),
                harm=fit_binary(x, (y < -old.TOL).astype(int), .2),
                fit_query_ids=rows['q'].tolist(), fit_repos=sorted(set(reps)), residual_oof_roles=records)


def predict_adoption(model, rows):
    return dict(rows, mu=np.clip(model['mean'].predict(rows['x']), -1, 1),
                s=np.maximum(np.exp(np.clip(model['scale'].predict(rows['x']), -12, 2)), .001),
                harm=binary_predict(model['harm'], rows['x']))


def decision(rows, chosen_positions):
    j = np.asarray(chosen_positions, int); y = rows['y'][j]
    values = {int(rows['q'][k]): float(rows['y'][k]) for k in j}
    return dict(accepted=len(j), coverage=len(j)/rows['query_count'],
        delta_mrr=float(y.sum()/rows['query_count']),
        repo_ci=clustered_ci(rows, values, SEED, 2000),
        positive=int((y > old.TOL).sum()), harmful=int((y < -old.TOL).sum()),
        severe=int((y <= -.1).sum()), zero=int((abs(y) <= old.TOL).sum()),
        harmful_rate=float(np.mean(y < -old.TOL)) if len(y) else None,
        gross_gain=float(np.maximum(y, 0).sum()/rows['query_count']),
        loss=float(np.maximum(-y, 0).sum()/rows['query_count']), query_values=values,
        accepted_query_ids=[int(v) for v in rows['q'][j]])


def calibration(rows, cal):
    correction = max(finite_quantile((cal['mu']-cal['y'])/cal['s']), 0.)
    constant = max(finite_quantile(cal['mu']-cal['y']), 0.)
    assert np.isfinite(correction) and np.isfinite(constant)
    return dict(rows, lower=rows['mu']-correction*rows['s'], constant_lower=rows['mu']-constant,
                correction=correction, constant_correction=constant)


def curves(rows):
    return {s: {str(c): decision(rows, np.argsort(-rows[k], kind='stable')[:round(c*rows['query_count'])])
                    for c in (.05, .1, .2)}
            for s, k in (('mean', 'mu'), ('lcb', 'lower'), ('constant_lcb', 'constant_lower'),
                         ('negative_harm', 'harm_negative'))}


def risk_policy(roles, gate):
    sel = roles['selection_valid']; score_key = 'mu' if gate == 'mean' else 'lower'
    choices = []
    for t in (.05, .1, .2, .4, 1.):
        ix = np.flatnonzero((sel[score_key] > 0) & (sel['harm'] <= t)); y = sel['y'][ix]
        if len(ix) and np.mean(y < -old.TOL) <= .1 and y.sum() > 0:
            choices.append((float(y.sum()), len(ix), t))
    threshold = max(choices)[2] if choices else -1.
    risk = roles['risk_calibration']
    ix = np.flatnonzero((risk[score_key] > 0) & (risk['harm'] <= threshold))
    n = len(ix); h = int((risk['y'][ix] < -old.TOL).sum())
    upper = float(stats.beta.ppf(.95, h+1, n-h)) if n > h else 1.
    passed = n > 0 and upper <= .1
    ev = roles['new_holdout']; accept = np.flatnonzero((ev[score_key] > 0) & (ev['harm'] <= threshold))
    return dict(threshold=threshold, risk_accepted=n, risk_harmful=h, query_iid_cp95_upper=upper,
        passes_descriptive_check=passed, before_risk=decision(ev, accept),
        after_risk=decision(ev, accept if passed else []),
        caveat='query-binomial descriptive diagnostic, not repository/multiplicity certification')


def prediction_metrics(rows):
    y = rows['y']; err = abs(rows['mu']-y); severe = y <= -.1
    residual = y-rows['lower']
    def rho(mask):
        if mask.sum() < 3 or np.ptp(rows['s'][mask]) == 0 or np.ptp(err[mask]) == 0: return None
        z = stats.spearmanr(rows['s'][mask], err[mask]).statistic
        return float(z) if np.isfinite(z) else None
    return dict(selected_mae=float(err.mean()),
        harm_brier=float(np.mean((rows['harm']-(y < -old.TOL))**2)),
        severe_mean_nonpositive_recall=float(np.mean(rows['mu'][severe] <= 0)) if severe.any() else None,
        severe_selected_count=int(severe.sum()), scale_error_rho=rho(np.ones(len(y), bool)),
        scale_error_rho_nonzero=rho(abs(y) > old.TOL),
        lcb_violation=float(np.mean(y < rows['lower'])),
        q10_pinball=float(np.mean(np.maximum(.1*residual, -.9*residual))),
        positive_lcb=decision(rows, np.flatnonzero(rows['lower'] > 0)))


def selector_metrics(d, pred):
    chosen = old.first(d, pred, 'multi8'); best = []; positive = []; pair = []
    for q in d['query_ids']:
        ix = np.flatnonzero(d['q'] == q)
        if not len(ix): best.append(0.); continue
        y = d['y'][ix]; mu = pred['mu'][ix]
        best.append(max(float(y.max()), 0.)); positive.append(max(float(y[np.argmax(mu)]), 0.))
        keep = (y[:, None]-y[None, :]) > old.TOL
        if keep.any():
            dif = mu[:, None]-mu[None, :]
            pair.append(float(np.mean((dif[keep] > old.TOL)+.5*(abs(dif[keep]) <= old.TOL))))
    return dict(oracle_sum=float(sum(best)), selected_positive_sum=float(sum(positive)),
        positive_opportunity_retention=float(sum(positive)/sum(best)) if sum(best) else None,
        all_candidate_oracle_delta_mrr=float(sum(best)/d['query_count']),
        first_choice_oracle_ceiling=old.decision(d, chosen[d['y'][chosen] > old.TOL]),
        within_query_pair_accuracy=float(np.mean(pair)) if pair else None)


def anchor_metrics(model, train_items, eval_items, evidence):
    y = np.array([e['original_top100'][0] == e['query_index'] for e in eval_items], int)
    p = binary_predict(model['anchor'], np.array([anchor_features(e, z) for e, z in zip(eval_items, evidence)]))
    prior = np.mean([e['original_top100'][0] == e['query_index'] for e in train_items])
    return dict(queries=len(y), positives=int(y.sum()), brier=float(np.mean((y-p)**2)),
        prior_brier=float(np.mean((y-prior)**2)), auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        probabilities=p.tolist(), query_ids=[e['query_index'] for e in eval_items])


def combine_rows(rows):
    keys = ('q', 'repo', 'y', 'mu', 's', 'harm', 'lower', 'constant_lower')
    out = {k: np.concatenate([r[k] for r in rows]) for k in keys}
    out.update(query_ids=sum([r['query_ids'] for r in rows], []),
               query_repos=sum([r['query_repos'] for r in rows], []),
               query_count=sum(r['query_count'] for r in rows))
    return out


def merge_decisions(rows, decisions):
    merged = combine_rows(rows)
    ids = {q for d in decisions for q in d['accepted_query_ids']}
    return decision(merged, np.flatnonzero(np.isin(merged['q'], list(ids))))


def audit_labels(items):
    checks = 0
    for e in items:
        rr = np.flatnonzero(np.asarray(e['original_top100']) == e['query_index'])
        expected = 1/(int(rr[0])+1) if len(rr) else 0.
        assert abs(expected-e['original_rr']) <= old.TOL
        for f in ('rewrite', 'fusion'):
            lists = {str(j): top if f == 'rewrite' else rrf(e['original_top100'], top)
                     for j, top in enumerate(e['candidate_top100'])}
            if lists:
                states = RankingStates.from_rankings(e['original_top100'], lists)
                np.testing.assert_allclose(states.gains[states.target_index(e['query_index'])],
                                           e['y'][f], atol=old.TOL, rtol=0)
            checks += len(lists)
    return checks


def run(protocol, items, evidence):
    if (OUT/'summary.json').exists(): print('Complete; protocol verified, no retraining', flush=True); return
    start = time.perf_counter(); checks = audit_labels(items); saved = {}; result = {}
    for family in ('rewrite', 'fusion'):
        fold_outputs = []; fold_rows = []; anchors = []
        for f, plan in enumerate(splits(items)):
            print(f'{family} outer {f+1}/5: roles '+str({r: len(ix) for r, ix in plan.items()}), flush=True)
            train = subset(items, plan['selector_fit']); train_z = subset(evidence, plan['selector_fit'])
            models = {}; rowsets = {}; sels = {}; selection_metrics = {}; audit = {}
            for learned in (False, True):
                label = 'learned' if learned else 'baseline'
                selector = fit_selector(train, train_z, family, learned, SEED+f)
                predicted = {}; ev_d = None; ev_p = None
                for role, ix in plan.items():
                    if role == 'selector_fit': continue
                    d, p = predict_selector(selector, subset(items, ix), subset(evidence, ix), family)
                    predicted[role] = selected_rows(d, p, learned)
                    if role == 'new_holdout': ev_d, ev_p = d, p
                adoption = fit_adoption(predicted['adoption_fit'], SEED+f)
                models[label] = dict(selector=selector, adoption=adoption)
                for kind in ('direct', 'adoption'):
                    name = f'{label}_{kind}'
                    raw = {r: v if kind == 'direct' else predict_adoption(adoption, v)
                           for r, v in predicted.items() if r != 'adoption_fit'}
                    cal = raw['lcb_calibration']
                    ps = {r: calibration(v, cal) for r, v in raw.items()}
                    for v in ps.values(): v['harm_negative'] = -v['harm']
                    rowsets[name] = ps
                selection_metrics[label] = selector_metrics(ev_d, ev_p)
                sels[label] = dict(query_ids=ev_d['query_ids'], candidate_q=ev_d['q'], candidate_y=ev_d['y'],
                                   candidate_pred=ev_p, eval_features=ev_d['augmented' if learned else 'full'])
                audit[label] = dict(selector_fit=selector['fit_query_ids'],
                    evidence_oof=selector['upstream_oof'], residual_oof=selector['residual_oof_roles'],
                    adoption_fit=adoption['fit_query_ids'], adoption_oof=adoption['residual_oof_roles'])
                if learned:
                    anchors.append(anchor_metrics(selector['evidence_model'], train,
                        subset(items, plan['new_holdout']), subset(evidence, plan['new_holdout'])))
            path = OUT/f'{family}_fold{f}_models.joblib'; joblib.dump(models, path)
            restored = joblib.load(path)
            for label, bundle in restored.items():
                for r in ('lcb_calibration', 'selection_valid', 'risk_calibration', 'new_holdout'):
                    ix = plan[r]; d, p = predict_selector(bundle['selector'], subset(items, ix), subset(evidence, ix), family)
                    replay = selected_rows(d, p, label == 'learned')
                    np.testing.assert_array_equal(replay['q'], rowsets[label+'_direct'][r]['q'])
                    for k in ('mu', 's', 'harm'):
                        np.testing.assert_array_equal(replay[k], rowsets[label+'_direct'][r][k])
                    replay = predict_adoption(bundle['adoption'], replay)
                    for k in ('mu', 's', 'harm'):
                        np.testing.assert_array_equal(replay[k], rowsets[label+'_adoption'][r][k])
            outputs = {}
            for name, ps in rowsets.items():
                ev = ps['new_holdout']
                outputs[name] = dict(prediction=prediction_metrics(ev), curves=curves(ev),
                    risk_mean=risk_policy(ps, 'mean'), risk_lcb=risk_policy(ps, 'lcb'),
                    correction=ev['correction'], constant_correction=ev['constant_correction'])
            fold_outputs.append(dict(fold=f, methods=outputs, selection=selection_metrics,
                                     role_counts={r: len(ix) for r, ix in plan.items()}, audit=audit))
            fold_rows.append({name: ps['new_holdout'] for name, ps in rowsets.items()})
            saved[f'{family}_fold{f}'] = dict(roles=plan, predictions=rowsets, selection=sels)
            print(f'{family} fold{f} mean@10% '+str({n: round(o['curves']['mean']['0.1']['delta_mrr'], 6)
                                                      for n, o in outputs.items()}), flush=True)
        aggregate = {}
        for name in ARMS:
            rows = [r[name] for r in fold_rows]; merged = combine_rows(rows)
            cv = {s: {str(c): merge_decisions(rows, [o['methods'][name]['curves'][s][str(c)] for o in fold_outputs])
                      for c in (.05, .1, .2)} for s in ('mean', 'lcb', 'constant_lcb', 'negative_harm')}
            risk = {}
            for gate in ('mean', 'lcb'):
                diags = [o['methods'][name]['risk_'+gate] for o in fold_outputs]
                risk[gate] = dict(passing_folds=sum(d['passes_descriptive_check'] for d in diags),
                    after_risk=merge_decisions(rows, [d['after_risk'] for d in diags]),
                    before_risk=merge_decisions(rows, [d['before_risk'] for d in diags]), folds=diags)
            aggregate[name] = dict(prediction=prediction_metrics(merged), curves=cv, risk=risk,
                                  mean10_fold_delta=[o['methods'][name]['curves']['mean']['0.1']['delta_mrr'] for o in fold_outputs])
        comparisons = {}
        pairs = [('primary_joint', 'learned_adoption', 'baseline_direct', 'mean', 'mean'),
                 ('evidence_only', 'learned_direct', 'baseline_direct', 'mean', 'mean'),
                 ('adoption_only', 'baseline_adoption', 'baseline_direct', 'mean', 'mean'),
                 ('adoption_with_evidence', 'learned_adoption', 'learned_direct', 'mean', 'mean')]
        pairs += [(name+'_uq', name, name, 'lcb', 'constant_lcb') for name in ARMS]
        merged = combine_rows([r['baseline_direct'] for r in fold_rows])
        for key, a, b, sa, sb in pairs:
            va = aggregate[a]['curves'][sa]['0.1']['query_values']; vb = aggregate[b]['curves'][sb]['0.1']['query_values']
            delta = {q: va.get(q, 0)-vb.get(q, 0) for q in merged['query_ids']}
            comparisons[key] = dict(delta_difference=sum(delta.values())/len(items),
                repo_ci=clustered_ci(merged, delta, SEED, 3000),
                caveat='exploratory cluster bootstrap, fixed fit/acceptance, no refit or multiplicity adjustment')
        selection = {}
        for label in ('baseline', 'learned'):
            metrics = [o['selection'][label] for o in fold_outputs]
            denominator = sum(m['oracle_sum'] for m in metrics)
            numerator = sum(m['selected_positive_sum'] for m in metrics)
            selection[label] = dict(positive_opportunity_retention=numerator/denominator,
                selected_positive_ceiling=numerator/len(items), all_candidate_oracle=denominator/len(items),
                pair_accuracy=float(np.mean([m['within_query_pair_accuracy'] for m in metrics])))
        result[family] = dict(methods=aggregate, comparisons=comparisons, selection=selection,
                              folds=fold_outputs, anchors=anchors)
    with (OUT/'predictions.pkl').open('wb') as stream: pickle.dump(saved, stream)
    assert old.sha256(old.STABLE) == protocol['stable_sha256']
    dump(OUT/'summary.json', dict(test_used=False, exploratory=True, development_only=True,
         queries=len(items), repositories=len({e['repo'] for e in items}), families=result,
         elapsed_seconds=time.perf_counter()-start, protocol_sha256=old.sha256(OUT/'PROTOCOL.json')))
    dump(OUT/'POSTRUN_AUDIT.json', dict(label_checks=checks, label_mismatches=0, test_used=False,
        stable_unchanged=True, repository_roles_disjoint=True, outer_query_coverage_once=True,
        model_roundtrip_all_external_roles=True, upstream_and_residual_crossfit=True,
        separate_adoption_fit=True, no_hyperparameter_search=True,
        protocol_sha256=old.sha256(OUT/'PROTOCOL.json'), source_sha256=old.sha256(Path(__file__)),
        predictions_sha256=old.sha256(OUT/'predictions.pkl'), summary_sha256=old.sha256(OUT/'summary.json')))
    print(f'Completed in {time.perf_counter()-start:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=('prepare', 'run'), default='run')
    args = parser.parse_args()
    items = pickle.load(SOURCE.open('rb')); evidence = pickle.load(EVIDENCE.open('rb'))
    assert [e['query_index'] for e in items] == [e['query_index'] for e in evidence]
    with threadpool_limits(limits=2):
        protocol = freeze(items)
        if args.stage == 'run': run(protocol, items, evidence)


if __name__ == '__main__': main()
