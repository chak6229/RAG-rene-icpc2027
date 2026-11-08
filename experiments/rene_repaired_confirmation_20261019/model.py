from __future__ import annotations

import copy

import numpy as np

from crsu.core import rrf
from experiments.strong_flow_uq_20261014 import flow as sf
from experiments.anchor_stage_risk_clean_20261016 import run as risk
from run_uq_pilot import booster, grouped_folds

from .common import ARMS, SEED, TOL, observable, reject_gold, assert_disjoint


def columns(arm, witness=False):
    if arm not in ARMS:
        raise ValueError('Unknown ablation arm')
    n = sf.FEATURE_DIM if witness else 84
    return np.array([i for i in range(n) if arm == 'seed_uq' or not 34 <= i < 46], int)


def x84(items, family, extras):
    arm = 'document' if family == 'rewrite' else 'joint'
    return [np.c_[e['x'], e['seed_x'], e['strong_evidence']['x'][family], sf.ab.mask_extra(a, arm)]
            for e, a in zip(items, extras)]


def train_gain(items, family, arm, seed):
    if not len(items) or not sum(len(e['x']) for e in items):
        raise ValueError('No gain training actions')
    evidence = [e['strong_evidence'] for e in items]
    extras, records = sf.base.crossfit_extra(items, evidence, family, seed)
    emodel = sf.base.fit_evidence(items, evidence)
    xx = x84(items, family, extras)
    xx = np.concatenate(xx)
    selected = columns(arm)
    q = np.concatenate([np.full(len(e['x']), e['query_index'], int) for e in items])
    repos = np.concatenate([np.full(len(e['x']), e['repo'], object) for e in items])
    y = np.concatenate([e['y'][family] for e in items])
    members = []
    for m in range(5):
        w = sf.member_weights(q, repos, m)
        members.append(dict(mean=booster(xx[:, selected], y, seed + m, weight=w),
                            harm=fit_binary(xx[:, selected], (y < -TOL).astype(int), seed + m, w)))
    return dict(arm=arm, family=family, columns=selected, evidence_model=emodel,
                members=members, fit_repos=sorted(set(repos)),
                fit_queries=[int(e['query_index']) for e in items], upstream_oof=records,
                input_dimension=len(selected), training_feature_names=feature_names(arm))


def fit_binary(x, y, seed, weights):
    # A genuinely one-class role has a smoothed constant, never a fictitious fitted model.
    prior = float((np.dot(weights, y) + .5) / (weights.sum() + 1.))
    return dict(constant=prior, booster=None if len(np.unique(y)) < 2 else
                booster(x, y, seed, 'binary', weights))


def binary_predict(model, x):
    return np.full(len(x), model['constant']) if model['booster'] is None else np.clip(model['booster'].predict(x), 0., 1.)


def feature_names(arm):
    import run_t5_seed_uq_pilot as seed_features
    import run_relevance_selection_pilot as relevance
    names = seed_features.BASE_FEATURES + seed_features.SEED_FEATURES + relevance.FEATURES + sf.base.EXTRA_NAMES
    return [names[i] for i in columns(arm)]


def predict_gain(model, items):
    for e in items:
        reject_gold(e)
    family = model['family']
    extras = sf.base.predict_extra(model['evidence_model'], items,
                                  [e['strong_evidence'] for e in items], family)
    arrays = x84(items, family, extras)
    all_x = np.concatenate(arrays) if arrays else np.empty((0, 84))
    n = len(all_x)
    mm = np.stack([m['mean'].predict(all_x[:, model['columns']]) for m in model['members']]) if n else np.empty((5, 0))
    hh = np.stack([binary_predict(m['harm'], all_x[:, model['columns']]) for m in model['members']]) if n else np.empty((5, 0))
    offsets = np.r_[0, np.cumsum([len(x) for x in arrays])]
    return [dict(x84=x, member_mu=mm[:, offsets[i]:offsets[i + 1]],
                 member_harm=hh[:, offsets[i]:offsets[i + 1]],
                 mu=mm[:, offsets[i]:offsets[i + 1]].mean(0),
                 harm=hh[:, offsets[i]:offsets[i + 1]].mean(0)) for i, x in enumerate(arrays)]


def gain_oof(items, family, arm, seed):
    repos = np.array([e['repo'] for e in items], object)
    output, records = [None] * len(items), []
    for f, held in enumerate(grouped_folds(repos, 4, SEED + 31)):
        tr = np.setdiff1d(np.arange(len(items)), held)
        model = train_gain([items[i] for i in tr], family, arm, seed + f)
        values = predict_gain(model, [observable(items[i]) for i in held])
        for i, value in zip(held, values):
            output[int(i)] = value
        records.append(dict(fit_repos=model['fit_repos'], held_repos=sorted(set(repos[held]))))
    if any(p is None for p in output):
        raise ValueError('Incomplete OOF predictions')
    return output, records


def prepare(item, pred, family, cut):
    reject_gold(item)
    short = sorted(np.flatnonzero(pred['mu'] > 0).tolist(), key=lambda j: (-pred['mu'][j], j))[:5]
    priority = float(pred['mu'][short[0]]) if short else 0.
    verified = bool(short and priority > 0 and priority >= cut)
    plan = None
    if verified:
        winner = int(short[0])
        rival = sf.choose_rival(pred, winner, short, 'point')
        plan = sf.plan_pair(item, pred, family, winner, rival, short)
    return dict(before=pred, priority=priority, verified=verified, plan=plan, family=family)


def train_update(items, scores, predictions, family, arm, seed):
    xx, yy, baseline, qq, repos = [], [], [], [], []
    for e, score, pred in zip(items, scores, predictions):
        # All positive-mean point-rival plans on a separate role train the same update.
        session = prepare(observable(e), pred, family, 0.)
        if not session['verified']:
            continue
        for j in session['plan']['candidates']:
            xx.append(sf.witness_x(observable(e), pred, family, score, session['plan'], j))
            yy.append(e['y'][family][j]); baseline.append(pred['member_mu'][:, j])
            qq.append(e['query_index']); repos.append(e['repo'])
    selected = columns(arm, witness=True)
    if not xx:
        return dict(available=False, columns=selected, members=[], fit_queries=[], fit_repos=[])
    xx, yy, baseline = np.array(xx), np.array(yy), np.array(baseline)
    members = []
    for m in range(5):
        w = sf.member_weights(np.array(qq), np.array(repos, object), m)
        members.append(dict(mean=booster(xx[:, selected], yy - baseline[:, m], seed + m, weight=w),
                            harm=fit_binary(xx[:, selected], (yy < -TOL).astype(int), seed + m, w)))
    return dict(available=True, columns=selected, members=members, input_dimension=len(selected),
                fit_queries=sorted(set(map(int, qq))), fit_repos=sorted(set(repos)),
                train_rows=len(xx), train_harm=int((yy < -TOL).sum()))


def choose_trace(item, session, update=None, score=None, tau=.1, use_ce=True):
    reject_gold(item)
    pred, family = session['before'], session['family']
    mm, hh = pred['member_mu'].copy(), pred['member_harm'].copy()
    allowed, updates = list(range(len(pred['mu']))), []
    verified = bool(use_ce and session['verified'])
    plan = session['plan'] if verified else None
    if verified:
        allowed = list(plan['candidates'])
        xs = np.stack([sf.witness_x(item, pred, family, score, plan, j) for j in allowed])
        if update['available']:
            x = xs[:, update['columns']]
            residual = np.stack([m['mean'].predict(x) for m in update['members']])
            direct = np.stack([binary_predict(m['harm'], x) for m in update['members']])
        else:
            residual = np.zeros((5, len(allowed)))
            direct = hh[:, allowed].copy()
        for k, j in enumerate(allowed):
            mm[:, j] += .5 * residual[:, k]
            hh[:, j] = direct[:, k]
            updates.append(dict(candidate=int(j), arm='full', x=xs[k],
                                member_mu=mm[:, j].copy(), member_harm=hh[:, j].copy()))
    mu, harm = mm.mean(0), hh.mean(0)
    eligible = [j for j in allowed if mu[j] > 0 and harm[j] <= tau]
    chosen = min(eligible, key=lambda j: (-mu[j], j)) if eligible else -1
    original = tuple(map(int, item['original_top100']))
    ranking = original if chosen < 0 else tuple(map(int,
                  item['candidate_top100'][chosen] if family == 'rewrite' else rrf(original, item['candidate_top100'][chosen])))
    identity = chosen >= 0 and ranking == original
    docs = [] if plan is None else list(map(int, plan['doc_ids']))
    # Token accounting uses actual separately truncated CE inputs, including four RoBERTa specials.
    tokens = sum(4 + item.get('ce_query_length', 0) + item.get('ce_code_lengths', {}).get(d, 0) for d in docs)
    return dict(query_index=int(item['query_index']), chosen=-1 if identity else int(chosen),
                accepted=bool(chosen >= 0 and not identity), ranking=ranking,
                identity_fallback=bool(identity), verified=verified, plan=plan,
                before=pred, updated=updates, final_mu=mu, final_harm=harm,
                mu=float(mu[chosen]) if chosen >= 0 else 0.,
                harm=float(harm[chosen]) if chosen >= 0 else 0., priority=session['priority'],
                billed_slots=8 if verified else 0, actual_pairs=len(docs), executed_ce_tokens=int(tokens),
                risk_certified=False)


def risk_x(item, trace):
    reject_gold(item)
    return np.r_[risk.s.observable_features(item, trace), risk.anchor_features(item, trace)]


def train_risk(items, traces, family):
    selected = [(e, t) for e, t in zip(items, traces) if t['accepted']]
    xx = np.stack([risk_x(observable(e), t) for e, t in selected]) if selected else np.empty((0, len(risk.FEATURES)))
    delta = np.array([e['y'][family][t['chosen']] for e, t in selected])
    data = dict(x=xx, y=(delta < -TOL).astype(float), delta=delta,
                native=np.array([t['harm'] for _, t in selected]),
                q=[e['query_index'] for e, _ in selected], repo=[e['repo'] for e, _ in selected])
    return risk.fit_model(data, 'threeway_anchor')


def apply_risk(model, items, traces):
    selected = [i for i, t in enumerate(traces) if t['accepted']]
    outputs = [dict(t, post_harm=None, post_risk_rejected=False) for t in traces]
    if not selected:
        return outputs
    x = np.stack([risk_x(items[i], traces[i]) for i in selected])
    p, classes, gain = risk.probabilities(model, dict(x=x, native=np.array([traces[i]['harm'] for i in selected])))
    for k, i in enumerate(selected):
        t = outputs[i]
        t.update(post_harm=float(p[k]), p_classes=classes[k], expected_gain=float(gain[k]), risk_x=x[k])
        if p[k] > .075:
            t.update(accepted=False, chosen=-1, ranking=tuple(map(int, items[i]['original_top100'])), post_risk_rejected=True)
    return outputs


def train_pipeline(items, scores, roles, family, arm, seed):
    byid = {e['query_index']: (e, s) for e, s in zip(items, scores)}
    repos = {role: [byid[q][0]['repo'] for q in ids] for role, ids in roles.items()}
    assert_disjoint(repos)
    gain_items = [byid[q][0] for q in roles['gain']]
    oof, oof_records = gain_oof(gain_items, family, arm, seed + 100)
    priorities = [float(max([0.] + p['mu'].tolist())) for p in oof]
    cut = float(np.quantile(priorities, .75, method='higher'))
    gain = train_gain(gain_items, family, arm, seed)
    ce_items, ce_scores = zip(*(byid[q] for q in roles['update']))
    ce_predictions = predict_gain(gain, [observable(e) for e in ce_items])
    update = train_update(ce_items, ce_scores, ce_predictions, family, arm, seed + 500)
    ri, rs = zip(*(byid[q] for q in roles['risk']))
    views = [observable(e) for e in ri]
    rp = predict_gain(gain, views)
    rt = [choose_trace(v, prepare(v, p, family, cut), update, s)
          for v, p, s in zip(views, rp, rs)]
    risk_model = train_risk(ri, rt, family)
    return dict(gain=gain, update=update, risk=risk_model, cut=cut, arm=arm, family=family,
                roles=roles, role_repositories=repos, priority_oof=oof_records,
                architecture='strong72/84 -> gain-priority point-rival CE -> direct harm .1 -> threeway .075',
                thresholds_tuned=False)


def predict_pipeline(model, items, scores):
    for item in items:
        reject_gold(item)
    predictions = predict_gain(model['gain'], items)
    sessions = [prepare(e, p, model['family'], model['cut']) for e, p in zip(items, predictions)]
    mean = [choose_trace(e, s, tau=1., use_ce=False) for e, s in zip(items, sessions)]
    base = [choose_trace(e, s, use_ce=False) for e, s in zip(items, sessions)]
    verified = [choose_trace(e, s, model['update'], ce) for e, s, ce in zip(items, sessions, scores)]
    gated = apply_risk(model['risk'], items, verified)
    return dict(mean=mean, base=base, verified=verified, risk=gated), predictions, sessions
