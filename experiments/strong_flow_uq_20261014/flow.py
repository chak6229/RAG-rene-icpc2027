"""Strong84 baseline plus OOF action-witness residuals.

Five repository bootstrap heads share supervised-evidence inputs and member
indices across candidates. Member variation is an empirical head proxy; the
shared upstream evidence scorer is not a Bayesian parameter posterior.
Risk/held outcomes never enter this module's policy fitting.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/action_witness_uq_20261014'))
import compare as c
import run_document_anchor_ablation as ab
import run_nested_relevance_adoption_pilot as base
from run_uq_pilot import booster, grouped_folds
from risk_bounded_ranking import worst_case_loss

OUT = ROOT / 'results/strong_flow_uq_20261014'


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

SEED = 261014
MEMBERS = 5
INNER_FOLDS = 3
GRID = (.025, .05, .1, .15, .2, .3, .5, 1.)
RHO = .5
HARM_BLEND = .5
TOL = 1e-10
SHORTLIST = 5
POLICIES = ('strong_ungated', 'strong_same_tau', 'ce_zero_point',
            'ce_full_point', 'ce_full_uq', 'weak_ungated', 'weak_same_tau')
PRIMARY = 'ce_full_uq'


def subset(items, ix):
    return [items[int(i)] for i in ix]


def evidence_of(items):
    return [e['strong_evidence'] for e in items]


def attach_evidence(items, evidence):
    assert len(items) == len(evidence)
    for e, z in zip(items, evidence):
        assert int(e['query_index']) == int(z['query_index'])
        assert set(z['x']) == {'rewrite', 'fusion'}
        for family in ('rewrite', 'fusion'):
            assert np.asarray(z['x'][family]).shape == (len(e['x']), 28)
            assert np.isfinite(z['x'][family]).all()
        assert np.asarray(z['doc_signals']).shape == (len(z['doc_ids']), 4)
        e['strong_evidence'] = z
    return items


def _xdata(items, family, extra):
    arm = 'document' if family == 'rewrite' else 'joint'
    xx = [np.c_[e['x'], e['seed_x'], z['x'][family], ab.mask_extra(a, arm)]
          for e, z, a in zip(items, evidence_of(items), extra)]
    assert all(x.shape == (len(e['x']), 84) and np.isfinite(x).all() for e, x in zip(items, xx))
    return xx


def query_weights(q, repo, seed=None):
    return c.cg.old.query_weights(np.asarray(q), np.asarray(repo, object), seed)


def member_weights(q, repo, member):
    """One immutable repository multiplier per member across ALL nested fits.

    Keying the multiplier to repository identity also preserves it when a
    subset omits repositories or the CE table omits a no-eligible query.
    Repository IDs define resampling weights only, never prediction inputs.
    """
    q = np.asarray(q); repo = np.asarray(repo, object)
    w = query_weights(q, repo)
    mult = {}
    for r in set(repo):
        key = f'{SEED}:bootstrap:{int(member)}:{r}'
        keyed_seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        mult[r] = float(np.random.default_rng(keyed_seed).poisson(1))
    w *= np.array([mult[r] for r in repo])
    if not w.sum():
        raise ValueError('All-zero frozen bootstrap weights; do not change bootstrap after outcomes')
    # Match the old LightGBM strong head's per-query total weight=1 scale.
    # lambda_l2 and min_sum_hessian are sensitive to this absolute scale.
    return w / w.sum() * len(np.unique(q))


def fit_strong(items, family, seed):
    """Old strong84 architecture; shared evidence OOF, five bootstrap heads."""
    z = evidence_of(items)
    extra, upstream = base.crossfit_extra(items, z, family, seed)
    evidence_model = base.fit_evidence(items, z)
    xx = _xdata(items, family, extra)
    d = base.old.flatten(items, family)
    x = np.concatenate(xx)
    members = []
    for m in range(MEMBERS):
        weights = member_weights(d['q'], d['repo'], m)
        members.append(dict(mean=booster(x, d['y'], seed + m, weight=weights),
                            harm=booster(x, (d['y'] < -TOL).astype(int), seed + m,
                                         'binary', weights), weights=weights,
                            bootstrap_member=m, bootstrap_key_seed=SEED))
    return dict(evidence_model=evidence_model, members=members,
                fit_query_ids=d['query_ids'], fit_repos=sorted(set(d['repo'])),
                upstream_oof=upstream, upstream_oof_extra=extra,
                fit_x84=x, fit_q=d['q'], fit_repo=d['repo'], fit_y=d['y'],
                fit_x84_sha256=hashlib.sha256(x.tobytes()).hexdigest(), family=family, seed=seed,
                semantics='shared OOF evidence inputs; five repo-Poisson LightGBM head members')


def predict_strong(model, items, family):
    extra = base.predict_extra(model['evidence_model'], items, evidence_of(items), family)
    xx = _xdata(items, family, extra)
    joined = np.concatenate(xx)
    mu = np.stack([m['mean'].predict(joined) for m in model['members']])
    harm = np.stack([np.clip(m['harm'].predict(joined), 0., 1.) for m in model['members']])
    off = np.r_[0, np.cumsum([len(x) for x in xx])]
    assert np.isfinite(mu).all() and np.isfinite(harm).all()
    return [dict(x84=x, member_mu=mu[:, off[i]:off[i + 1]],
                 member_harm=harm[:, off[i]:off[i + 1]],
                 mu=mu[:, off[i]:off[i + 1]].mean(0),
                 harm=harm[:, off[i]:off[i + 1]].mean(0)) for i, x in enumerate(xx)]


def fit_weak(items, family, seed):
    """Same heads/member weights as strong, with only the old x46 inputs."""
    d = base.old.flatten(items, family)
    x = np.c_[d['x'], d['seed_x']]
    members = []
    for m in range(MEMBERS):
        w = member_weights(d['q'], d['repo'], m)
        members.append(dict(mean=booster(x, d['y'], seed + m, weight=w),
                            harm=booster(x, (d['y'] < -TOL).astype(int), seed + m, 'binary', w),
                            weights=w, bootstrap_member=m, bootstrap_key_seed=SEED))
    return dict(members=members, fit_query_ids=d['query_ids'], fit_repos=sorted(set(d['repo'])),
                semantics='same strong head capacity/bootstrap; x46 only')


def predict_weak(model, items):
    x = np.concatenate([np.c_[e['x'], e['seed_x']] for e in items])
    off = np.r_[0, np.cumsum([len(e['x']) for e in items])]
    mu = np.stack([m['mean'].predict(x) for m in model['members']])
    harm = np.stack([np.clip(m['harm'].predict(x), 0., 1.) for m in model['members']])
    return [dict(x84=x[off[i]:off[i + 1]], member_mu=mu[:, off[i]:off[i + 1]],
                 member_harm=harm[:, off[i]:off[i + 1]], mu=mu[:, off[i]:off[i + 1]].mean(0),
                 harm=harm[:, off[i]:off[i + 1]].mean(0)) for i in range(len(items))]


def crossfit_strong(items, family, seed):
    repos = np.array([e['repo'] for e in items], object)
    result = [None] * len(items)
    records = []
    for f, held in enumerate(grouped_folds(repos, INNER_FOLDS, seed)):
        tr = np.setdiff1d(np.arange(len(items)), held)
        assert not set(repos[tr]) & set(repos[held])
        model = fit_strong(subset(items, tr), family, seed + f)
        for i, p in zip(held, predict_strong(model, subset(items, held), family)):
            result[int(i)] = p
        records.append(dict(fit_query_ids=model['fit_query_ids'], fit_repos=model['fit_repos'],
                            held_query_ids=[int(items[i]['query_index']) for i in held],
                            held_repos=sorted(set(repos[held])), model=model))
    assert all(p is not None for p in result)
    return result, records


def query_hash(query):
    return int(hashlib.sha256((str(SEED) + ':' + query).encode()).hexdigest()[:12], 16) / 16**12


def choose_before(pred, tau):
    eligible = np.flatnonzero((pred['mu'] > 0) & (pred['harm'] <= tau))
    short = eligible[np.argsort(-pred['mu'][eligible], kind='stable')[:SHORTLIST]]
    return (int(short[0]) if len(short) else -1), short


def choose_rival(pred, winner, short, mode):
    options = np.asarray([j for j in short if int(j) != winner], int)
    if not len(options):
        return -1
    if mode == 'point':
        score = pred['mu'][options]
    elif mode == 'uq':
        score = np.maximum(pred['member_mu'][:, options] - pred['member_mu'][:, winner, None], 0.).mean(0)
    else:
        raise ValueError(mode)
    return int(options[np.lexsort((options, -pred['mu'][options], -score))[0]])


def plan_pair(item, pred, family, winner, rival, short):
    if winner < 0:
        return dict(shortlist=[], candidates=[], doc_ids=[], winner=-1, rival=-1,
                    slot_cost=0, kind='none')
    chosen = [winner] if rival < 0 else [winner, rival]
    # Communicate fixed candidate order to the historical gold-free geometry
    # helper. These synthetic values are not a posterior or UQ measurement.
    synthetic = np.zeros((1, len(pred['mu']), len(c.cg.SUPPORT)))
    zero = int(np.flatnonzero(c.cg.SUPPORT == 0)[0])
    synthetic[:, :, zero] = 1.
    order = chosen + [int(j) for j in short if int(j) not in chosen]
    for j, value in zip(order, np.linspace(.9, .1, len(order))):
        synthetic[:, j, zero] = 1 - value
        synthetic[:, j, -1] = value
    plan = c.plan_operation(c.observable(item), synthetic, family, 'winner' if rival < 0 else 'pair')
    plan.update(shortlist=list(map(int, short)), winner=winner, rival=rival,
                candidates=chosen, kind='winner' if rival < 0 else 'pair')
    assert plan['candidates'][0] == winner and len(plan['doc_ids']) <= 8
    return plan


def plan_for(item, pred, family, tau, mode):
    winner, short = choose_before(pred, tau)
    rival = choose_rival(pred, winner, short, mode) if winner >= 0 else -1
    return plan_pair(item, pred, family, winner, rival, short)


# Keep strong84 and rank geometry identical when zeroing the CE evidence.
BASE_DIM = 84 + 6
GEOMETRY_DIM = 8 * 4
CE_RAW_DIM = 8
CE_POOL_DIM = 3 * 32
FEATURE_DIM = BASE_DIM + GEOMETRY_DIM + CE_RAW_DIM + CE_POOL_DIM
CE_MASK = np.ones(FEATURE_DIM)
CE_MASK[BASE_DIM + GEOMETRY_DIM:] = 0.


def witness_x(item, pred, family, score, plan, j):
    docs = np.asarray(plan['doc_ids'], int)
    lookup = c.score_lookup(score)
    if any(int(d) not in lookup for d in docs):
        raise KeyError('Existing CE cache lacks a frozen planned witness')
    ix = np.array([lookup[int(d)] for d in docs], int)
    logits = np.asarray(score['logits'], float)[ix]
    cls = np.asarray(score['cls_projection'], float)[ix]
    assert cls.shape == (len(docs), 32)
    r0 = c.reciprocal(c.ranks_for(item['original_top100'], docs))
    ra = c.reciprocal(c.ranks_for(c.action_ranking(c.observable(item), j, family), docs))
    rival = next((int(v) for v in plan['candidates'] if int(v) != int(j)), -1)
    rb = r0 if rival < 0 else c.reciprocal(c.ranks_for(c.action_ranking(c.observable(item), rival, family), docs))
    pad = np.zeros((8, 4))
    pad[:len(docs)] = np.c_[np.ones(len(docs)), r0, ra, rb]
    raw_ce = np.zeros(8)
    raw_ce[:len(docs)] = logits
    shift = ra - r0
    def pool(w):
        return w @ cls / max(float(w.sum()), 1e-12)
    ordinary = [pred['mu'][j], pred['harm'][j],
                pred['mu'][rival] if rival >= 0 else 0.,
                pred['harm'][rival] if rival >= 0 else 0.,
                pred['mu'][j] - (pred['mu'][rival] if rival >= 0 else 0.), len(docs)/8]
    x = np.r_[pred['x84'][j], ordinary, pad.ravel(), raw_ce,
              pool(r0), pool(ra), pool(np.maximum(shift, 0)) - pool(np.maximum(-shift, 0))]
    assert x.shape == (FEATURE_DIM,) and np.isfinite(x).all()
    return x


def verifier_data(items, scores, family, predictions):
    xx, yy, hh, mm, qq, repos, plans = [], [], [], [], [], [], []
    for e, ce, pred in zip(items, scores, predictions):
        seen = set()
        for tau in GRID:
            winner, short = choose_before(pred, tau)
            if winner < 0:
                continue
            rivals = [int(j) for j in short if int(j) != winner] or [-1]
            for rival in rivals:
                plan = plan_pair(e, pred, family, winner, rival, short)
                signature = (tuple(plan['candidates']), tuple(plan['doc_ids']))
                if signature in seen:
                    continue
                seen.add(signature)
                for j in plan['candidates']:
                    xx.append(witness_x(e, pred, family, ce, plan, j))
                    yy.append(float(e['y'][family][j]))
                    hh.append(float(e['y'][family][j] < -TOL))
                    mm.append(pred['member_mu'][:, j])
                    qq.append(int(e['query_index'])); repos.append(e['repo'])
                    plans.append(dict(query_index=int(e['query_index']), candidate_index=int(j), plan=plan, tau=tau))
    if not xx:
        return dict(x=np.empty((0, FEATURE_DIM)), y=np.empty(0), harm=np.empty(0),
                    baseline_member_mu=np.empty((0, MEMBERS)), q=np.empty(0, int), repo=np.empty(0, object), plans=[])
    return dict(x=np.array(xx), y=np.array(yy), harm=np.array(hh), baseline_member_mu=np.array(mm),
                q=np.array(qq), repo=np.array(repos, object), plans=plans)


def fit_updates(data, seed):
    models = {}
    for arm, mask in (('zero', CE_MASK), ('full', np.ones(FEATURE_DIM))):
        members = []
        if not len(data['x']):
            models[arm] = dict(members=[], mask=mask, available=False)
            continue
        for m in range(MEMBERS):
            w = member_weights(data['q'], data['repo'], m)
            residual = data['y'] - data['baseline_member_mu'][:, m]
            members.append(dict(mean=booster(data['x'] * mask, residual, seed + m, weight=w),
                                harm=booster(data['x'] * mask, data['harm'], seed + m, 'binary', w),
                                weights=w, bootstrap_member=m, bootstrap_key_seed=SEED))
        models[arm] = dict(members=members, mask=mask, available=True,
                           fit_query_ids=sorted(set(map(int, data['q']))), fit_repos=sorted(set(data['repo'])))
    return models


def fit_pipeline(items, scores, family, seed):
    oof, records = crossfit_strong(items, family, seed + 1000)
    training = verifier_data(items, scores, family, oof)
    updates = fit_updates(training, seed + 2000)
    strong = fit_strong(items, family, seed)
    return dict(strong=strong, updates=updates, training=training, before_oof=oof, strong_oof=records,
                seed=seed, family=family, fit_query_ids=[int(e['query_index']) for e in items],
                fit_repos=sorted({e['repo'] for e in items}))


def after_pair(pipeline, item, pred, score, family, plan, arm, rho=RHO):
    updated = []
    model = pipeline['updates'][arm]
    for j in plan['candidates']:
        x = witness_x(item, pred, family, score, plan, j)
        if model['available']:
            residual = np.array([m['mean'].predict((x * model['mask'])[None])[0] for m in model['members']])
            ce_harm = np.array([np.clip(m['harm'].predict((x * model['mask'])[None])[0], 0., 1.) for m in model['members']])
        else:
            residual = np.zeros(MEMBERS); ce_harm = pred['member_harm'][:, j].copy()
        mu = pred['member_mu'][:, j] + rho * residual
        harm = (pred['member_harm'][:, j].copy() if rho == 0 else
                (1 - HARM_BLEND) * pred['member_harm'][:, j] + HARM_BLEND * ce_harm)
        updated.append(dict(candidate_index=int(j), member_mu=mu, member_harm=harm,
                            mu=float(mu.mean()), harm=float(harm.mean()), residual=residual, ce_harm=ce_harm, x=x))
    return updated


def infer_one(pipeline, item, pred, score, family, tau, policy, rho=RHO):
    gate = 1. if policy.endswith('_ungated') else tau
    winner, short = choose_before(pred, gate)
    chosen = winner
    plan = None; updated = []; verified = False
    h = query_hash(item['query'])
    if policy.startswith('ce_') and winner >= 0 and h < .25:
        mode = 'uq' if policy == PRIMARY else 'point'
        arm = 'zero' if policy == 'ce_zero_point' else 'full'
        plan = plan_for(item, pred, family, tau, mode)
        verified = True
        updated = after_pair(pipeline, item, pred, score, family, plan, arm, rho)
        eligible = [u for u in updated if u['mu'] > 0 and u['harm'] <= tau]
        chosen = sorted(eligible, key=lambda u: (-u['mu'], u['candidate_index']))[0]['candidate_index'] if eligible else -1
    if chosen < 0:
        ranking = tuple(map(int, item['original_top100'])); chosen_mu = chosen_harm = 0.
    else:
        ranking = tuple(map(int, c.action_ranking(c.observable(item), chosen, family)))
        selected = next((u for u in updated if u['candidate_index'] == chosen), None)
        chosen_mu = float(selected['mu'] if selected else pred['mu'][chosen])
        chosen_harm = float(selected['harm'] if selected else pred['harm'][chosen])
    identity = chosen >= 0 and ranking == tuple(map(int, item['original_top100']))
    accepted = chosen >= 0 and not identity
    docs = [] if plan is None else plan['doc_ids']
    return dict(raw_proposal=int(chosen), chosen=int(chosen) if accepted else -1,
                baseline_winner=winner, accepted=accepted, identity_fallback=identity,
                ranking=ranking, mu=chosen_mu, harm=chosen_harm, plan=plan, updated=updated,
                verified=verified, query_hash=h, tau=float(tau),
                billed_slots=8 if verified else 0, actual_pairs=len(docs),
                query_words=len(item['query'].split()) * len(docs),
                code_words=sum(item.get('code_word_counts', {}).get(int(d), 0) for d in docs),
                lmax=float(worst_case_loss(item['original_top100'], ranking)))


def rr(top, target):
    where = np.flatnonzero(np.asarray(top, int)[:100] == int(target))
    return 1 / (int(where[0]) + 1) if len(where) else 0.


def attach_labels(trace, item, family):
    y = rr(trace['ranking'], item['query_index']) - rr(item['original_top100'], item['query_index'])
    assert trace['accepted'] or abs(y) < TOL
    assert y >= -trace['lmax'] - TOL
    if trace['accepted']:
        assert abs(y - item['y'][family][trace['chosen']]) < TOL
    return dict(trace, query_index=int(item['query_index']), repo=item['repo'],
                delta_rr=float(y), harmful=bool(y < -TOL))


def inner_pipeline_oof(items, scores, family, seed):
    """Refit all strong, evidence, CE heads before each inner held prediction."""
    repos = np.array([e['repo'] for e in items], object)
    outputs = {tau: [None] * len(items) for tau in GRID}
    records = []
    for f, held in enumerate(grouped_folds(repos, INNER_FOLDS, seed)):
        tr = np.setdiff1d(np.arange(len(items)), held)
        assert not set(repos[tr]) & set(repos[held])
        pipeline = fit_pipeline(subset(items, tr), subset(scores, tr), family, seed + f)
        for i, pred in zip(held, predict_strong(pipeline['strong'], subset(items, held), family)):
            for tau in GRID:
                t = infer_one(pipeline, items[int(i)], pred, scores[int(i)], family, tau, PRIMARY)
                outputs[tau][int(i)] = attach_labels(t, items[int(i)], family)
        records.append(dict(fit_query_ids=pipeline['fit_query_ids'], fit_repos=pipeline['fit_repos'],
                            held_query_ids=[int(items[i]['query_index']) for i in held],
                            held_repos=sorted(set(repos[held])), pipeline=pipeline))
    assert all(all(t is not None for t in v) for v in outputs.values())
    candidates = []; diagnostics = []
    for tau, traces in outputs.items():
        active = [t for t in traces if t['accepted']]
        n = len(active); h = sum(t['harmful'] for t in active)
        utility = sum(t['delta_rr'] for t in traces)
        row = dict(tau=tau, n=n, harmful=h, empirical_harm=h/n if n else None,
                   utility_sum=utility, delta_mrr=utility/len(items), feasible=bool(n > 0 and h/n <= .05))
        diagnostics.append(row)
        if row['feasible']:
            candidates.append(row)
    tau = sorted(candidates, key=lambda r: (-r['utility_sum'], -r['n'], r['tau']))[0]['tau'] if candidates else GRID[0]
    return dict(tau=float(tau), feasible=bool(candidates), selection=diagnostics,
                outputs=outputs, records=records,
                fallback='predeclared .025 when no nonempty empirical<=.05 fit-OOF policy')
