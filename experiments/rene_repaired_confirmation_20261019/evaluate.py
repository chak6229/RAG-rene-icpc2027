"""Open outcomes only after verifying an immutable seal of actual decisions."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from crsu.core import rrf
from experiments.rene_repaired_confirmation_20261019.common import (
    OUT, ARMS, FAMILIES, TOL, SEED, read_json, write_json, load, save, sha, utc, rr,
    check_hashes, require_runtime,
)

FRESH = OUT / 'fresh'


def interval(values, alpha=.05, reps=10000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED + 37)
    boot = []
    for start in range(0, reps, 256):
        ix = rng.integers(len(values), size=(min(256, reps - start), len(values)))
        boot.extend(values[ix].mean(1).tolist())
    return np.quantile(boot, [alpha / 2, 1 - alpha / 2]).tolist()


def cp_upper(h, n, alpha=.05):
    return float(stats.beta.ppf(1 - alpha, h + 1, n - h)) if 0 <= h < n else 1.


def summary(traces, originals, targetby, with_interval=True):
    baseline = np.array([rr(originals[t['query_index']], targetby[t['query_index']]) for t in traces])
    outcome = np.array([rr(t['ranking'], targetby[t['query_index']]) for t in traces])
    delta = outcome - baseline
    accepted = np.array([t['accepted'] for t in traces], bool)
    if np.any(abs(delta[~accepted]) > TOL):
        raise ValueError('Rejected/identity action changed the metric')
    n = int(accepted.sum()); harmful = int((delta < -TOL).sum())
    z = dict(queries=len(traces), no_rewrite_mrr=float(baseline.mean()), policy_mrr=float(outcome.mean()),
             delta_mrr=float(delta.mean()), accepted=n, coverage=float(accepted.mean()),
             positive=int((delta > TOL).sum()), harmful=harmful,
             harmful_rate=harmful / n if n else None, unchanged_accepted=int(n - (abs(delta) > TOL).sum()),
             cp_upper95=cp_upper(harmful, n), cp_numerical_pass=bool(n and cp_upper(harmful, n) <= .1),
             verified=sum(t['verified'] for t in traces),
             billed_slots=sum(t['billed_slots'] for t in traces),
             actual_pairs=sum(t['actual_pairs'] for t in traces),
             executed_ce_tokens=sum(t['executed_ce_tokens'] for t in traces))
    if with_interval:
        z['descriptive_ci95'] = interval(delta)
    return z, delta


def core_table(summaries):
    # Canonical 12-digit metrics allow a separate stdlib evaluator to reproduce the table hash.
    numeric = ('no_rewrite_mrr', 'policy_mrr', 'delta_mrr', 'coverage', 'harmful_rate')
    integer = ('queries', 'accepted', 'positive', 'harmful', 'unchanged_accepted',
               'verified', 'billed_slots', 'actual_pairs', 'executed_ce_tokens')
    return {key: {**{k: None if value[k] is None else round(value[k], 12) for k in numeric},
                  **{k: value[k] for k in integer}} for key, value in sorted(summaries.items())}


def holm(pvalues):
    names = sorted(pvalues, key=lambda k: pvalues[k])
    maximum, result = 0., {}
    for i, name in enumerate(names):
        maximum = max(maximum, min(1., (len(names) - i) * pvalues[name]))
        result[name] = maximum
    return result


def contrast(values):
    sd = float(np.std(values, ddof=1))
    pvalue = float(stats.ttest_1samp(values, 0.).pvalue) if sd else (1. if not np.mean(values) else 0.)
    return dict(mean=float(np.mean(values)), paired_sd=sd, ci95=interval(values),
                simultaneous_ci97_5=interval(values, .025), paired_t_p_two_sided=pvalue,
                positive_queries=int((values > TOL).sum()), negative_queries=int((values < -TOL).sum()))


def evaluate():
    if (OUT / 'RESULTS.json').exists():
        audit = read_json(OUT / 'EVALUATION_SEAL.json')
        check_hashes(audit['outputs'])
        print('Verified previously sealed evaluation; refusing a second outcome-driven run', flush=True)
        return
    protocol = read_json(FRESH / 'PROTOCOL.json')
    check_hashes(protocol['binding_hashes'])
    check_hashes(protocol['protected_hashes'])
    seal = read_json(FRESH / 'DECISION_SEAL.json')
    if not seal['actual_decisions'] or seal['outcomes_decoded']:
        raise ValueError('A seal of actual label-free decisions is required')
    for p, key in [('label_free_decisions.pkl', 'decisions_sha256'), ('observable_items.pkl', 'observations_sha256'),
                   ('ce_scores.pkl', 'scores_sha256'), ('PROTOCOL.json', 'protocol_sha256')]:
        if sha(FRESH / p) != seal[key]:
            raise ValueError('Changed sealed input: ' + p)
    if sha(FRESH / 'private_outcomes.json') != protocol['private_outcomes_sha256']:
        raise ValueError('Outcome mapping changed')
    # The first decode of the outcome mapping is below this point.
    write_json(OUT / 'OUTCOME_OPENING.json', dict(opened_at_utc=utc(),
               decision_seal_sha256=sha(FRESH / 'DECISION_SEAL.json'), labels_used_to_adjust_policy=False), immutable=True)
    outcomes = read_json(FRESH / 'private_outcomes.json')
    targetby = {r['query_index']: int(r['target_doc_id']) for r in outcomes}
    items = load(FRESH / 'observable_items.pkl')
    byid = {e['query_index']: e for e in items}
    originals = {q: e['original_top100'] for q, e in byid.items()}
    saved = load(FRESH / 'label_free_decisions.pkl')
    if set(targetby) != set(byid) or len({r['repo'] for r in outcomes}) != len(items):
        raise ValueError('Confirmation query/repository alignment failure')
    summaries, deltas = {}, {}
    qids = [e['query_index'] for e in items]
    for name, traces in saved['outputs'].items():
        if [t['query_index'] for t in traces] != qids:
            raise ValueError('Decision order mismatch: ' + name)
        family = name.split('/')[0]
        for t in traces:
            e = byid[t['query_index']]
            if t['chosen'] < 0:
                expected = tuple(map(int, e['original_top100']))
            else:
                top = e['candidate_top100'][t['chosen']]
                expected = tuple(map(int, top if family == 'rewrite' else rrf(e['original_top100'], top)))
            if tuple(t['ranking']) != expected or t['accepted'] != (expected != tuple(e['original_top100'])):
                raise ValueError('Sealed action/ranking inconsistency')
        summaries[name], deltas[name] = summary(traces, originals, targetby)
    risk_cost_preserved = True
    for family in FAMILIES:
        for arm in ARMS:
            before = saved['outputs'][f'{family}/{arm}/verified']
            after = saved['outputs'][f'{family}/{arm}/risk']
            for a, b in zip(before, after):
                if any(a[k] != b[k] for k in ('actual_pairs', 'billed_slots', 'executed_ce_tokens', 'verified')):
                    raise ValueError('Post-risk rejection erased already incurred CE costs')
                if b['accepted'] and (not a['accepted'] or b['ranking'] != a['ranking']):
                    raise ValueError('Risk gate reselected an action')
    key_values = dict(uq_increment=deltas['fusion/seed_uq/risk'] - deltas['fusion/no_seed_uq/risk'],
                      versus_no_rewrite=deltas['fusion/seed_uq/risk'])
    contrasts = {name: contrast(values) for name, values in key_values.items()}
    adjusted = holm({name: value['paired_t_p_two_sided'] for name, value in contrasts.items()})
    for name in contrasts:
        contrasts[name]['holm_adjusted_p'] = adjusted[name]
        contrasts[name]['positive_confirmed'] = contrasts[name]['simultaneous_ci97_5'][0] > 0 and adjusted[name] < .05
    stages = {f'{family}/{stage}': contrast(deltas[f'{family}/seed_uq/{stage}'] - deltas[f'{family}/no_seed_uq/{stage}'])
              for family in FAMILIES for stage in ('mean', 'base', 'verified', 'risk', 'fixed50')}
    for value in stages.values():
        value['descriptive_only'] = True
    oracle = {}
    for family in FAMILIES:
        all_delta, seed_delta = [], []
        for e in items:
            target = targetby[e['query_index']]
            original = rr(e['original_top100'], target)
            gains = [rr(top if family == 'rewrite' else rrf(e['original_top100'], top), target) - original
                     for top in e['candidate_top100']]
            all_delta.append(max([0.] + gains))
            seed_delta.append(max([0.] + [g for g, a in zip(gains, e['seed_appearances']) if 101 in a]))
        oracle[family] = dict(all_candidate_delta_mrr=float(np.mean(all_delta)), seed101_best_delta_mrr=float(np.mean(seed_delta)),
                              queries_with_positive_candidate=int((np.array(all_delta) > TOL).sum()),
                              label_access='oracle diagnostic after opening; not a deployed selection method')
    quality = {}
    for name, predictions in saved['raw_predictions'].items():
        family = name.split('/')[0]
        errors, briers, yy, pp, weights = [], [], [], [], []
        for e, pred in zip(items, predictions):
            target = targetby[e['query_index']]
            y = np.array([rr(top if family == 'rewrite' else rrf(e['original_top100'], top), target)
                          - rr(e['original_top100'], target) for top in e['candidate_top100']])
            if len(y):
                errors.append(float(np.mean((pred['mu'] - y) ** 2)))
                briers.append(float(np.mean((pred['harm'] - (y < -TOL)) ** 2)))
                yy.extend((y < -TOL).tolist()); pp.extend(pred['harm'].tolist()); weights.extend([1 / len(y)] * len(y))
        quality[name] = dict(query_weighted_rmse=math.sqrt(float(np.mean(errors))),
                             query_weighted_harm_brier=float(np.mean(briers)),
                             weighted_harm_auc=float(roc_auc_score(yy, pp, sample_weight=weights)) if len(set(yy)) > 1 else None)
    baseline_mrr = float(np.mean([rr(e['original_top100'], targetby[e['query_index']]) for e in items]))
    result = dict(completed=True, evaluation_at_utc=utc(), queries=len(items), corpus=protocol['corpus'],
                  no_rewrite_mrr=baseline_mrr, methods=summaries, primary_contrasts=contrasts,
                  stage_uq_increment=stages, candidate_prediction=quality, oracle=oracle,
                  planned_power=read_json(OUT / 'development/POWER_PRIMARY.json'),
                  actual_planning_mde=protocol['actual_planning_mde'],
                  sample_plan_complete=protocol['sample_plan_complete'],
                  risk_cost_preserved=risk_cost_preserved, ce_physical_cost=read_json(FRESH / 'CE_COMPLETE.json'),
                  decision_seal_sha256=sha(FRESH / 'DECISION_SEAL.json'), protocol_sha256=sha(FRESH / 'PROTOCOL.json'),
                  test_used=False, risk_certified=False, formal_components_changed=False,
                  interpretation='finite frozen-corpus downstream repository confirmation; upstream retriever train exposure disclosed')
    write_json(OUT / 'core_table.json', core_table(summaries), immutable=True)
    save(OUT / 'per_query_outcomes.pkl', dict(query_ids=qids, delta=deltas, key_contrasts=key_values))
    write_json(OUT / 'RESULTS.json', result, immutable=True)
    write_json(OUT / 'EVALUATION_SEAL.json', dict(evaluated_at_utc=utc(),
               decision_seal_sha256=sha(FRESH / 'DECISION_SEAL.json'),
               outputs={str((OUT / p).relative_to(ROOT)): sha(OUT / p)
                        for p in ('RESULTS.json', 'core_table.json', 'per_query_outcomes.pkl', 'OUTCOME_OPENING.json')}), immutable=True)
    print('KEY CONTRASTS ' + str(contrasts), flush=True)
    print('FUSION ' + str({k: summaries[k] for k in ('fusion/no_seed_uq/risk', 'fusion/seed_uq/risk')}), flush=True)


if __name__ == '__main__':
    require_runtime()
    evaluate()
