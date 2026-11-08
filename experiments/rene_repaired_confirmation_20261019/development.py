from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from experiments.rene_repaired_confirmation_20261019.common import (
    OUT, SEED, SEEDS, ARMS, FAMILIES, load, save, sha, write_json, read_json,
    observable, protected_hashes, require_runtime, rr, utc, assert_disjoint,
)
from experiments.rene_repaired_confirmation_20261019.model import train_pipeline, predict_pipeline

HERE = Path(__file__).resolve().parent


def ordered(items, key):
    return sorted(items, key=lambda e: hashlib.sha256((str(SEED) + ':' + key + ':' + e['repo']).encode()).hexdigest())


def role_plan(items):
    outer = np.array_split(ordered(items, 'outer'), 4)
    folds = []
    for fold in range(4):
        held = list(outer[fold])
        train = [e for f in range(4) if f != fold for e in outer[f]]
        train = ordered(train, 'roles:' + str(fold))
        roles = dict(gain=train[:192], update=train[192:288], risk=train[288:], held=held)
        assert_disjoint({k: [e['repo'] for e in v] for k, v in roles.items()})
        folds.append({k: [int(e['query_index']) for e in v] for k, v in roles.items()})
    final = ordered(items, 'final')
    final = dict(gain=final[:256], update=final[256:384], risk=final[384:])
    return folds, {k: [int(e['query_index']) for e in v] for k, v in final.items()}


def compact(trace):
    fields = ('query_index', 'chosen', 'accepted', 'ranking', 'verified', 'priority', 'mu', 'harm',
              'post_harm', 'post_risk_rejected', 'identity_fallback', 'billed_slots',
              'actual_pairs', 'executed_ce_tokens')
    return {k: trace[k] for k in fields if k in trace}


def delta(trace, item):
    return rr(trace['ranking'], item['query_index']) - rr(item['original_top100'], item['query_index'])


def summarize(traces, byid):
    values = np.array([delta(t, byid[t['query_index']]) for t in traces])
    adopted = np.array([t['accepted'] for t in traces])
    return dict(queries=len(traces), accepted=int(adopted.sum()), coverage=float(adopted.mean()),
                delta_mrr=float(values.mean()), positive=int((values > 1e-10).sum()),
                harmful=int((values < -1e-10).sum()), paired_sd=float(values.std(ddof=1)),
                actual_pairs=sum(t['actual_pairs'] for t in traces),
                executed_ce_tokens=sum(t['executed_ce_tokens'] for t in traces))


def freeze_design(items):
    folds, final = role_plan(items)
    path = OUT / 'development/DESIGN.json'
    fixed = dict(seeds=SEEDS, arms=ARMS, families=FAMILIES, folds=folds, final_roles=final,
                 historical_development=True, test_used=False, protected_hashes=protected_hashes(),
                 priority='75th higher quantile of gain-role repository-OOF max-positive-mean',
                 thresholds=dict(native_harm=.1, threeway_harm=.075, mean=0., residual_weight=.5),
                 stage_roles='disjoint repositories: gain -> CE update -> selected-action risk; outer held is untouched',
                 no_seed_ablation='refit gain72 and CE214 after removing actual seed columns; no inference zeroing',
                 common_features='base34 retains generation entropy; this is a multi-seed feature ablation, not all UQ',
                 power=dict(sesoi=.005, power=.8, familywise_alpha=.05, key_contrasts=2,
                            sd_upper_confidence=.8, sd_floor=.05, minimum_queries=1024,
                            rounding=128, sensitivity_sesoi=[.0025, .005, .01],
                            method='max development OOF paired SD across stages/families, 80% chi-square upper SD'),
                 primary='fusion seed_uq risk minus no_seed_uq risk',
                 key_secondary='fusion seed_uq risk minus NoRewrite',
                 multiplicity='two key contrasts use Bonferroni 97.5% intervals and Holm-adjusted paired t p-values',
                 cost_comparison='same gain-priority algorithm, actual CE costs reported; not claimed exactly cost-matched',
                 risk='one fixed seed_uq fusion risk policy; CP upper descriptive on finite corpus, no deployment certificate')
    write_json(path, fixed, immutable=True)
    return fixed


def calculate_power(outputs, items, design):
    byid = {e['query_index']: e for e in items}
    candidates = []
    for family in FAMILIES:
        for stage in ('mean', 'base', 'verified', 'risk'):
            aa = {t['query_index']: t for t in outputs[family]['seed_uq'][stage]}
            bb = {t['query_index']: t for t in outputs[family]['no_seed_uq'][stage]}
            paired = np.array([delta(aa[q], e) - delta(bb[q], e) for q, e in byid.items()])
            absolute = np.array([delta(aa[q], e) for q, e in byid.items()])
            for contrast, values in (('uq_increment', paired), ('versus_no_rewrite', absolute)):
                sd = float(values.std(ddof=1))
                upper = sd * math.sqrt((len(values) - 1) / stats.chi2.ppf(.2, len(values) - 1))
                candidates.append(dict(family=family, stage=stage, contrast=contrast, sd=sd, sd_upper80=upper,
                                       development_mean=float(values.mean())))
    cfg = design['power']
    planning_sd = max(cfg['sd_floor'], max(c['sd_upper80'] for c in candidates))
    z = stats.norm.ppf(1 - cfg['familywise_alpha'] / cfg['key_contrasts'] / 2) + stats.norm.ppf(cfg['power'])
    required = int(math.ceil(z * z * planning_sd * planning_sd / cfg['sesoi'] ** 2))
    n = int(math.ceil(max(required, cfg['minimum_queries']) / cfg['rounding']) * cfg['rounding'])
    sensitivity = {str(s): int(math.ceil(z * z * planning_sd * planning_sd / s ** 2)) for s in cfg['sensitivity_sesoi']}
    return dict(computed_at_utc=utc(), design_sha256=sha(OUT / 'development/DESIGN.json'),
                development_queries=len(items), one_query_per_repository=True, planning_sd=planning_sd,
                observations=candidates, target_effect=cfg['sesoi'], target_power=cfg['power'],
                required_unrounded=required, confirmation_queries=n, sensitivity_n=sensitivity,
                formula='ceil((z_(1-alpha/(2*m))+z_power)^2 * sd_upper^2 / SESOI^2)',
                fresh_outcomes_opened=False, fixed_corpus_conditional=True,
                model_unseen_claim=False, risk_certification_power_not_guaranteed=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    runtime = require_runtime()
    items = load(OUT / 'development/items.pkl')
    scores = load(OUT / 'development/scores.pkl')
    byid = {e['query_index']: e for e in items}
    scoreby = {e['query_index']: e for e in scores}
    design = freeze_design(items)
    began = time.perf_counter()
    outputs = {f: {a: {s: [] for s in ('mean', 'base', 'verified', 'risk')} for a in ARMS} for f in FAMILIES}
    with threadpool_limits(limits=2):
        for fold, roles in enumerate(design['folds']):
            for family in FAMILIES:
                for arm in ARMS:
                    path = OUT / f'development/folds/{family}_{arm}_{fold}.pkl'
                    if path.exists():
                        result = load(path)
                        if result['design_sha256'] != sha(OUT / 'development/DESIGN.json'):
                            raise ValueError('Checkpoint design changed')
                    else:
                        fit_ids = roles['gain'] + roles['update'] + roles['risk']
                        model = train_pipeline([byid[q] for q in fit_ids], [scoreby[q] for q in fit_ids],
                                               {k: v for k, v in roles.items() if k != 'held'}, family, arm, SEED + fold)
                        vals, predictions, sessions = predict_pipeline(model, [observable(byid[q]) for q in roles['held']],
                                                                      [scoreby[q] for q in roles['held']])
                        result = dict(outputs={s: [compact(t) for t in ts] for s, ts in vals.items()},
                                      predictions=[dict(query_index=q, mu=p['mu'], harm=p['harm'])
                                                   for q, p in zip(roles['held'], predictions)],
                                      model=model, design_sha256=sha(OUT / 'development/DESIGN.json'))
                        if not args.smoke:
                            save(path, result)
                    for stage, values in result['outputs'].items():
                        outputs[family][arm][stage].extend(values)
                    print(f'Development fold {fold} {family} {arm}: ' + str(summarize(result['outputs']['risk'], byid)) +
                          f' elapsed={time.perf_counter() - began:.1f}s', flush=True)
                    if args.smoke:
                        print('Smoke only: no held confirmation data used', flush=True)
                        return
        save(OUT / 'development/OOF_OUTPUTS.pkl', outputs) if not (OUT / 'development/OOF_OUTPUTS.pkl').exists() else None
        summary = {f: {a: {s: summarize(ts, byid) for s, ts in stages.items()} for a, stages in arms.items()}
                   for f, arms in outputs.items()}
        write_json(OUT / 'development/OOF_SUMMARY.json', summary, immutable=True)
        power = calculate_power(outputs, items, design)
        existing = OUT / 'development/POWER.json'
        if existing.exists():
            old = read_json(existing)
            power['computed_at_utc'] = old['computed_at_utc']
        write_json(existing, power, immutable=True)
        print('FROZEN POWER ' + str({k: power[k] for k in ('planning_sd', 'confirmation_queries', 'required_unrounded')}), flush=True)
        for family in FAMILIES:
            for arm in ARMS:
                path = OUT / f'models/{family}_{arm}.joblib'
                if not path.exists():
                    roles = design['final_roles']
                    model = train_pipeline(items, scores, roles, family, arm, SEED + 2000)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, path)
                print('Frozen retrained full flow: ' + str(path), flush=True)
    write_json(OUT / 'development/TRAINING_COMPLETE.json', dict(completed=True, runtime=runtime,
               models={str(p.relative_to(ROOT)): sha(p) for p in sorted((OUT / 'models').glob('*.joblib'))},
               power_sha256=sha(OUT / 'development/POWER.json'), test_used=False,
               formal_components_changed=False), immutable=True)


if __name__ == '__main__':
    main()
