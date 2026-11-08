"""Standalone clean-directory retraining, prediction and independent RR evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.rene_strict_confirmation_20261019.common import (
    OUT, SEED, ARMS, FAMILIES, load, read_json, write_json, sha, observable,
    reject_gold, canonical_bytes, jsonable,
)
from experiments.rene_strict_confirmation_20261019.model import train_pipeline, predict_pipeline
from experiments.rene_strict_confirmation_20261019.development import compact
from experiments.rene_strict_confirmation_20261019.decide import fixed50, baseline
from experiments.rene_strict_confirmation_20261019 import weak_control
from experiments.rene_strict_confirmation_20261019 import budget_control


def fingerprint(outputs):
    fields = ('query_index', 'chosen', 'accepted', 'ranking', 'verified',
              'billed_slots', 'actual_pairs', 'executed_ce_tokens')
    value = {name: [{k: t[k] for k in fields} for t in ts] for name, ts in outputs.items()}
    return hashlib.sha256(canonical_bytes(jsonable(value))).hexdigest()


def independent_core(outputs, items, outcomes):
    """Metric implementation uses plain sequence positions and math.fsum, no old summary."""
    targets = {int(r['query_index']): int(r['target_doc_id']) for r in outcomes}
    originals = {int(e['query_index']): list(map(int, e['original_top100'])) for e in items}

    def reciprocal(sequence, target):
        try:
            return 1. / (list(sequence)[:100].index(target) + 1)
        except ValueError:
            return 0.

    table = {}
    for name, traces in sorted(outputs.items()):
        old = [reciprocal(originals[t['query_index']], targets[t['query_index']]) for t in traces]
        new = [reciprocal(t['ranking'], targets[t['query_index']]) for t in traces]
        delta = [b - a for a, b in zip(old, new)]
        n = len(traces)
        accepted = sum(bool(t['accepted']) for t in traces)
        positive = sum(d > 1e-10 for d in delta)
        harmful = sum(d < -1e-10 for d in delta)
        table[name] = dict(no_rewrite_mrr=round(math.fsum(old) / n, 12),
                          policy_mrr=round(math.fsum(new) / n, 12),
                          delta_mrr=round(math.fsum(delta) / n, 12),
                          coverage=round(accepted / n, 12),
                          harmful_rate=round(harmful / accepted, 12) if accepted else None,
                          queries=n, accepted=accepted, positive=positive, harmful=harmful,
                          unchanged_accepted=accepted - positive - harmful,
                          verified=sum(bool(t['verified']) for t in traces),
                          billed_slots=sum(int(t['billed_slots']) for t in traces),
                          actual_pairs=sum(int(t['actual_pairs']) for t in traces),
                          executed_ce_tokens=sum(int(t['executed_ce_tokens']) for t in traces))
    return table


def compare_numeric_traces(actual, expected):
    largest = 0.
    for name, traces in actual.items():
        for a, b in zip(traces, expected[name]):
            for field in ('mu', 'harm', 'post_harm'):
                if field not in a or field not in b:
                    continue
                if a[field] is None or b[field] is None:
                    if a[field] is not b[field]:
                        raise ValueError('Replayed probability availability differs')
                else:
                    error = abs(float(a[field]) - float(b[field]))
                    largest = max(largest, error)
                    if error > 1e-12:
                        raise ValueError('Refitted action probability differs: ' + name + '/' + field)
    return largest


def main():
    packet = read_json(ROOT / 'PACKET_MANIFEST.json')
    for name, expected in packet['files'].items():
        if name.endswith('private_outcomes.json'):
            continue
        if sha(ROOT / name) != expected:
            raise ValueError('Packet hash mismatch: ' + name)
    if '--verify-only' in sys.argv:
        return
    original = packet['original_workspace'].replace('\\', '/').casefold().rstrip('/') + '/'
    runtime = str(Path(sys.prefix).resolve()).replace('\\', '/').casefold().rstrip('/') + '/'
    state = dict(outcomes_allowed=False, forbidden_original_reads=0, premature_outcome_reads=0)

    def guard(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
            return
        p = str(Path(str(args[0])).resolve()).replace('\\', '/').casefold()
        if p.startswith(original) and not p.startswith(runtime):
            state['forbidden_original_reads'] += 1
            raise PermissionError('Clean replay attempted an original-workspace read: ' + p)
        if 'private_outcomes' in p and not state['outcomes_allowed']:
            state['premature_outcome_reads'] += 1
            raise PermissionError('Fresh outcomes are sealed during retraining and prediction')
    sys.addaudithook(guard)
    dev = load(OUT / 'development/items.pkl')
    dev_scores = load(OUT / 'development/scores.pkl')
    roles = read_json(OUT / 'development/DESIGN.json')['final_roles']
    items = [observable(e) for e in load(OUT / 'fresh/observable_items.pkl')]
    scores = load(OUT / 'fresh/ce_scores.pkl')
    expected = load(OUT / 'fresh/label_free_decisions.pkl')
    reject_gold(expected)
    outputs, budget_outputs, max_prediction_error = {}, {}, 0.
    budget_scores = load(OUT / 'fresh/budget_ce_scores.pkl')
    budget_expected = load(OUT / 'fresh/budget_label_free_decisions.pkl')
    with threadpool_limits(limits=2):
        for family in FAMILIES:
            fitted_pair = {}
            for arm in ARMS:
                # This is a real refit from the frozen development roles in this new directory.
                fitted = train_pipeline(dev, dev_scores, roles, family, arm, SEED + 2000)
                fitted_pair[arm] = fitted
                values, predictions, sessions = predict_pipeline(fitted, items, scores)
                for stage, traces in values.items():
                    outputs[f'{family}/{arm}/{stage}'] = [compact(t) for t in traces]
                outputs[f'{family}/{arm}/fixed50'] = fixed50(items, predictions, family)
                old = expected['raw_predictions'][f'{family}/{arm}']
                for p, e in zip(predictions, old):
                    for field in ('mu', 'harm', 'member_mu', 'member_harm'):
                        if np.asarray(p[field]).size:
                            max_prediction_error = max(max_prediction_error, float(np.max(abs(p[field] - e[field]))))
                        np.testing.assert_allclose(p[field], e[field], atol=1e-12, rtol=0)
                print('Clean refit and actual fresh prediction: ' + family + '/' + arm, flush=True)
            paired_risks = budget_control.fit_risks(fitted_pair, dev, dev_scores, family)
            paired_outputs, _, _ = budget_control.infer_pair(fitted_pair, paired_risks, items, budget_scores, family)
            budget_outputs.update(paired_outputs)
            gain_set = set(roles['gain'])
            fit_items = [e for e in dev if e['query_index'] in gain_set]
            for dimension in (34, 46):
                fitted = weak_control.fit(fit_items, family, dimension)
                outputs[f'{family}/base{dimension}/mean'] = [compact(t) for t in weak_control.predict(fitted, items)]
            outputs[f'{family}/seed101_IG'] = baseline(items, family, True)
            outputs[f'{family}/multiseed_IG'] = baseline(items, family, False)
    actual = fingerprint(outputs)
    if actual != fingerprint(expected['outputs']) or actual != packet['expected_decisions_fingerprint']:
        raise ValueError('Refitted decision/ranking/cost fingerprint does not match the seal')
    budget_fingerprint = fingerprint(budget_outputs)
    if budget_fingerprint != fingerprint(budget_expected['outputs']) or budget_fingerprint != packet['expected_budget_fingerprint']:
        raise ValueError('Equal-budget refitted decisions do not reproduce')
    max_prediction_error = max(max_prediction_error, compare_numeric_traces(outputs, expected['outputs']),
                               compare_numeric_traces(budget_outputs, budget_expected['outputs']))
    write_json(ROOT / 'REPLAY_DECISION_SEAL.json', dict(actual_decision_fingerprint=actual,
               outcomes_decoded=False, fresh_predictions=len(items), refitted_models=8), immutable=True)
    state['outcomes_allowed'] = True
    outcome_path = OUT / 'fresh/private_outcomes.json'
    outcome_relative = str(outcome_path.relative_to(ROOT)).replace('\\', '/')
    if sha(outcome_path) != packet['files'][outcome_relative]:
        raise ValueError('Outcome mapping was changed')
    outcomes = read_json(outcome_path)
    core = independent_core(outputs, items, outcomes)
    write_json(ROOT / 'recomputed_core.json', core, immutable=True)
    core_hash = sha(ROOT / 'recomputed_core.json')
    if core_hash != packet['expected_core_sha256']:
        raise ValueError('Independent core table differs from the prebound expected hash')
    budget_core = independent_core(budget_outputs, items, outcomes)
    write_json(ROOT / 'recomputed_budget_core.json', budget_core, immutable=True)
    budget_core_hash = sha(ROOT / 'recomputed_budget_core.json')
    if budget_core_hash != packet['expected_budget_core_sha256']:
        raise ValueError('Independent equal-budget core table differs')
    report = dict(passed=True, fresh_queries=len(items), methods=len(outputs),
                  actual_training_executed=True, refitted_models=8, actual_prediction_executed=True,
                  independent_rr_evaluation=True, original_summary_read=False,
                  max_prediction_error=max_prediction_error, decisions_fingerprint=actual,
                  core_table_sha256=core_hash, access_audit=state,
                  budget_core_sha256=budget_core_hash, exact_pair_budget_replayed=True,
                  additional_threeway_heads_refitted=4,
                  scope='cached upstream four-seed evidence -> refit all compared heads -> new decisions -> independent RR/cost table',
                  neural_upstream_regenerated_in_clean_directory=False,
                  upstream_regeneration_evidence='full fresh run plus exact first-eight-query generation replay')
    write_json(ROOT / 'WORKER_ACCEPTANCE.json', report, immutable=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
