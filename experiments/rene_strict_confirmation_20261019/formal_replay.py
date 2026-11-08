"""Execute the registered eight-seed formal flow again, not summary aggregation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from transformers import AutoTokenizer
from threadpoolctl import threadpool_limits

from experiments.rene_strict_confirmation_20261019.common import (
    DATA, OUT, FAMILIES, load, save, write_json, protected_hashes, rr, sha, require_runtime, observable,
)
from experiments.rene_strict_confirmation_20261019.model import risk_x
from experiments.verification_policy_registry_20261015.registry import VerificationPolicy, score_view
from experiments.risk_module_registry_20261016.registry import RiskModule


def main():
    require_runtime()
    protected = protected_hashes()
    items = load(DATA / 'action_items.pkl')
    evidence = load(ROOT / 'results/strong_flow_uq_20261014/evidence/relevance_evidence.pkl')
    scores = load(DATA / 'ce_large_dense_4096_scores_action.pkl')
    for e, z in zip(items, evidence):
        e['strong_evidence'] = z
    byid = {e['query_index']: e for e in items}
    scoreby = {e['query_index']: e for e in scores}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/codebert-base', local_files_only=True)
    codes = load(DATA / 'action_code_corpus.pkl')
    lengths = [len(x) for x in tokenizer([e['code_text'] for e in codes], add_special_tokens=False,
                                       truncation=True, max_length=160)['input_ids']]
    outputs, checks = {}, []
    with threadpool_limits(limits=2):
        for family in FAMILIES:
            collected = []
            for fold in range(4):
                source_path = ROOT / f'results/priority_verification_risk_20261015/{family}_fold{fold}.pkl'
                risk_path = ROOT / f'results/anchor_stage_risk_clean_20261016/{family}_fold{fold}.pkl'
                source = load(source_path); expected_risk = load(risk_path)
                expected = source['outputs']['held']['gain_priority']['traces']
                after_expected = expected_risk['outputs']['held']['threeway_anchor']['traces']
                ids = [t['query_index'] for t in expected]
                current = [byid[q] for q in ids]; ce = [scoreby[q] for q in ids]
                policy = VerificationPolicy('gain-priority-20261015', family, fold)
                sessions = policy.prepare(current)
                raw = policy.finalize(sessions, ce)
                traces = []
                error = 0.
                for e, session, t, sc, old in zip(current, sessions, raw, ce, expected):
                    for key in ('chosen', 'accepted', 'ranking', 'verified', 'plan'):
                        if t[key] != old[key]:
                            raise ValueError(f'Registered flow changed {family}/{fold}/{e["query_index"]}/{key}')
                    error = max(error, abs(t['mu'] - old['mu']), abs(t['harm'] - old['harm']))
                    tr = dict(t, query_index=e['query_index'], before=session['before'])
                    consumed = score_view(sc, session['plan']['doc_ids']) if session['verified'] else None
                    flow = policy.models[fold]['gain'].flow
                    tr['updated'] = [dict(u, arm='full', x=flow.witness_x(session['_item'], session['before'],
                                         family, consumed, session['plan'], u['candidate'])) for u in t['updated']]
                    docs = [] if tr['plan'] is None else tr['plan']['doc_ids']
                    qlength = len(tokenizer(e['query'], add_special_tokens=False, truncation=True, max_length=64)['input_ids'])
                    tr.update(actual_pairs=len(docs), billed_slots=8 if tr['verified'] else 0,
                              executed_ce_tokens=sum(4 + qlength + lengths[int(d)] for d in docs))
                    for key in ('actual_pairs', 'billed_slots', 'executed_ce_tokens'):
                        if tr[key] != old[key]:
                            raise ValueError('Registered cost replay mismatch: ' + key)
                    traces.append(tr)
                risk_model = RiskModule('anchor-stage-threeway-20261016', family, fold)
                selected = [i for i, t in enumerate(traces) if t['accepted']]
                if selected:
                    vals = risk_model.predict(dict(x=np.stack([risk_x(observable(current[i]), traces[i]) for i in selected]),
                             native=np.array([traces[i]['harm'] for i in selected]), feature_names=risk_model.feature_names))
                    for k, i in enumerate(selected):
                        t = traces[i]; t['post_harm'] = float(vals['p_harm'][k])
                        if t['post_harm'] > .075:
                            t.update(accepted=False, chosen=-1, ranking=tuple(map(int, current[i]['original_top100'])))
                for e, t, old in zip(current, traces, after_expected):
                    for key in ('chosen', 'accepted', 'ranking', 'actual_pairs', 'billed_slots', 'executed_ce_tokens'):
                        if t[key] != old[key]:
                            raise ValueError('Registered threeway replay mismatch: ' + key)
                    t['delta_rr'] = rr(t['ranking'], e['query_index']) - rr(e['original_top100'], e['query_index'])
                    np.testing.assert_allclose(t['delta_rr'], old['delta_rr'], atol=1e-12, rtol=0)
                if error > 1e-12:
                    raise ValueError('Registered numeric prediction changed: ' + str(error))
                checks.append(dict(family=family, fold=fold, queries=len(ids),
                                   max_prediction_error=error, rankings_exact=True, costs_exact=True,
                                   source_sha256=sha(source_path), risk_reference_sha256=sha(risk_path)))
                collected.extend(traces)
                print(f'Actual registered inference replay {family}/{fold}: {len(ids)} queries PASS', flush=True)
            outputs[family] = collected
    if protected_hashes() != protected:
        raise ValueError('Formal protected files changed')
    destination = OUT / 'formal_replay'
    save(destination / 'freshly_inferred_traces.pkl', outputs) if not (destination / 'freshly_inferred_traces.pkl').exists() else None
    report = dict(passed=True, actual_model_inference_executed=True, summary_json_used=False,
                  checks=checks, predictions=1024, test_used=False, historical_development=True,
                  registered_schema_seeds=8, formal_components_changed=False, protected_hashes=protected,
                  limitation='Re-executes registered prediction/update/risk on frozen upstream caches; does not recover unknown historical generator snapshots',
                  metrics={f: dict(delta_mrr=float(np.mean([t['delta_rr'] for t in ts])),
                                   accepted=sum(t['accepted'] for t in ts), harmful=sum(t['delta_rr'] < -1e-10 for t in ts))
                           for f, ts in outputs.items()})
    write_json(destination / 'REPLAY_AUDIT.json', report, immutable=True)
    print(report['metrics'], flush=True)


if __name__ == '__main__':
    main()
