"""Pre-outcome supplement: identical actual CE pair budgets for both UQ arms."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.rene_repaired_confirmation_20261019.common import (
    DATA, OUT, ARMS, FAMILIES, read_json, write_json, load, save, sha, utc, observable, require_runtime,
)
from experiments.rene_repaired_confirmation_20261019.model import (
    predict_gain, prepare, choose_trace, train_risk, apply_risk,
)
from experiments.rene_repaired_confirmation_20261019.development import compact
from experiments.rene_repaired_confirmation_20261019.decide import deny_outcome_reads

FRESH = OUT / 'fresh'


def paired_sessions(models, items, family):
    sessions = {}
    eligible = {}
    for arm in ARMS:
        pred = predict_gain(models[arm]['gain'], items)
        sessions[arm] = [prepare(e, p, family, 0.) for e, p in zip(items, pred)]
        eligible[arm] = [i for i, s in enumerate(sessions[arm]) if s['verified'] and len(s['plan']['doc_ids']) == 8]
    quota = min(len(items) // 4, *(len(v) for v in eligible.values()))
    for arm in ARMS:
        chosen = set(sorted(eligible[arm], key=lambda i: (-sessions[arm][i]['priority'], items[i]['query_index']))[:quota])
        for i, s in enumerate(sessions[arm]):
            if i not in chosen:
                s.update(verified=False, plan=None)
    assert all(sum(s['verified'] for s in sessions[a]) == quota for a in ARMS)
    return sessions, quota


def fit_risks(models, dev, scores, family):
    role = models[ARMS[0]]['roles']['risk']
    assert all(models[a]['roles']['risk'] == role for a in ARMS)
    byid = {e['query_index']: (e, s) for e, s in zip(dev, scores)}
    selected = [byid[q][0] for q in role]
    current_scores = [byid[q][1] for q in role]
    views = [observable(e) for e in selected]
    sessions, quota = paired_sessions(models, views, family)
    return {a: train_risk(selected, [choose_trace(e, s, models[a]['update'], ce)
                                   for e, s, ce in zip(views, sessions[a], current_scores)], family) for a in ARMS}


def infer_pair(models, risks, items, scores, family):
    sessions, quota = paired_sessions(models, items, family)
    outputs = {}
    for arm in ARMS:
        verified = [choose_trace(e, s, models[arm]['update'], ce) for e, s, ce in zip(items, sessions[arm], scores)]
        gated = apply_risk(risks[arm], items, verified)
        outputs[f'{family}/{arm}/budget25_verified'] = [compact(t) for t in verified]
        outputs[f'{family}/{arm}/budget25_risk'] = [compact(t) for t in gated]
        assert sum(t['actual_pairs'] for t in verified) == 8 * quota
        assert sum(t['actual_pairs'] for t in gated) == 8 * quota
    return outputs, sessions, quota


def load_models(family):
    return {a: joblib.load(OUT / f'models/{family}_{a}.joblib') for a in ARMS}


def freeze():
    if (OUT / 'OUTCOME_OPENING.json').exists():
        raise RuntimeError('New comparison protocols are forbidden after outcome opening')
    path = FRESH / 'BUDGET_PROTOCOL.json'
    if path.exists():
        p = read_json(path)
        if sha(__file__) != p['source_sha256']:
            raise ValueError('Budget controller source changed')
        return p
    dev, scores = load(OUT / 'development/items.pkl'), load(OUT / 'development/scores.pkl')
    for family in FAMILIES:
        risks = fit_risks(load_models(family), dev, scores, family)
        joblib.dump(risks, OUT / f'models/budget_{family}_risk.joblib')
    p = dict(frozen_at_utc=utc(), main_protocol_sha256=sha(FRESH / 'PROTOCOL.json'),
             source_sha256=sha(__file__), outcomes_opened=False, primary_hypotheses_unchanged=True,
             role='predeclared descriptive cost-matched sensitivity analysis',
             candidate_pool='same actual four-seed candidates and base/semantic evidence',
             allocation='each arm selects its highest-priority floor(n/4) positive-mean full-eight-document plans; '
                        'shared quota reduced only if either arm has fewer observable eligible plans',
             exact_pair_cost=True, equal_token_count_claim=False,
             risk_fit='refit same threeway model on the original separate risk role using the quota allocation',
             models={str(p.relative_to(ROOT)): sha(p) for p in sorted((OUT / 'models').glob('budget_*_risk.joblib'))},
             thresholds=dict(base=.1, risk=.075), source='no fresh RR or target mapping used')
    write_json(path, p, immutable=True)
    return p


def decide():
    import torch
    from transformers import AutoTokenizer
    from experiments.action_witness_uq_20261014.gpu_data import CE, ce_tokens
    freeze()
    items = [observable(e) for e in load(FRESH / 'observable_items.pkl')]
    original_scores = load(FRESH / 'ce_scores.pkl')
    original = {e['query_index']: e for e in original_scores}
    models = {f: load_models(f) for f in FAMILIES}
    session_by_family = {f: paired_sessions(models[f], items, f)[0] for f in FAMILIES}
    needed = {e['query_index']: set() for e in items}
    for sessions in session_by_family.values():
        for arm in ARMS:
            for e, s in zip(items, sessions[arm]):
                if s['verified']:
                    needed[e['query_index']].update(s['plan']['doc_ids'])
    missing = [(q, int(d)) for q, docs in needed.items() for d in sorted(docs)
               if d not in set(map(int, original[q]['doc_ids']))]
    cache = FRESH / 'budget_ce_scores.pkl'
    if not cache.exists():
        queries = {e['query_index']: e for e in read_json(FRESH / 'queries.json')}
        corpora = {b['block']: load(ROOT / b['path']) for b in read_json(FRESH / 'PROTOCOL.json')['corpus']}
        extra = {}
        if missing:
            tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/codebert-base', local_files_only=True)
            model = CE().to('cuda').eval()
            model.load_state_dict(torch.load(DATA / 'ce_large_dense_4096.pt', map_location='cpu', weights_only=True))
            projection = torch.tensor(np.load(DATA / 'cls_projection.npy'), device='cuda')
            with torch.inference_mode():
                for begin in range(0, len(missing), 32):
                    batch = missing[begin:begin + 32]
                    tokens = ce_tokens(tokenizer, [queries[q]['query'] for q, d in batch],
                             [corpora[queries[q]['block']][d]['code_text'] for q, d in batch])
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits, cls = model(return_cls=True, **tokens)
                    projected = (cls.float() @ projection).cpu().numpy()
                    for i, pair in enumerate(batch):
                        extra[pair] = (float(logits[i].float()), projected[i])
            del model
            torch.cuda.empty_cache()
        merged = []
        for e in items:
            q = e['query_index']; old = original[q]
            values = {int(d): (float(old['logits'][i]), old['cls_projection'][i]) for i, d in enumerate(old['doc_ids'])}
            for d in needed[q]:
                if int(d) not in values:
                    values[int(d)] = extra[(q, int(d))]
            ids = sorted(values)
            merged.append(dict(query_index=q, doc_ids=np.array(ids, int), logits=np.array([values[d][0] for d in ids]),
                               cls_projection=np.array([values[d][1] for d in ids]).reshape(-1, 32)))
        save(cache, merged)
    scores = load(cache)
    outputs, quotas = {}, {}
    for family in FAMILIES:
        risks = joblib.load(OUT / f'models/budget_{family}_risk.joblib')
        current, sessions, quota = infer_pair(models[family], risks, items, scores, family)
        outputs.update(current); quotas[family] = quota
    destination = FRESH / 'budget_label_free_decisions.pkl'
    if not destination.exists():
        save(destination, dict(outputs=outputs))
    write_json(FRESH / 'BUDGET_DECISION_SEAL.json', dict(sealed_at_utc=utc(), actual_decisions=True,
               protocol_sha256=sha(FRESH / 'BUDGET_PROTOCOL.json'), decisions_sha256=sha(destination),
               scores_sha256=sha(cache), quotas=quotas, exact_pairs_each_arm={f: 8 * q for f, q in quotas.items()},
               extra_physical_pairs=len(missing), outcomes_opened=False), immutable=True)
    print('Equal-pair-budget decisions sealed: ' + str(quotas), flush=True)


def evaluate():
    from experiments.rene_repaired_confirmation_20261019.evaluate import summary, interval, core_table
    p = read_json(FRESH / 'BUDGET_PROTOCOL.json'); seal = read_json(FRESH / 'BUDGET_DECISION_SEAL.json')
    main_seal = read_json(FRESH / 'DECISION_SEAL.json')
    opening = read_json(OUT / 'OUTCOME_OPENING.json')
    if not (p['frozen_at_utc'] < seal['sealed_at_utc'] < opening['opened_at_utc'] and main_seal['sealed_at_utc'] < opening['opened_at_utc']):
        raise ValueError('Budget comparison was not sealed before outcome opening')
    if sha(FRESH / 'budget_label_free_decisions.pkl') != seal['decisions_sha256']:
        raise ValueError('Budget decisions changed')
    outcomes = read_json(FRESH / 'private_outcomes.json')
    targetby = {e['query_index']: e['target_doc_id'] for e in outcomes}
    items = load(FRESH / 'observable_items.pkl')
    originals = {e['query_index']: e['original_top100'] for e in items}
    saved = load(FRESH / 'budget_label_free_decisions.pkl')
    summaries, deltas = {}, {}
    for name, ts in saved['outputs'].items():
        summaries[name], deltas[name] = summary(ts, originals, targetby)
    contrasts = {}
    for family in FAMILIES:
        for stage in ('verified', 'risk'):
            a = deltas[f'{family}/seed_uq/budget25_{stage}']
            b = deltas[f'{family}/no_seed_uq/budget25_{stage}']
            contrasts[f'{family}/{stage}'] = dict(delta=float(np.mean(a - b)), ci95=interval(a - b), descriptive_only=True)
    write_json(OUT / 'BUDGET_RESULTS.json', dict(methods=summaries, uq_increment=contrasts,
               exact_ce_pair_budget=True, equal_token_count_claim=False, labels_used_to_tune=False), immutable=True)
    write_json(OUT / 'budget_core_table.json', core_table(summaries), immutable=True)
    print('Same-pair-budget contrasts: ' + str(contrasts), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=('freeze', 'decide', 'evaluate'))
    args = ap.parse_args()
    require_runtime(gpu=args.stage == 'decide')
    if args.stage != 'evaluate':
        deny_outcome_reads()
    with threadpool_limits(limits=2):
        {'freeze': freeze, 'decide': decide, 'evaluate': evaluate}[args.stage]()


if __name__ == '__main__':
    main()
