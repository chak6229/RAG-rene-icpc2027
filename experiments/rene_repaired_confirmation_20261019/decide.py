"""Plan, score only observable witnesses, then seal actual label-free decisions."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from crsu.core import rrf
from experiments.rene_repaired_confirmation_20261019.common import (
    DATA, OUT, ARMS, FAMILIES, read_json, write_json, load, save, sha, utc,
    require_runtime, check_hashes, reject_gold, observable,
)
from experiments.rene_repaired_confirmation_20261019.model import predict_gain, prepare, predict_pipeline
from experiments.rene_repaired_confirmation_20261019.development import compact
from experiments.rene_repaired_confirmation_20261019 import weak_control

FRESH = OUT / 'fresh'


def deny_outcome_reads():
    """An audit hook makes accidental outcome/dataset reads fail during inference."""
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
            return
        path = str(args[0]).replace('\\', '/').casefold()
        if 'private_outcomes' in path or '/data/codexglue/' in path or path.endswith('/results.json'):
            raise PermissionError('Outcome/source/summary access is forbidden during decisions: ' + path)
    sys.addaudithook(hook)


def inputs():
    p = read_json(FRESH / 'PROTOCOL.json')
    check_hashes(p['binding_hashes'])
    check_hashes(p['protected_hashes'])
    check_hashes(read_json(FRESH / 'EVIDENCE_COMPLETE.json')['outputs'])
    items = load(FRESH / 'observable_items.pkl')
    reject_gold(items)
    # Source repository and URLs never enter prediction.
    return p, [observable(e) for e in items]


def plan():
    p, items = inputs()
    path = FRESH / 'sessions.pkl'
    if path.exists():
        seal = read_json(FRESH / 'PLAN_SEAL.json')
        if sha(path) != seal['sessions_sha256']:
            raise ValueError('Plan seal mismatch')
        return load(path)
    result = {}
    for family in FAMILIES:
        for arm in ARMS:
            model = joblib.load(OUT / f'models/{family}_{arm}.joblib')
            predictions = predict_gain(model['gain'], items)
            result[f'{family}/{arm}'] = [prepare(e, pred, family, model['cut']) for e, pred in zip(items, predictions)]
    reject_gold(result)
    save(path, result)
    write_json(FRESH / 'PLAN_SEAL.json', dict(protocol_sha256=sha(FRESH / 'PROTOCOL.json'),
               sessions_sha256=sha(path), observations_sha256=sha(FRESH / 'observable_items.pkl'),
               sealed_at_utc=utc(), actual_plans=True, outcomes_decoded=False), immutable=True)
    return result


def score():
    import torch
    from transformers import AutoTokenizer
    from experiments.action_witness_uq_20261014.gpu_data import CE, ce_tokens
    require_runtime(gpu=True)
    sessions = plan()
    path = FRESH / 'ce_scores.pkl'
    if path.exists():
        if sha(path) != read_json(FRESH / 'CE_COMPLETE.json')['scores_sha256']:
            raise ValueError('CE cache changed')
        return
    queries = read_json(FRESH / 'queries.json')
    queryby = {e['query_index']: e for e in queries}
    corpora = {b['block']: load(ROOT / b['path']) for b in read_json(FRESH / 'PROTOCOL.json')['corpus']}
    docs_by_query = {q['query_index']: set() for q in queries}
    for values in sessions.values():
        for q, session in zip(queries, values):
            if session['verified']:
                docs_by_query[q['query_index']].update(map(int, session['plan']['doc_ids']))
    pairs = [(q['query_index'], d) for q in queries for d in sorted(docs_by_query[q['query_index']])]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/codebert-base', local_files_only=True)
    model = CE().to('cuda').eval()
    model.load_state_dict(torch.load(DATA / 'ce_large_dense_4096.pt', map_location='cpu', weights_only=True))
    projection = torch.tensor(np.load(DATA / 'cls_projection.npy'), device='cuda')
    logits, projections, actual_tokens = [], [], 0
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(pairs), 32):
            current = pairs[begin:begin + 32]
            texts = [queryby[q]['query'] for q, _ in current]
            codes = [corpora[queryby[q]['block']][d]['code_text'] for q, d in current]
            tokens = ce_tokens(tokenizer, texts, codes)
            actual_tokens += int(tokens['attention_mask'].sum().item())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values, cls = model(return_cls=True, **tokens)
            logits.extend(values.float().cpu().tolist())
            projections.extend((cls.float() @ projection).cpu().numpy())
            if (begin // 32 + 1) % 256 == 0:
                print(f'CE observable planned pairs {min(begin + 32, len(pairs))}/{len(pairs)}', flush=True)
    bypair = {key: i for i, key in enumerate(pairs)}
    scores = []
    for q in queries:
        docs = sorted(docs_by_query[q['query_index']])
        ix = [bypair[(q['query_index'], d)] for d in docs]
        scores.append(dict(query_index=q['query_index'], doc_ids=np.array(docs, int),
                           logits=np.array([logits[i] for i in ix]),
                           cls_projection=np.array([projections[i] for i in ix]).reshape(-1, 32)))
    reject_gold(scores)
    save(path, scores)
    write_json(FRESH / 'CE_COMPLETE.json', dict(completed=True, physical_unique_pairs=len(pairs),
               physical_executed_tokens=actual_tokens, scores_sha256=sha(path),
               plan_seal_sha256=sha(FRESH / 'PLAN_SEAL.json'), labels_opened=False,
               elapsed_seconds=time.perf_counter() - start), immutable=True)
    print(f'Actual CE complete: {len(pairs)} query/document pairs', flush=True)
    del model
    torch.cuda.empty_cache()


def fixed50(items, predictions, family):
    proposals = []
    for e, p in zip(items, predictions):
        if len(p['mu']):
            j = int(np.argmax(p['mu']))
            proposals.append((float(p['mu'][j]), e['query_index'], j))
    count = min(len(proposals), int(np.floor(.5 * len(items) + .5)))
    selected = {q: j for _, q, j in sorted(proposals, key=lambda v: (-v[0], v[1]))[:count]}
    output = []
    for e in items:
        j = selected.get(e['query_index'], -1)
        ranking = e['original_top100'] if j < 0 else (e['candidate_top100'][j] if family == 'rewrite' else
                                                       rrf(e['original_top100'], e['candidate_top100'][j]))
        accepted = tuple(ranking) != tuple(e['original_top100'])
        output.append(dict(query_index=e['query_index'], chosen=j if accepted else -1, accepted=accepted,
                           ranking=tuple(map(int, ranking)), verified=False, actual_pairs=0,
                           billed_slots=0, executed_ce_tokens=0, nominal_selected=bool(j >= 0)))
    return output


def baseline(items, family, seed101):
    output = []
    for e in items:
        options = [j for j in range(len(e['x'])) if not seed101 or 101 in e['seed_appearances'][j]]
        j = min(options, key=lambda j: (-e['candidates'][j]['information_gain'], j)) if options else -1
        ranking = e['original_top100'] if j < 0 else (e['candidate_top100'][j] if family == 'rewrite' else
                                                       rrf(e['original_top100'], e['candidate_top100'][j]))
        accepted = tuple(ranking) != tuple(e['original_top100'])
        output.append(dict(query_index=e['query_index'], chosen=j if accepted else -1,
                           accepted=accepted, ranking=tuple(map(int, ranking)), verified=False,
                           actual_pairs=0, billed_slots=0, executed_ce_tokens=0))
    return output


def make_decisions():
    protocol, items = inputs()
    sessions = load(FRESH / 'sessions.pkl')
    if sha(FRESH / 'sessions.pkl') != read_json(FRESH / 'PLAN_SEAL.json')['sessions_sha256']:
        raise ValueError('Sealed plans changed')
    scores = load(FRESH / 'ce_scores.pkl')
    if sha(FRESH / 'ce_scores.pkl') != read_json(FRESH / 'CE_COMPLETE.json')['scores_sha256']:
        raise ValueError('CE cache changed')
    outputs, raw_predictions = {}, {}
    for family in FAMILIES:
        for arm in ARMS:
            model = joblib.load(OUT / f'models/{family}_{arm}.joblib')
            values, predictions, current_sessions = predict_pipeline(model, items, scores)
            previous = sessions[f'{family}/{arm}']
            for a, b in zip(previous, current_sessions):
                if a['plan'] != b['plan'] or a['verified'] != b['verified']:
                    raise ValueError('Final decisions changed the sealed CE plans')
                np.testing.assert_array_equal(a['before']['mu'], b['before']['mu'])
            for stage, traces in values.items():
                outputs[f'{family}/{arm}/{stage}'] = [compact(t) for t in traces]
            outputs[f'{family}/{arm}/fixed50'] = fixed50(items, predictions, family)
            raw_predictions[f'{family}/{arm}'] = [dict(query_index=e['query_index'], mu=p['mu'], harm=p['harm'],
                         member_mu=p['member_mu'], member_harm=p['member_harm']) for e, p in zip(items, predictions)]
        for dimension in (34, 46):
            model = joblib.load(OUT / f'diagnostic_models/{family}_base{dimension}.joblib')
            outputs[f'{family}/base{dimension}/mean'] = [compact(t) for t in weak_control.predict(model, items)]
        outputs[f'{family}/seed101_IG'] = baseline(items, family, True)
        outputs[f'{family}/multiseed_IG'] = baseline(items, family, False)
    result = dict(outputs=outputs, raw_predictions=raw_predictions,
                  protocol_sha256=sha(FRESH / 'PROTOCOL.json'))
    reject_gold(result)
    return result


def decide():
    destination = FRESH / 'label_free_decisions.pkl'
    if destination.exists():
        seal = read_json(FRESH / 'DECISION_SEAL.json')
        if sha(destination) != seal['decisions_sha256']:
            raise ValueError('Decision seal changed')
        print('Verified existing actual decision seal; no relabeling', flush=True)
        return
    result = make_decisions()
    save(destination, result)
    write_json(FRESH / 'DECISION_SEAL.json', dict(sealed_at_utc=utc(), actual_decisions=True,
               protocol_sha256=sha(FRESH / 'PROTOCOL.json'), decisions_sha256=sha(destination),
               observations_sha256=sha(FRESH / 'observable_items.pkl'), scores_sha256=sha(FRESH / 'ce_scores.pkl'),
               plan_seal_sha256=sha(FRESH / 'PLAN_SEAL.json'), outcomes_decoded=False,
               test_used=False, methods=len(result['outputs']),
               accepted={k: sum(t['accepted'] for t in v) for k, v in result['outputs'].items()}), immutable=True)
    print('Actual predictions, selected candidates and rankings sealed before outcome decoding', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('plan', 'score', 'decide', 'all'), required=True)
    args = ap.parse_args()
    require_runtime(gpu=args.stage in ('score', 'all'))
    deny_outcome_reads()
    with threadpool_limits(limits=2):
        if args.stage == 'plan':
            plan()
        elif args.stage == 'score':
            score()
        elif args.stage == 'decide':
            decide()
        else:
            plan(); score(); decide()


if __name__ == '__main__':
    main()
