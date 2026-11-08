from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from experiments.rene_budget_aggregation_20261020.metrics import contrasts, exact_random_max, factorial_values

from .common import COMPONENTS, NAME, OUT, ROOT, now, progress, read, record, require, rows, sha, verify_design, verify_seal, write, writel
from .prepare import reference_rows
from .statistics import summarize


def qrels(cfg):
    output = defaultdict(set)
    for r in rows(ROOT / cfg['qrels']):
        output[r['query_id']].update(r.get('relevant_doc_ids', [r.get('doc_id')]))
    return dict(output)


def rr(ranked, relevant):
    return next((1. / (i + 1) for i, doc in enumerate(ranked[:100]) if doc in relevant), 0.)


def score_names():
    names = list(COMPONENTS)
    names += [f'{p}_{c}' for p in ('PROB', 'ENTR_minus_PROB') for c in ('B_budget_mean_1_to_3', 'G_aggregation_at3_mean_to_max')]
    names += [f'{p}_J_exact' for p in ('ENTR', 'PROB')]
    names += [f'S_{p}_k{k}_{a}' for p in ('ENTR', 'PROB') for k in (1, 2, 3) for a in ('mean', 'max')]
    names += [f'delta_{p}_k{k}_{a}' for p in ('ENTR', 'PROB', 'RAND_exact') for k, a in ((1, 'mean'), (3, 'mean'), (3, 'max'))]
    names += ['ENTR_minus_PROB_k1_mean', 'ENTR_minus_PROB_k3_max']
    return names


def raw_names():
    return ['NoRewrite'] + [f'{p}_k{k}_{a}' for p in ('ENTR', 'PROB', 'RAND_exact') for k, a in ((1, 'mean'), (3, 'mean'), (3, 'max'))]


def main():
    design = verify_design(private=True)
    verify_seal('RANKINGS_SEALED.json')
    require(read(OUT / 'PREOPEN_AUDIT.json')['passed'], 'Preopening audit failed')
    require(not (OUT / 'RESULTS.json').exists(), 'New model effects already opened')
    require(not (OUT / 'OUTCOME_OPENING.json').exists(), 'Interrupted outcome opening requires explicit recovery audit')
    write(OUT / 'OUTCOME_OPENING.json', {'created_utc': now(), 'design_sha256': sha(OUT / 'DESIGN.json'),
          'ranking_seal_sha256': sha(OUT / 'RANKINGS_SEALED.json'), 'preopen_audit': record(OUT / 'PREOPEN_AUDIT.json'),
          'prior_dataset_and_two_retriever_results_known': True, 'new_model_effects_opened_once': True,
          'new_confirmation': False, 'no_effect_dependent_tuning_or_expansion': True})
    started = time.perf_counter()
    ledger, domains = [], {}
    anchor_errors = []
    for domain, cfg in design['samples'].items():
        queries = list(rows(ROOT / cfg['queries']))
        actions = {r['query_id']: r for r in rows(ROOT / cfg['actions'])}
        relevant = qrels(cfg)
        rankings = {r['text']: r['ranked_doc_ids'] for r in rows(OUT / f'rankings/{domain}.jsonl')}
        candidates = defaultdict(list)
        for r in rows(ROOT / cfg['candidates']):
            candidates[r['query_id']].append(r)
        new = {}
        for q in queries:
            qid = q['query_id']
            scores = {r['position']: rr(rankings[r['rewrite']], relevant[qid]) for r in candidates[qid]}
            base = rr(rankings[q['query']], relevant[qid])
            values = factorial_values(scores, actions[qid]['orders'], base, exact_random_max(list(scores.values()), 3))
            fx = contrasts(values)
            values = {k: v for k, v in values.items() if 'RAND20' not in k}
            fx = {k: v for k, v in fx.items() if 'RAND20' not in k}
            r = {'domain': domain, 'retriever': NAME, 'query_id': qid, 'group': actions[qid]['group'],
                 'positions': len(scores), 'position_rr': {str(k): v for k, v in scores.items()},
                 'values': values, 'contrasts': fx}
            ledger.append(r)
            new[qid] = r
        del rankings
        cfg_paths = {k: ROOT / v if k in ('queries', 'reference_rows', 'reference_results') else v for k, v in cfg.items()}
        ref = reference_rows(domain, cfg_paths)
        combined = {NAME: new, **ref}
        # Validate each accepted reference contrast against its original aggregate.
        accepted = read(ROOT / cfg['reference_results'])
        for ret in ref:
            if domain == 'cosqa':
                accepted_summaries = next(v for v in accepted['views'] if v['domain'] == 'cosqa' and v['retriever'] == ret)['primary_decomposition']
            else:
                accepted_summaries = accepted['views'][ret]['summaries']
            for c in COMPONENTS:
                error = abs(float(np.mean([ref[ret][q['query_id']]['contrasts'][c] for q in queries])) - accepted_summaries[c]['mean'])
                require(error < 1e-13, 'Reference aggregate changed')
                anchor_errors.append(error)
        all_names, family_names = [], []
        for ret in combined:
            all_names += [f'{ret}::{c}' for c in score_names()]
            all_names += [f'{ret}::raw_{c}' for c in raw_names()]
        family_names += [f'{NAME}::{c}' for c in COMPONENTS]
        for ret in ref:
            pair = f'{NAME}_minus_{ret}'
            all_names += [f'{pair}::{c}' for c in COMPONENTS]
            family_names += [f'{pair}::{c}' for c in COMPONENTS]
        matrix = []
        for q in queries:
            qid = q['query_id']
            row = []
            for ret, current in combined.items():
                row += [current[qid]['contrasts'][c] for c in score_names()]
                row += [current[qid]['values'][c] for c in raw_names()]
            for ret in ref:
                row += [new[qid]['contrasts'][c] - ref[ret][qid]['contrasts'][c] for c in COMPONENTS]
            matrix.append(row)
        groups = [actions[q['query_id']]['group'] for q in queries]
        progress('statistics', domain=domain, queries=len(queries), contrasts=len(all_names), primary_family_total=12)
        estimates = summarize(matrix, all_names, groups, family_names=family_names)
        domains[domain] = {'queries': len(queries), 'groups': len(set(groups)), 'corpus_documents': cfg['documents_expected'],
            'views': {ret: {'summaries': {k.split('::', 1)[1]: v for k, v in estimates.items() if k.startswith(ret + '::')},
                            'new_predictions': ret == NAME} for ret in combined},
            'paired_differences': {ret: {k.split('::', 1)[1]: v for k, v in estimates.items() if k.startswith(f'{NAME}_minus_{ret}::')} for ret in ref},
            'primary_directions': {k: estimates[k]['direction'] for k in family_names}}
    writel(OUT / 'per_query.jsonl', ledger)
    write(OUT / 'RESULTS.json', {'created_utc': now(), 'design_sha256': sha(OUT / 'DESIGN.json'),
          'opening': record(OUT / 'OUTCOME_OPENING.json'), 'role': design['role'], 'domains': domains,
          'reference_anchor_max_error': max(anchor_errors), 'new_confirmation': False,
          'external_preregistration': False, 'risk_certified': False, 'seconds': time.perf_counter() - started})
    print({'results_written': True, 'primary_directions': {d: v['primary_directions'] for d, v in domains.items()}}, flush=True)


if __name__ == '__main__':
    main()
