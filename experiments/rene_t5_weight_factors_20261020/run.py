from __future__ import annotations

import argparse
import math
import platform
from collections import Counter, defaultdict

from .common import (BRIDGE, DOMAINS, FAMILY_SIZE, OLD, OUT, PACKAGE, REPETITIONS, ROOT,
                     SEED, VIEWS, WEIGHTS, now, paths, primary_names, read, record,
                     rows, sha, text_id, verify, write, writel)
from .metrics import factorial, order_candidates, paired_difference, summarize


def prepare(samples):
    cache_rows = rows(OLD / 'candidate_RR.jsonl')
    cache = {(r['domain'], r['rewriter'], r['query_id'], r['position']): r for r in cache_rows}
    if len(cache) != len(cache_rows):
        raise AssertionError('Repeated cache key')
    prior_rows = rows(OLD / 'per_query.jsonl')
    prior = {(r['domain'], r['rewriter'], r['query_id']): r for r in prior_rows}
    if len(prior) != len(prior_rows):
        raise AssertionError('Repeated prior query key')
    prepared, coverage = {}, {}
    consumed = set()
    for domain in DOMAINS:
        p = paths(domain)
        ids = samples[domain]['query_ids']
        queries = {r['query_id']: r['query'] for r in rows(p['queries'])}
        if len(ids) != len(set(ids)) or len(ids) != 512:
            raise AssertionError('Original 512-query scope required')
        for arm in WEIGHTS:
            pools = defaultdict(list)
            for c in rows(p[f'{arm}_candidates']):
                if c['query_id'] in set(ids):
                    pools[c['query_id']].append(c)
            if set(pools) != set(ids):
                raise AssertionError('Missing matched query')
            arm_rows = []
            for qid in ids:
                pool = sorted(pools[qid], key=lambda c: c['position'])
                n = len(queries[qid].split())
                if [c['position'] for c in pool] != list(range(1, n + 1)):
                    raise AssertionError('Incomplete original position pool')
                saved = prior[domain, arm, qid]
                if (saved['positions'], saved['group']) != (n, samples[domain]['groups'][qid]):
                    raise AssertionError('Prior count/group mismatch')
                scores = {}
                for c in pool:
                    key = domain, arm, qid, c['position']
                    r = cache[key]
                    if r['text_id'] != text_id(c['rewrite']) or r['full_vocab_entropy'] != c['full_vocab_entropy']:
                        raise AssertionError('Cache identity/entropy mismatch')
                    if not math.isfinite(r['rr']) or not 0 <= r['rr'] <= 1:
                        raise AssertionError('Invalid cached RR')
                    consumed.add(key)
                    scores[c['position']] = r['rr']
                orders = {policy: order_candidates(pool, policy) for policy in ('ENTR', 'PROB')}
                arm_rows.append({'query_id': qid, 'pool': pool, 'scores': scores, 'orders': orders, 'prior': saved})
            prepared[domain, arm] = arm_rows
            coverage[domain + '/' + arm] = {'queries': len(ids), 'groups': len(set(samples[domain]['groups'].values())),
                'candidate_positions': sum(len(r['pool']) for r in arm_rows),
                'duplicate_text_extra_positions': sum(len(r['pool']) - len({c['rewrite'] for c in r['pool']}) for r in arm_rows),
                'pool_size_counts': dict(sorted(Counter(len(r['pool']) for r in arm_rows).items())),
                'shorter_than_three_queries': sum(len(r['pool']) < 3 for r in arm_rows)}
        for local, base in zip(prepared[domain, 'local'], prepared[domain, 't5base']):
            if local['query_id'] != base['query_id'] or local['prior']['RR']['NoRewrite'] != base['prior']['RR']['NoRewrite']:
                raise AssertionError('Unmatched query/control')
    if consumed != set(cache) or len(prior) != 2048:
        raise AssertionError('Cache scope differs from matched design')
    return prepared, coverage


def freeze():
    if (OUT / 'DESIGN.json').exists():
        raise FileExistsError('Design already frozen')
    old_acceptance = read(OLD / 'FINAL_ACCEPTANCE.json')
    if not old_acceptance['passed'] or old_acceptance['protocol_sha256'] != sha(OLD / 'PROTOCOL.json') or old_acceptance['results_sha256'] != sha(OLD / 'RESULTS.json') or old_acceptance['audit_sha256'] != sha(OLD / 'AUDIT.json'):
        raise AssertionError('Previous evidence completion not accepted')
    samples = read(OLD / 'PROTOCOL.json')['samples']
    _, coverage = prepare(samples)  # Contract checks only; no new factorial effect calculation.
    protected = {r['path'] for r in read(ROOT / 'results/rene_third_retriever_20261020/DESIGN.json')['protected_files']}
    protected.update(r['path'] for r in read(OLD / 'PROTOCOL.json')['protected_files'])
    protected.update(['results/rene_envelope_completion_20261020/FINAL_ACCEPTANCE.json',
                      'results/rene_evidence_release_20261020/FINAL_ACCEPTANCE.json',
                      'results/rene_third_retriever_20261020/FINAL_ACCEPTANCE.json',
                      'results/rene_third_retriever_20261020/DELIVERY_ACCEPTANCE.json'])
    bound = list(PACKAGE.glob('*.py')) + [OUT / 'TEST_RESULTS.json']
    bound += [OLD / f for f in ('PROTOCOL.json', 'RANKINGS_SEALED.json', 'RESULTS.json', 'per_query.jsonl', 'candidate_RR.jsonl', 'AUDIT.json')]
    for domain in DOMAINS:
        bound += list(paths(domain).values())
    write(OUT / 'DESIGN.json', {
        'created_utc': now(), 'status': 'FROZEN_BEFORE_NEW_FACTORIAL_EFFECTS',
        'role': 'post_hoc_matched_weight_sensitivity_on_consumed_samples',
        'prior_Top1_best3_and_local_full_sample_factors_known': True,
        'new_factorial_effects_computed_before_freeze': False,
        'new_confirmation': False, 'external_preregistration': False,
        'samples': samples, 'coverage': coverage,
        'retriever': 'frozen local SSQR CodeBERT pooler, raw dot product, RR@100; reuse accepted rankings',
        'weights': {'local': 'models/t5-cqc-codenn-1m/best', 't5base': 'models/t5-base'},
        'selectors': {'ENTR': 'ascending full_vocab_entropy, then original integer position',
                      'PROB': 'descending label_mean_log_probability, then original integer position',
                      'RAND_exact': 'exact uniform subset without replacement over all original positions; retain duplicate-text mass'},
        'candidate_counts': [1, 2, 3], 'short_pool': 'k clipped to pool size; no original-query padding/fallback',
        'aggregations': {'mean': 'arithmetic mean of selected RR; expected utility if one selected candidate is adopted uniformly',
                         'max': 'maximum selected RR; label-assisted opportunity, not executable selection/fusion'},
        'estimands': {'S': 'selector_RR(k,a) - exact_uniform_RR(k,a)',
                      'B': 'S(1,mean) - S(3,mean)', 'G': 'S(3,mean) - S(3,max)',
                      'J': 'B+G = S(1,mean) - S(3,max)',
                      'paired': 'base minus local, same query/group/original control, per-query difference before statistics'},
        'primary_family': {'size': FAMILY_SIZE, 'per_domain_names': primary_names(),
                           'definition': '2 domains x [2 weights and paired weight difference] x 2 selectors x B/G',
                           'alpha_family': .05, 'correction': 'two-sided Bonferroni across all 24 contrasts',
                           'direction': 'both corrected intervals on same side of zero and at least 20 nonzero groups'},
        'statistics': {'estimand': 'query-weighted mean; whole connected/repository groups resampled',
                       'cluster_t': 'df = groups-1; finite-G cluster sandwich for query-weighted mean',
                       'bootstrap': 'percentile, shared group draws across all metrics and both weights in a domain',
                       'repetitions': REPETITIONS, 'seed': SEED,
                       'secondary': 'full k1/2/3 mean/max curves, J, ENTR-PROB, and interactions: pointwise95 only'},
        'interpretation': 'specified arithmetic path, not unique causal attribution; full weight-condition comparison, not isolated CQC-training effect',
        'no_new_training': True, 'no_model_inference': True, 'no_new_retrieval': True,
        'no_H_evaluation': True, 'not_cross_rewriter_family': True, 'no_equivalence_test': True,
        'risk_certified': False, 'environment': {'python': platform.python_version()},
        'bound_files': [record(f) for f in sorted(set(bound))],
        'protected_files': [record(ROOT / f) for f in sorted(protected)],
    })
    print({'stage': 'frozen', 'primary_comparisons': FAMILY_SIZE, 'coverage': coverage}, flush=True)


def evaluate():
    design = verify()
    if (OUT / 'RESULTS.json').exists():
        raise FileExistsError('Results already exist')
    prepared, coverage = prepare(design['samples'])
    exported, paired_export, result_domains = [], [], {}
    anchor_error, anchor_fields = 0., 0
    for domain in DOMAINS:
        by_view = {}
        for arm in WEIGHTS:
            query_rows = []
            for r in prepared[domain, arm]:
                saved = r['prior']
                values, contrasts = factorial(r['scores'], r['orders'], saved['RR']['NoRewrite'])
                anchors = {'NoRewrite': values['NoRewrite'], 'ENTR_top1': values['ENTR_k1_mean'],
                           'ENTR_best3': values['ENTR_k3_max'], 'PROB_top1': values['PROB_k1_mean'],
                           'PROB_best3': values['PROB_k3_max'], 'Uniform_single_exact': values['RAND_exact_k1_mean'],
                           'Uniform_best3_exact': values['RAND_exact_k3_max'], 'Oracle_all': max(r['scores'].values())}
                for key, value in anchors.items():
                    error = abs(value - saved['RR'][key])
                    if error > 1e-12:
                        raise AssertionError('Prior anchor not reproduced: ' + key)
                    anchor_error = max(anchor_error, error)
                    anchor_fields += 1
                if r['orders']['ENTR'][:3] != saved['selected_positions']:
                    raise AssertionError('Old ENTR selection not reproduced')
                query_rows.append({'domain': domain, 'view': arm, 'query_id': r['query_id'], 'group': saved['group'],
                                   'positions': saved['positions'], 'orders': r['orders'], 'values': values, 'contrasts': contrasts})
            by_view[arm] = query_rows
            exported.extend(query_rows)
        by_view['base_minus_local'] = [paired_difference(a, b) for a, b in zip(by_view['local'], by_view['t5base'])]
        paired_export.extend(by_view['base_minus_local'])
        names = [f'{view}::{section}::{key}' for view in VIEWS for section in ('values', 'contrasts') for key in by_view[view][0][section]]
        matrix = [[by_view[view][i][section][key] for view, section, key in (name.split('::') for name in names)] for i in range(512)]
        groups = [r['group'] for r in by_view['local']]
        summary = summarize(matrix, names, groups, set(primary_names()))
        views = {view: {section: {key: summary[f'{view}::{section}::{key}'] for key in by_view[view][0][section]} for section in ('values', 'contrasts')} for view in VIEWS}
        result_domains[domain] = {'queries': len(groups), 'groups': len(set(groups)), 'views': views}
        print({'stage': 'evaluated', 'domain': domain, 'primary': {name: (summary[name]['mean'], summary[name]['direction']) for name in primary_names()}}, flush=True)
    writel(OUT / 'per_query.jsonl', exported)
    writel(OUT / 'paired_per_query.jsonl', paired_export)
    write(OUT / 'RESULTS.json', {'created_utc': now(), 'design_sha256': sha(OUT / 'DESIGN.json'),
          'role': design['role'], 'new_confirmation': False, 'risk_certified': False,
          'domains': result_domains, 'coverage': coverage,
          'old_anchor_fields_reproduced': anchor_fields, 'old_anchor_max_error': anchor_error,
          'no_model_inference': True, 'no_new_retrieval': True, 'no_new_training': True})
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'evaluate'])
    args = parser.parse_args()
    {'freeze': freeze, 'evaluate': evaluate}[args.mode]()
