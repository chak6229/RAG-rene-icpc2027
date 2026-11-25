from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction

from .common import OUT, POLICIES, ROOT, frozen, now, put, put_rows, read, rows, sha


def rr_at(ranking, relevant, cutoff):
    return next((Fraction(1, i) for i, doc in enumerate(ranking[:cutoff], 1) if doc in relevant), Fraction(0))


def query_bounds(query, reviewer, threshold, cutoff, supplement=None):
    target = set(query['target_doc_ids'])
    panel_by_id = {p['id']: p for p in query['panels']}
    doc_to_panel = {d: p['id'] for p in panel_by_id.values() for d in p['doc_ids']}
    lists = {k: v[:cutoff] for k, v in query['rankings'].items()}
    active = {doc_to_panel[d] for ranking in lists.values() for d in ranking}
    positive, unknown = set(target), []
    unresolved = Counter()
    for panel_id in active:
        p = panel_by_id[panel_id]
        if target & set(p['doc_ids']):
            positive.update(p['doc_ids'])
            continue
        label = p['prior_labels'][reviewer] if p['prior_labels'] is not None else (supplement or {}).get((query['query_id'], panel_id))
        if label is None:
            unknown.append(p['doc_ids'])
            unresolved['missing_panels'] += 1
        elif label['relevance'] == 'u':
            unknown.append(p['doc_ids'])
            unresolved['unknown_panels'] += 1
        elif int(label['relevance']) >= threshold:
            positive.update(p['doc_ids'])
    base = {k: rr_at(ranking, target, cutoff) for k, ranking in lists.items()}
    baseline = {policy: base[action]-base['original'] for policy, action in query['actions'].items()}
    baseline['SAFE_ENTR_best3_opportunity'] = max(base.values())-base['original']
    values = defaultdict(list)
    for mask in range(1 << len(unknown)):
        relevant = positive.copy()
        for i, docs in enumerate(unknown):
            if mask & (1 << i):
                relevant.update(docs)
        current = {k: rr_at(ranking, relevant, cutoff) for k, ranking in lists.items()}
        for policy, action in query['actions'].items():
            values[policy].append(current[action]-current['original'])
        values['SAFE_ENTR_best3_opportunity'].append(max(current.values())-current['original'])
    result = []
    for policy, outcomes in values.items():
        low, high = min(outcomes), max(outcomes)
        original = baseline[policy]
        result.append({'query_id': query['query_id'], 'group_id': query['group_id'], 'stratum': query['stratum'],
                       'frame_n': query['frame_n'], 'sample_n': query['sample_n'], 'weight': query['weight'],
                       'reviewer': reviewer, 'threshold': threshold, 'cutoff': cutoff, 'policy': policy,
                       'original_qrels_delta_fraction': str(original),
                       'expanded_delta_bounds_fraction': [str(low), str(high)],
                       'change_from_original_delta_bounds_fraction': [str(low-original), str(high-original)],
                       'expanded_delta_point_identified': low == high,
                       'unknown_text_variables': len(unknown), 'assignments_enumerated': 1 << len(unknown),
                       'missing_panels': unresolved['missing_panels'], 'unknown_panels': unresolved['unknown_panels']})
    return result


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['reviewer'], row['threshold'], row['cutoff'], row['policy']].append(row)
    result = []
    for (reviewer, threshold, cutoff, policy), items in groups.items():
        if len(items) != 128:
            raise ValueError('All queries must be retained')
        weights = [Fraction(r['frame_n'], r['sample_n']) for r in items]
        if sum(weights) != 8530:
            raise ValueError('Stratum weights must sum to H8530')
        def aggregate(vals, ws):
            return float(sum(v*w for v, w in zip(vals, ws))/sum(ws))
        baseline = [Fraction(r['original_qrels_delta_fraction']) for r in items]
        bounds = [[Fraction(v) for v in r['expanded_delta_bounds_fraction']] for r in items]
        increments = [[Fraction(v) for v in r['change_from_original_delta_bounds_fraction']] for r in items]
        result.append({'reviewer': reviewer, 'threshold': threshold, 'cutoff': cutoff, 'policy': policy,
                       'queries': len(items), 'sample_groups': len({r['group_id'] for r in items}),
                       'point_identified_queries': sum(r['expanded_delta_point_identified'] for r in items),
                       'weighted_original_qrels_delta': aggregate(baseline, weights),
                       'weighted_expanded_delta_identification_bounds': [aggregate([b[i] for b in bounds], weights) for i in (0, 1)],
                       'weighted_change_from_original_delta_bounds': [aggregate([b[i] for b in increments], weights) for i in (0, 1)],
                       'unweighted_original_qrels_delta': aggregate(baseline, [Fraction(1)]*len(items)),
                       'unweighted_expanded_delta_bounds': [aggregate([b[i] for b in bounds], [Fraction(1)]*len(items)) for i in (0, 1)],
                       'interval_type': 'sample identification range over shared unknown relevance, not a confidence interval'})
    return result


def report(result):
    lines = ['# Relevance sensitivity: preview of existing manual labels' if result['phase'] == 'existing_labels_preview' else '# Relevance sensitivity: post-retroactivity analysis', '',
             "This analysis uses the original 128 Query/120 group, the original eight-layer weights and fixed actions. Expand the pool to retain the original qrels text, as well as the same text in the pool, and add manual determination of the relevant code. Value is RR@ 1 or RR@ 3 Policy - original query error with original MRR@ 100 It's not a mix. Unknown panel and unknown text-by-text sharing status, with the following ranges being sample identifiable boundaries, not confidence compartments.", '',
             f"Merge Top-3{result['coverage']['total_pool_panels']}A code panel, already available{result['coverage']['inherited_panels']}I’ll need two men to do it.{result['coverage']['new_panels_per_reviewer']}individual. Current number of new manual tags:{result['new_labels_received']}。", '',
             "A reviewer's threshold qrels margin extension range of qrels is open to query.",
             '|---|---:|---:|---|---:|---|---|---:|']
    for r in result['summary']:
        if r['threshold'] != 2 or r['policy'] not in ('ENTR_top1', 'RIDGE_verified', 'RIDGE_risk'):
            continue
        def interval(pair):
            return '[' + ', '.join(f'{v:+.6f}' for v in pair) + ']'
        lines.append(f"|{r['reviewer']}|{r['threshold']}|{r['cutoff']}|{r['policy']}|{r['weighted_original_qrels_delta']:+.6f}|{interval(r['weighted_expanded_delta_identification_bounds'])}|{interval(r['weighted_change_from_original_delta_bounds'])}|{r['point_identified_queries']}/128|")
    lines += ['', 'All 48 Item grouping in RESULTS.json. Two reviewers and two thresholds were retained, respectively, without a consensus vote. Security best3 opportunities are used only for tag-aid references. Threshold 2, RIDGE_verified, RR @ 3 To the main diagnostic endpoint.', '',
              'The existing tab preview is ex post sensitive; The completion of the sentence does not amount to having received a judgement. The weighted community reflects the unknown relevance of the sample and does not include a sampling error or an evaluation error, which cannot be used to claim a significant gain overall or a percentage explanation for the cause.', '',
              'Run: python - m experiments.rene_relevance_sensitivity_20261021.collect  --reviewer  A  --input < Complete JSON >; B Same thing. Run after two imports analyze --phase  completed.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('preview', 'completed'), default='preview')
    args = parser.parse_args()
    frozen()
    phase = 'existing_labels_preview' if args.phase == 'preview' else 'completed_returns'
    destination = OUT / ('preview' if args.phase == 'preview' else 'completed')
    supplements = {'A': {}, 'B': {}}
    new_count = 0
    if args.phase == 'completed':
        for reviewer in ('A', 'B'):
            acceptance = read(OUT / f'returns/{reviewer}/ACCEPTANCE.json')
            label_path = OUT / f'returns/{reviewer}/labels.jsonl'
            if not acceptance['passed'] or sha(label_path) != acceptance['labels_sha256']:
                raise ValueError('Unaccepted return')
            supplements[reviewer] = {(r['query_id'], r['panel_id']): r['label'] for r in rows(label_path)}
            new_count += len(supplements[reviewer])
    queries = list(rows(OUT / 'private/query_pools.jsonl'))
    records = [row for query in queries for reviewer in ('A', 'B') for threshold in (2, 1) for cutoff in (1, 3)
               for row in query_bounds(query, reviewer, threshold, cutoff, supplements[reviewer])]
    result = {'created_utc': now(), 'phase': phase, 'new_labels_received': new_count,
              'coverage': read(OUT / 'BUILD_ACCEPTANCE.json'), 'summary': summarize(records),
              'query_records': len(records), 'all_128_queries_retained': True,
              'unknowns_treated_as_nonrelevant': False, 'new_confirmatory_effect_test': False,
              'old_qrels_and_review_labels_unchanged': True,
              'inputs': {'protocol_sha256': sha(OUT / 'PROTOCOL.json'),
                         'pools_sha256': sha(OUT / 'private/query_pools.jsonl')}}
    put(destination / 'RESULTS.json', result)
    put_rows(destination / 'query_bounds.jsonl', records)
    (destination / 'FINDINGS.md').write_text(report(result), encoding='utf-8')
    print({'phase': phase, 'records': len(records), 'summary_rows': len(result['summary']), 'new_labels': new_count})


if __name__ == '__main__':
    main()
