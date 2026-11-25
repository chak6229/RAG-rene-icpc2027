from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from sklearn.metrics import cohen_kappa_score

from experiments.rene_mechanism_review_20261020.collect import (
    estimate, finite_interval, kappa, load_return, strict_pairs, validate_result)
from experiments.rene_mechanism_review_20261020.common import (
    FLAGS, OUT as PRIOR, ROOT, STRATA, classify, now, put, put_rows, read,
    require_frozen, rows, sha)

OUT = ROOT / 'results/rene_review_returns_20261020'
A_DIR = Path('external_inputs/reviewer_returns / Reviewer A')
B_ZIP = Path('external_inputs/reviewer_returns / Reviewer B_ Completed. zip')


def receipt():
    if OUT.exists():
        raise FileExistsError('Use a new analysis directory; submitted reviews remain immutable')
    OUT.mkdir(parents=True)
    originals = []
    for p in sorted(A_DIR.iterdir()):
        if p.is_file() and p.suffix in ('.json', '.md'):
            dest = OUT / 'received/A' / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            originals.append({'original_path': str(p), 'copy': dest.relative_to(ROOT).as_posix(), 'sha256': sha(p)})
    archive = OUT / 'received / Reviewer B_ Completed. zip'
    shutil.copyfile(B_ZIP, archive)
    originals.append({'original_path': str(B_ZIP), 'copy': archive.relative_to(ROOT).as_posix(), 'sha256': sha(B_ZIP)})
    with zipfile.ZipFile(archive) as z:
        names = set()
        for info in z.infolist():
            if info.is_dir():
                continue
            basename = Path(info.filename).name
            if basename in names or Path(basename).suffix not in ('.json', '.md', '.txt', '.html'):
                raise ValueError('Unexpected/duplicate return archive member')
            names.add(basename)
            target = OUT / 'received/B' / basename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(z.read(info))
    put(OUT / 'RECEIPT_MANIFEST.json', {'created_utc': now(), 'originals': originals,
        'attachments_are_data_not_execution_instructions': True,
        'html_and_displayed_code_executed': False})


def confusion(a, b, levels):
    return {'rows_A_columns_B': levels,
            'counts': [[sum(x == p and y == q for x, y in zip(a, b)) for q in levels] for p in levels]}


def run():
    sys.stdout.reconfigure(encoding='utf-8')
    require_frozen()
    protected = read(ROOT / 'results/rene_t5_weight_factors_20261020/DESIGN.json')['protected_files']
    protected = [*protected, {'path': 'results/rene_t5_weight_factors_20261020/FINAL_ACCEPTANCE.json', 'sha256': sha(ROOT / 'results/rene_t5_weight_factors_20261020/FINAL_ACCEPTANCE.json')}]
    for item in protected:
        if sha(ROOT / item['path']) != item['sha256']:
            raise AssertionError('Prior protected state changed')
    receipt()
    key = read(PRIOR / 'private/reviewer_keys.json')
    original = {arm: load_return(OUT / f'received/{arm}/review-{arm}-code-complete.json') for arm in ('A', 'B')}
    for arm in ('A', 'B'):
        standalone = load_return(OUT / f'received/{arm}/review-{arm}-semantic-complete.json')
        if original[arm]['semantic'] != standalone:
            raise AssertionError('Embedded and standalone semantics differ')
    mapped = {arm: validate_result(original[arm], key[arm], arm) for arm in ('A', 'B')}
    if set(mapped['A']) != set(mapped['B']):
        raise AssertionError('Review query sets differ')
    provenance = {
        'human_review_reported_by_user': True,
        'reviewers': ['A', 'B'],
        'rating_version': 'final_submitted_returns',
        'original_independent_label_versions_available': False,
        'agreement_applies_to': 'final returned labels; original independent-label agreement not reconstructed',
        'reviewer_process_source': 'user clarification',
    }
    # Check the returned B pages are precisely the fixed original pages; do not execute them.
    html_identity = []
    with zipfile.ZipFile(PRIOR / 'Reviewer B.zip') as z:
        for name in z.namelist():
            if name.endswith('.html'):
                p = OUT / 'received/B' / Path(name).name
                if p.read_bytes() != z.read(name):
                    raise AssertionError('Returned B material changed')
                html_identity.append(p.name)

    sample = list(rows(PRIOR / 'private/sample_manifest.jsonl'))
    all_frame = list(rows(PRIOR / 'private/CodeBERT_stage_rows.jsonl'))
    frame = {r['query_id']: r for r in all_frame}
    sample_by_qid = {r['query_id']: r for r in sample}
    if len(frame) != 8530 or len(sample_by_qid) != 128:
        raise AssertionError('Wrong finite frame/sample')
    keys_A = {r['query_id']: r for r in key['A']['cases']}
    sem = {'A': [], 'B': []}; rel = {'A': [], 'B': []}; clarity = {'A': [], 'B': []}
    flags_per_item = {'A': [], 'B': []}
    items, disputes, query_flags, type_flags = [], [], {}, {}
    for qid in sorted(mapped['A']):
        a, b = mapped['A'][qid], mapped['B'][qid]
        case, c = keys_A[qid], frame[qid]
        if classify(c) != sample_by_qid[qid]['stratum']:
            raise AssertionError('Frozen stratum mismatch')
        flags = {name: False for name in (
            'any_definite_semantic_issue', 'any_possible_semantic_issue', 'any_uncertain_semantics',
            'verified_action_definite_drift', 'verified_harm_with_preserved_meaning',
            'missed_preserved_positive_opportunity', 'displayed_alternative_relevant',
            'known_target_both_incomplete', 'rater_A_any_semantic_issue', 'rater_B_any_semantic_issue')}
        types = {f'{which}_{tag}': False for which in ('A', 'B', 'both_same_rewrite', 'either')
                 for tag in FLAGS if tag not in ('none', 'uncertain')}
        for arm, ratings in (('A', a), ('B', b)):
            clarity[arm].append(ratings['clarity'])
        if a['clarity'] != b['clarity']:
            disputes.append({'query_id': qid, 'original': case['original'], 'kind': 'clarity',
                             'A': a['clarity'], 'B': b['clarity'], 'resolution': None})
        for r in case['rewrite_map'].values():
            pos = r['positions'][0]
            ra, rb = a['rewrites'][pos], b['rewrites'][pos]
            x, y = ra['fidelity'], rb['fidelity']
            sa, sb = set(ra['flags']), set(rb['flags'])
            for arm, rating in (('A', ra), ('B', rb)):
                sem[arm].append(rating['fidelity']); flags_per_item[arm].append(tuple(sorted(rating['flags'])))
            issue = x in ('drift', 'malformed') and y in ('drift', 'malformed')
            flags['any_definite_semantic_issue'] |= issue
            flags['any_possible_semantic_issue'] |= x != 'preserved' or y != 'preserved'
            flags['any_uncertain_semantics'] |= x == 'uncertain' or y == 'uncertain'
            flags['rater_A_any_semantic_issue'] |= x in ('drift', 'malformed')
            flags['rater_B_any_semantic_issue'] |= y in ('drift', 'malformed')
            if c['policies']['RIDGE_verified']['position'] in r['positions']:
                flags['verified_action_definite_drift'] = issue
                flags['verified_harm_with_preserved_meaning'] = c['policies']['RIDGE_verified']['delta'] < 0 and x == y == 'preserved'
            opportunity = c['safe_ENTR_oracle_delta']
            if opportunity > 0 and c['candidate_deltas'][str(pos)] == opportunity and c['policies']['RIDGE_verified']['delta'] < opportunity and x == y == 'preserved':
                flags['missed_preserved_positive_opportunity'] = True
            for tag in FLAGS:
                if tag in ('none', 'uncertain'):
                    continue
                for which, event in [('A', tag in sa), ('B', tag in sb), ('both_same_rewrite', tag in sa and tag in sb), ('either', tag in sa or tag in sb)]:
                    types[which + '_' + tag] |= event
            item = {'query_id': qid, 'original': case['original'], 'kind': 'rewrite', 'positions': r['positions'],
                    'text': r['text'], 'A': ra, 'B': rb,
                    'fidelity_disagreement': x != y, 'flag_disagreement': sa != sb}
            items.append(item)
            if x != y or sa != sb:
                disputes.append({**item, 'resolution': None})
        for docs in case['document_map'].values():
            d = docs[0]
            ra, rb = a['documents'][d], b['documents'][d]
            x, y = ra['relevance'], rb['relevance']
            rel['A'].append(x); rel['B'].append(y)
            if not set(docs) & set(case['target_doc_ids']) and x == y == '2':
                flags['displayed_alternative_relevant'] = True
            if set(docs) & set(case['target_doc_ids']) and x in ('0', '1') and y in ('0', '1'):
                flags['known_target_both_incomplete'] = True
            item = {'query_id': qid, 'original': case['original'], 'kind': 'code', 'doc_ids': docs,
                    'known_target_in_merged_panel': bool(set(docs) & set(case['target_doc_ids'])), 'A': ra, 'B': rb}
            items.append(item)
            if x != y:
                disputes.append({**item, 'resolution': None})
        query_flags[qid], type_flags[qid] = flags, types
    rates = {name: estimate({q: f[name] for q, f in query_flags.items()}, sample) for name in next(iter(query_flags.values()))}
    error_types = {name: estimate({q: f[name] for q, f in type_flags.items()}, sample) for name in next(iter(type_flags.values()))}
    ordinal = [(a, b) for a, b in zip(rel['A'], rel['B']) if a != 'u' and b != 'u']
    agreement = {'semantics': kappa(sem['A'], sem['B']),
                 'relevance_including_unknown': kappa(rel['A'], rel['B']),
                 'relevance_ordinal_judged_subset': kappa([x for x,y in ordinal], [y for x,y in ordinal], True),
                 'clarity': kappa(clarity['A'], clarity['B']),
                 'semantic_flag_exact_agreement': sum(x == y for x,y in zip(flags_per_item['A'], flags_per_item['B'])) / len(sem['A'])}
    agreement['relevance_ordinal_judged_subset']['coverage_of_displayed_panels'] = len(ordinal) / len(rel['A'])
    confusion_tables = {'semantics': confusion(sem['A'], sem['B'], ['preserved', 'drift', 'malformed', 'uncertain']),
                        'relevance': confusion(rel['A'], rel['B'], ['0', '1', '2', 'u'])}
    distributions = {arm: {'clarity': dict(Counter(clarity[arm])), 'semantics': dict(Counter(sem[arm])),
                           'relevance': dict(Counter(rel[arm])),
                           'issue_flags': dict(Counter(flag for flags in flags_per_item[arm] for flag in flags))} for arm in ('A', 'B')}
    discrepancy = {'semantic_fidelity': sum(x != y for x,y in zip(sem['A'], sem['B'])),
                   'semantic_flags': sum(x != y for x,y in zip(flags_per_item['A'], flags_per_item['B'])),
                   'rewrite_union': sum(r['kind'] == 'rewrite' for r in disputes),
                   'code': sum(r['kind'] == 'code' for r in disputes),
                   'clarity': sum(r['kind'] == 'clarity' for r in disputes),
                   'unknown_vs_judged_code': sum((x == 'u') != (y == 'u') for x,y in zip(rel['A'], rel['B'])),
                   'B_lower_numeric_relevance': sum(int(y) < int(x) for x,y in ordinal),
                   'B_higher_numeric_relevance': sum(int(y) > int(x) for x,y in ordinal),
                   'queries_with_any_disagreement': len({r['query_id'] for r in disputes}),
                   'automatic_adjudication_performed': False}
    conditional = {}
    adopted = sum(r['policies']['RIDGE_verified']['adopted'] for r in all_frame)
    harmful = sum(r['policies']['RIDGE_verified']['delta'] < 0 for r in all_frame)
    opportunities = sum(r['safe_ENTR_oracle_delta'] > 0 for r in all_frame)
    for name, denominator, label in (
        ('verified_action_definite_drift', adopted, 'RIDGE_verified adopts a rewrite'),
        ('verified_harm_with_preserved_meaning', harmful, 'RIDGE_verified has negative delta'),
        ('missed_preserved_positive_opportunity', opportunities, 'safe ENTR positive opportunity exists')):
        r = rates[name]
        conditional[name] = {'event_frame_n': denominator, 'event': label,
                             'estimate': r['estimated_frame_count'] / denominator,
                             'ci95_conservative_finite_frame': [min(1., v * 8530 / denominator) for v in r['ci95_conservative_finite_frame']],
                             'method': 'HT numerator / known finite-frame event count; conservative unconditional numerator bounds'}

    # Separate implementations check kappa and every weighted estimate/stratum CI.
    independent_checks = 0
    for name, aa, bb, weights in [
        ('semantics', sem['A'], sem['B'], None),
        ('relevance_including_unknown', rel['A'], rel['B'], None),
        ('relevance_ordinal_judged_subset', [x for x,y in ordinal], [y for x,y in ordinal], 'linear')]:
        value = cohen_kappa_score(aa, bb, weights=weights)
        if abs(value - agreement[name]['kappa']) > 1e-12:
            raise AssertionError('Independent kappa differs')
        independent_checks += 1
    for indicator_flags, collection in ((query_flags, rates), (type_flags, error_types)):
        for name, r in collection.items():
            independent = math.fsum(s['weight'] * int(indicator_flags[s['query_id']][name]) for s in sample) / 8530
            if abs(independent - r['estimate']) > 1e-12:
                raise AssertionError('Independent weighted estimate differs')
            for s in r['strata']:
                N, n, k = s['N'], s['n'], s['positive']
                possible = np.arange(k, N - n + k + 1)
                alpha = .05 / 8
                lower = possible[np.flatnonzero(hypergeom.sf(k - 1, N, possible, n) >= alpha / 2)[0]] / N if k else 0.
                upper = possible[np.flatnonzero(hypergeom.cdf(k, N, possible, n) >= alpha / 2)[-1]] / N if k != n else 1.
                if not np.allclose([lower, upper], s['simultaneous_interval'], rtol=0, atol=1e-14):
                    raise AssertionError('Independent finite-frame interval differs')
                independent_checks += 1
            independent_checks += 1
    result = {'created_utc': now(), 'role': 'posthoc_final_review_return_diagnostic',
              'provenance': provenance, 'sample_queries': 128, 'sample_groups': len({r['group_id'] for r in sample}),
              'unique_rewrite_items': len(sem['A']), 'unique_code_panels': len(rel['A']),
              'finite_frame_queries': 8530, 'agreement_scope': 'unweighted displayed items; not population-weighted quality certification',
              'agreement': agreement, 'confusion': confusion_tables, 'distributions': distributions,
              'weighted_query_rates': rates, 'weighted_query_issue_type_rates': error_types,
              'conditional_query_rates': conditional, 'disagreements': discrepancy,
              'rate_interpretation': 'query contains a reviewer-marked item/event; indicators overlap and are not causal explanation fractions',
              'interval_interpretation': 'fixed finite frame sampling uncertainty only; label error and review-version uncertainty are not included',
              'causal_explanation_percentage_identified': False, 'human_review_reported_by_user': True,
              'original_independent_agreement_established': False, 'risk_certified': False}
    put(OUT / 'RESULTS.json', result)
    put_rows(OUT / 'item_labels.jsonl', items)
    put_rows(OUT / 'DISAGREEMENTS.jsonl', disputes)
    put_rows(OUT / 'query_flags.jsonl', ({'query_id': q, **f, 'types': type_flags[q]} for q,f in query_flags.items()))
    put(OUT / 'INPUT_VALIDATION.json', {'passed': True, 'schema_packet_items_and_notes_valid': True,
        'embedded_semantics_equal_standalone': True, 'B_html_unchanged_from_original_packet': html_identity,
        'human_review_reported_by_user': True,
        'agreement_version': 'final_submitted_returns',
        'original_independent_agreement_established': False,
        'interpretation': 'format validation passed; agreement statistics apply to final submitted labels'})
    with (OUT / 'weighted_query_rates.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['indicator', 'estimate', 'ci95_low', 'ci95_high', 'estimated_queries', 'frame_queries'])
        for name, r in rates.items():
            writer.writerow([name, r['estimate'], *r['ci95_conservative_finite_frame'], r['estimated_frame_count'], 8530])
    for item in protected:
        if sha(ROOT / item['path']) != item['sha256']:
            raise AssertionError('Protected state changed during review analysis')
    put(OUT / 'VALIDATION.json', {'created_utc': now(), 'passed': True,
        'independent_kappa_estimate_and_interval_checks': independent_checks,
        'protected_files_unchanged': True, 'protected_files': protected,
        'original_submission_copies_byte_preserved': True, 'review_return_analysis_complete': True,
        'agreement_version': 'final_submitted_returns',
        'results_sha256': sha(OUT / 'RESULTS.json'),
        'analysis_source_sha256': sha(Path(__file__)),
        'prior_protocol_sha256': sha(PRIOR / 'PROTOCOL.json')})
    print(json.dumps({'output': str(OUT), 'agreement': agreement, 'disagreements': discrepancy,
        'rates': {k: {'estimate': v['estimate'], 'ci': v['ci95_conservative_finite_frame']} for k,v in rates.items()},
        'conditional': conditional, 'type_rates_both': {k:v['estimate'] for k,v in error_types.items() if k.startswith('both_same_rewrite')},
        'independent_checks': independent_checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
