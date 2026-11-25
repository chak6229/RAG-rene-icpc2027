from __future__ import annotations

import csv
import json
import math
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction

from .common import (CORPUS, OUT, PARENT, POLICIES, RAW_MAP, RAW_SOURCE, ROOT, SOURCE, STRATA,
                     STRATUM_ZH, now, order_key, put, put_rows, read, record, require_frozen, rows,
                     rr, sample_frame, sha)
from .pages import make_pages


def freeze():
    if (OUT / 'PROTOCOL.json').exists():
        return require_frozen()
    inputs = [PARENT / name for name in ('FINAL_ACCEPTANCE.json', 'DESIGN.json', 'PRIMARY_RESULT.json',
              'DECISIONS_SEALED.json', 'public/H_queries.jsonl', 'public/H_groups.json', 'private/H_qrels.jsonl',
              'candidates.jsonl', 'selections.jsonl', 'ce_scores.jsonl')]
    for ret in ('CodeBERT', 'BM25'):
        inputs += [PARENT / f'{folder}/{ret}{suffix}' for folder, suffix in
                   (('decisions', '.jsonl'), ('rankings', '.jsonl'), ('metrics', '_per_query.jsonl'), ('metrics', '_full_H.json'))]
    inputs += [CORPUS, RAW_MAP, RAW_SOURCE]
    inputs += [ROOT / x['path'] for x in read(PARENT / 'DESIGN.json')['protected_files']]
    inputs += list(SOURCE.glob('*.py')) + [SOURCE / 'DESIGN.md']
    p = {'created_utc': now(), 'role': 'posthoc_consumed_H_mechanism_diagnostic_before_new_human_labels',
         'known_outcomes_before_design': True, 'external_preregistration': False,
         'reviewers': ['A', 'B'], 'human_labels_available': False, 'frame': 'CodeBERT full H8530/1963 groups',
         'per_stratum_n': 16, 'strata': list(STRATA), 'sample_algorithm': 'SHA256 rank within stratum; salt in bound common.py',
         'inference': 'finite fixed frame; stratified simultaneous hypergeometric intervals, no new confirmatory test',
         'bound_files': [record(p) for p in dict.fromkeys(inputs)],
         'old_primary_sha256': sha(PARENT / 'PRIMARY_RESULT.json'), 'risk_certified': False}
    put(OUT / 'PROTOCOL.json', p)
    return p


def stage_row(values, counts, n, total_oracle):
    if set(values) & set(counts):
        raise ValueError('count/magnitude column collision')
    result = {k: float(v/n) for k, v in values.items()}
    result.update(counts)
    result['positive_retention'] = float(values['positive']/total_oracle)
    return result


def analyze_retriever(ret, selected):
    qrels = defaultdict(set)
    for row in rows(PARENT / 'private/H_qrels.jsonl'):
        if row.get('relevance', 1) > 0:
            qrels[row['query_id']].add(row['doc_id'])
    reference = {r['query_id']: r for r in rows(PARENT / f'metrics/{ret}_per_query.jsonl')}
    rankings = {}
    for r in rows(PARENT / f'rankings/{ret}.jsonl'):
        q, pos = r['query_id'], r['position']
        if q in selected and (pos is None or pos in selected[q]):
            key = q, pos
            if key in rankings:
                raise ValueError('duplicate ranking')
            rankings[key] = r['ranked_doc_ids']
    decisions = defaultdict(dict)
    for r in rows(PARENT / f'decisions/{ret}.jsonl'):
        q, policy = r['query_id'], r['policy']
        if policy in POLICIES or policy == 'NoRewrite':
            if policy in decisions[q]:
                raise ValueError('duplicate policy')
            decisions[q][policy] = r
    if set(reference) != set(decisions) or set(reference) != set(selected) or len(reference) != 8530:
        raise ValueError('frame drift')
    frame, sums = [], {p: defaultdict(Fraction) for p in POLICIES}
    counts = {p: Counter() for p in POLICIES}
    total_oracle = Fraction(0)
    gain_score_error, score_n, fraction_checks = 0., 0, 0
    verification = Counter()
    gate = Counter()
    gate_sums = defaultdict(Fraction)
    max_error = 0.
    for q in sorted(reference):
        old = rankings[q, None]
        if len(old) != 100 or len(set(old)) != 100:
            raise ValueError('invalid original ranking')
        base = rr(old, qrels[q]); fraction_checks += 1
        deltas = {pos: rr(rankings[q, pos], qrels[q]) - base for pos in selected[q]}
        fraction_checks += len(deltas)
        oracle = max(Fraction(0), *deltas.values())
        total_oracle += oracle
        current = {'query_id': q, 'group_id': reference[q]['group_id'], 'NoRewrite': float(base),
                   'safe_ENTR_oracle_delta': float(oracle), 'candidate_deltas': {str(k): float(v) for k, v in deltas.items()},
                   'candidate_fractions': {str(k): str(v) for k, v in deltas.items()}, 'policies': {}}
        for policy in POLICIES:
            d = decisions[q][policy]
            ranking = d['ranking']
            if len(ranking) != 100 or len(set(ranking)) != 100 or d['accepted'] != (ranking != old):
                raise ValueError('adoption/ranking contract failed')
            if d['accepted']:
                if d['position'] not in deltas or ranking != rankings[q, d['position']]:
                    raise ValueError('non-pool action')
            elif ranking != old:
                raise ValueError('fallback differs')
            delta = rr(ranking, qrels[q]) - base; fraction_checks += 1
            expected = reference[q]['policies'][policy]
            max_error = max(max_error, abs(float(delta) - expected['delta']))
            if d['accepted'] != expected['adopted'] or max_error > 1e-12:
                raise ValueError('cached metric mismatch')
            positive, harm = max(delta, 0), max(-delta, 0)
            missed = oracle - positive
            if missed < 0 or oracle - delta != missed + harm:
                raise ValueError('opportunity conservation failed')
            for k, v in (('net', delta), ('positive', positive), ('harm', harm), ('missed', missed), ('regret', oracle-delta)):
                sums[policy][k] += v
            counts[policy].update(adopted=int(d['accepted']), improved=int(delta > 0), negative=int(delta < 0),
                                  zero_changed=int(d['accepted'] and delta == 0), opportunity_missed=int(missed > 0))
            current['policies'][policy] = {'delta': float(delta), 'delta_fraction': str(delta), 'adopted': d['accepted'],
                                          'position': d['position'], 'verified': d['verified'], 'pairs': d['actual_pairs']}
        pre, post = decisions[q]['RIDGE_verified'], decisions[q]['RIDGE_risk']
        gain = decisions[q]['RIDGE_gain']
        if len(gain['final_scores']) != len(selected[q]):
            raise ValueError('score/position order mismatch')
        for pos, pred in zip(selected[q], gain['final_scores']):
            gain_score_error += (pred - float(deltas[pos])) ** 2 / len(selected[q])
            score_n += 1
        current['gain_prediction_scores'] = gain['final_scores']
        current['verified_prediction_scores'] = pre['final_scores']
        current['verification_doc_ids'] = pre.get('verification_doc_ids', [])
        step = Fraction(current['policies']['RIDGE_verified']['delta_fraction']) - Fraction(current['policies']['RIDGE_gain']['delta_fraction'])
        verification.update(allocated=int(pre['verified']), changed_position=int(pre['position'] != gain['position']),
                            improved_RR=int(step > 0), worsened_RR=int(step < 0), unchanged_RR=int(step == 0),
                            positive_opportunity=int(oracle > 0), allocated_with_opportunity=int(pre['verified'] and oracle > 0))
        if pre['verified']:
            verification['known_target_in_evidence'] += int(bool(qrels[q] & set(pre.get('verification_doc_ids', []))))
        if pre['accepted'] and not post['accepted']:
            delta = Fraction(current['policies']['RIDGE_verified']['delta_fraction'])
            cat = 'positive' if delta > 0 else 'negative' if delta < 0 else 'zero'
            gate[cat] += 1
            gate_sums['positive_removed'] += max(delta, 0)
            gate_sums['loss_avoided'] += max(-delta, 0)
        elif post['accepted'] and post['ranking'] != pre['ranking']:
            raise ValueError('risk reselected')
        frame.append(current)
    n = len(frame)
    old_summary = read(PARENT / f'metrics/{ret}_full_H.json')
    summary = {'queries': n, 'groups': len({r['group_id'] for r in frame}), 'oracle': float(total_oracle/n),
               'stage_results': {}, 'verification_counts': dict(verification), 'gate_rejected_counts': dict(gate),
               'gate_removed_positive': float(gate_sums['positive_removed']/n), 'gate_avoided_loss': float(gate_sums['loss_avoided']/n),
               'candidate_gain_query_weighted_MSE': gain_score_error/n,
               'candidate_prediction_pairs': score_n, 'Fraction_RR_checks': fraction_checks, 'max_old_metric_difference': max_error}
    for policy in POLICIES:
        m = stage_row(sums[policy], counts[policy], n, total_oracle)
        summary['stage_results'][policy] = m
        for a, b in (('net', 'delta_MRR'), ('positive', 'positive_gain'), ('harm', 'loss_magnitude')):
            expected = old_summary['policy_results'][policy][b]
            expected = expected['mean'] if isinstance(expected, dict) else expected
            if abs(m[a]-expected) > 1e-12:
                raise ValueError('summary mismatch: ' + policy + '/' + a)
    if abs(summary['oracle'] - old_summary['safe_ENTR_oracle_delta']) > 1e-12:
        raise ValueError('oracle mismatch')
    put_rows(OUT / f'private/{ret}_stage_rows.jsonl', frame)
    return summary, frame, rankings, decisions, qrels


def create_packets(sample, selected, rankings, decisions, qrels):
    chosen = {r['query_id'] for r in sample}
    queries = {r['query_id']: r['query'] for r in rows(PARENT / 'public/H_queries.jsonl') if r['query_id'] in chosen}
    candidates = {(r['query_id'], r['position']): r['rewrite'] for r in rows(PARENT / 'candidates.jsonl') if r['query_id'] in chosen}
    raw = {int(v): k for k, v in read(RAW_MAP).items()}
    corpus = {r['doc_id']: r['code_text'] for r in rows(CORPUS)}
    token_identity = {}
    for r in read(RAW_SOURCE):
        text = r['code_tokens'] if isinstance(r['code_tokens'], str) else ' '.join(r['code_tokens'])
        token_identity[r['code']] = text.strip()
    for doc, text in corpus.items():
        if token_identity.get(raw[doc]) != text:
            raise ValueError('raw-source/tokenized-corpus identity mismatch')
    canonical = []
    for s in sample:
        q = s['query_id']; original = queries[q]
        by_text = defaultdict(list)
        for pos in selected[q]:
            by_text[candidates[q, pos]].append(pos)
        # Identity text remains as an explicitly reviewable rewrite; it is never silently dropped.
        rewrites = [{'text': text, 'positions': positions} for text, positions in by_text.items()]
        docs = list(dict.fromkeys(sorted(qrels[q]) + [rankings[q, None][0]] + [rankings[q, pos][0] for pos in selected[q]]))
        evidence = decisions[q]['RIDGE_verified'].get('verification_doc_ids', [])
        extra = next((d for d in evidence if d not in docs), None)
        if extra is not None:
            docs.append(extra)
        if len(docs) < 2:
            docs.append(rankings[q, None][1])
        if len(docs) > 6 or len(docs) != len(set(docs)):
            raise ValueError('document panel contract failed')
        canonical.append({**s, 'original': original, 'rewrites': rewrites, 'documents': docs,
                          'target_doc_ids': sorted(qrels[q]), 'code_texts_merged': len({raw[d] for d in docs}) != len(docs)})
    keys = {}
    for reviewer in ('A', 'B'):
        packet_id = order_key('packet', reviewer, sha(OUT / 'PROTOCOL.json'))[:24]
        phase1, phase2, private_map = [], [], []
        for item in canonical:
            q = item['query_id']
            case_id = order_key('case', reviewer, q)[:12]
            rw = sorted(item['rewrites'], key=lambda r: order_key('rewrites', reviewer, q, r['text']))
            pub_rw = [{'id': f'Q{i+1}', 'text': r['text']} for i, r in enumerate(rw)]
            doc_by_text = defaultdict(list)
            for d in item['documents']:
                doc_by_text[raw[d]].append(d)
            ds = sorted(doc_by_text, key=lambda text: order_key('docs', reviewer, q, text))
            pub_ds = [{'id': f'C{i+1}', 'code': text} for i, text in enumerate(ds)]
            phase1.append({'case_id': case_id, 'original': item['original'], 'rewrites': pub_rw})
            phase2.append({'case_id': case_id, 'original': item['original'], 'documents': pub_ds})
            private_map.append({**item, 'case_id': case_id,
                                'rewrite_map': {f'Q{i+1}': r for i, r in enumerate(rw)},
                                'document_map': {f'C{i+1}': doc_by_text[text] for i, text in enumerate(ds)}})
        phase1.sort(key=lambda r: order_key('case_order', reviewer, r['case_id']))
        phase2.sort(key=lambda r: order_key('code_order', reviewer, r['case_id']))
        folder = OUT / f'reviewer_{reviewer}'
        make_pages(folder, reviewer, packet_id, phase1, phase2)
        keys[reviewer] = {'packet_id': packet_id, 'cases': private_map}
        with zipfile.ZipFile(OUT / f'Reviewer{reviewer}.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(folder.iterdir()):
                z.write(p, p.name)
    put(OUT / 'private/reviewer_keys.json', keys)
    return {'unique_queries': len(canonical), 'unique_groups': len({r['group_id'] for r in sample}),
            'unique_rewrite_judgments_per_reviewer': sum(len(r['rewrites']) for r in canonical),
            'code_judgments_per_reviewer': sum(len(set(raw[d] for d in r['documents'])) for r in canonical),
            'raw_code_corpus_identity_verified': len(corpus)}


def reports(summaries, sample, packet_stats):
    strata = Counter(r['stratum'] for r in sample)
    lines = ['# Candidates to automatic benefits: stage diagnosis and double blinding', '',
             '** The crediting phase of the cache has been completed; separate manual tags for both persons have not yet returned and semantic/relevance mechanism conclusions to be determined.**', '',
             'This round uses the consumption group CoSQA H8530/1963; it is an ex post diagnostic and does not produce new conclusive effects.', '',
             '## Complete overall reconciliation of accounts', '',
             '|Retrieval/stage netting MRR',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    table = []
    for ret, summary in summaries.items():
        for policy, m in summary['stage_results'].items():
            lines.append(f"|{ret}/{policy}|{m['net']:+.8f}|{m['positive']:.8f}|{m['harm']:.8f}|{m['missed']:.8f}|{m['positive_retention']:.2%}|{m['adopted']}|{m['negative']}|")
            table.append({'retriever': ret, 'stage': policy, **m})
    lines += ['', 'Every query U = max (0 ENTR Pool candidate RR, D = P−H; U−D=(U−P)+H。 The opportunity leakage amount is U−P, which includes loss of experience in ranking and adoption and cannot be directly interpreted as the causal share of semantic drift or learning error.', '']
    for ret, summary in summaries.items():
        v = summary['verification_counts']; g = summary['gate_rejected_counts']
        lines += [f"- {ret}: Safe candidate opportunities{summary['oracle']:+.8f};validation of allocations{v['allocated']}Article, RR Improvement{v['improved_RR']}Change{v['worsened_RR']}(includes unapproved total denominator); evidence of entry of known targets in distribution{v['known_target_in_evidence']}Article.",
                  f"- {ret}Door denied: positive{g.get('positive',0)}Negative.{g.get('negative',0)}nil{g.get('zero',0)}; removing positive gains{summary['gate_removed_positive']:.8f}and avoidance of losses{summary['gate_avoided_loss']:.8f}. Target coverage is only an observational clue and does not prove sufficient or insufficient evidence."]
    lines += ['', '## Manual review preparation', '',
              'The whole length of the bar in a platinum layer, the fixed sample, the weight of each bar.', '|---|---:|---:|---:|']
    for code, label in zip(STRATA, STRATUM_ZH):
        selected = [r for r in sample if r['stratum'] == code]
        lines.append(f"|{label}|{selected[0]['frame_n']}|{strata[code]}|{selected[0]['weight']:.4f}|")
    lines += ['', f"Total{packet_stats['unique_queries']}Article queries/{packet_stats['unique_groups']}Associated groups;per capita{packet_stats['unique_rewrite_judgments_per_reviewer']}It’s different to rewrite the judgment,{packet_stats['code_judgments_per_reviewer']}A copy of the code is judged. A layered equivalent sample cannot be considered as an overall proportion of the sample nudity.", '',
              'Two examiners received the reviewer A.zip and the examiner B.zip. Finish the syntax page 01 first, save full export; Imports the semantic record again on the 02 code page. Only the respective packages will be handed over to the corresponding examiner; private Catalogues, current reports and statistical results are not submitted to reviewers.', '',
              'The code is restored in the official original format, and all 6267 entries correspond to the actual tokenized code identity; Artificial reading information is not the same as a model visible feature.', '',
              "## So far, we've been able to answer and wait for judgment.", '',
              'Phased accounting allows for the identification of actions on which candidates are not selected, as well as verification/retroactivity changes in positive gains and losses. Whether semantics are transposed, whether the target really meets the original demand, and whether the unmarked code is relevant requires two independent judgments on return. The incorrect statistical record of revenue projections still cannot be the only reason for separating training, features, models and indications.', '',
              'Recycling portal: python - m experiments.rene_mechanism_review_20261020 collect - a < A Complete Phase II JSON -- b < B Complete Phase II JSON > --out New Output Directory. Incomplete/strings/repeated items were verified and the statistics rejected. All discussions were held after the independent submission, with the original reservation.', '',
              'The protocol/sampling/anonymous mapping is retained; original confirmation and official five pointers are maintained.']
    (OUT / 'FINDINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (OUT / 'STAGES.csv').open('x', encoding='utf-8-sig', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)


def main():
    freeze(); require_frozen()
    reference = list(rows(PARENT / 'metrics/CodeBERT_per_query.jsonl'))
    sample = sample_frame(reference)
    put_rows(OUT / 'private/sample_manifest.jsonl', sample)
    selection = {}
    for r in rows(PARENT / 'selections.jsonl'):
        if r['policy'] == 'ENTR':
            if r['query_id'] in selection:
                raise ValueError('duplicate ENTR selection')
            selection[r['query_id']] = [s['position'] for s in sorted(r['selected'], key=lambda s: s['selected_rank'])]
    summaries, packet_stats = {}, None
    for ret in ('CodeBERT', 'BM25'):
        summary, frame, rankings, decisions, qrels = analyze_retriever(ret, selection)
        summaries[ret] = summary
        if ret == 'CodeBERT':
            packet_stats = create_packets(sample, selection, rankings, decisions, qrels)
        print({'retriever': ret, 'queries': len(frame), 'Fraction_RR_checks': summary['Fraction_RR_checks']}, flush=True)
    put(OUT / 'STAGE_RESULTS.json', summaries)
    put(OUT / 'PACKET_STATUS.json', {'created_utc': now(), **packet_stats, 'packets_built': True,
                                    'human_review_complete': False, 'human_labels': 0, 'causal_attribution_complete': False})
    reports(summaries, sample, packet_stats)
    print({'packets': packet_stats, 'human_labels': 0}, flush=True)


if __name__ == '__main__':
    main()
