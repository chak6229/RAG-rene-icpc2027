from __future__ import annotations

import json
import re
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from .common import OLD, OUT, PARENT, REVIEW, ROOT, frozen, now, put, read, rows, sha


def main():
    frozen()
    checked = Counter()
    def require(ok, kind, detail):
        if not ok:
            raise AssertionError(detail)
        checked[kind] += 1
    queries = list(rows(OUT / 'private/query_pools.jsonl'))
    sample = list(rows(OLD / 'private/sample_manifest.jsonl'))
    require([{k: q[k] for k in s} for q, s in zip(queries, sample)] == sample, 'sample_identity', 'Sample altered')
    require(len(queries) == 128, 'sample_identity', 'Sample size')
    original_labels = {}
    for r in rows(REVIEW / 'item_labels.jsonl'):
        if r['kind'] == 'code':
            for d in r['doc_ids']:
                original_labels[r['query_id'], d] = {v: r[v] for v in ('A', 'B')}
    ids = {q['query_id'] for q in queries}
    rankings = {(r['query_id'], r['position']): r['ranked_doc_ids'][:3]
                for r in rows(PARENT / 'rankings/CodeBERT.jsonl') if r['query_id'] in ids}
    raw = {int(v): k for k, v in read(ROOT / 'results/rene_cosqa_ssqr_audit_20261019/selected_source/coclr/data/search/code_idx_map.txt').items()}
    for q in queries:
        by_doc = {d: p for p in q['panels'] for d in p['doc_ids']}
        require(len(by_doc) == len(set(by_doc)), 'pool', 'Repeated doc')
        expected_pool = set()
        for name, ranking in q['rankings'].items():
            position = None if name == 'original' else int(name)
            require(rankings[q['query_id'], position] == ranking, 'raw_ranking', q['query_id'])
            expected_pool.update(ranking)
        require(set(by_doc) == expected_pool, 'pool', q['query_id'])
        for p in q['panels']:
            require(all(raw[d] == p['code'] for d in p['doc_ids']), 'code_identity', q['query_id'])
            old = [original_labels[q['query_id'], d] for d in p['doc_ids'] if (q['query_id'], d) in original_labels]
            if old:
                require(all(label == p['prior_labels'] for label in old), 'original_labels', q['query_id'])
    packets = read(OUT / 'private/reviewer_keys.json')
    for reviewer, packet in packets.items():
        path = OUT / f'reviewer_{reviewer}/代码相关性补充审阅.html'
        html = path.read_text(encoding='utf-8')
        payload = json.loads(re.search(r'<script id="packet" type="application/json">(.*?)</script>', html, re.S).group(1))
        require(payload['packet_id'] == packet['packet_id'], 'packet_identity', reviewer)
        require(set(payload) == {'schema_version', 'phase', 'reviewer', 'packet_id', 'cases'}, 'blinding', 'Metadata exposure')
        mappings = {c['case_id']: c for c in packet['cases']}
        new_panels = set()
        for c in payload['cases']:
            require(set(c) == {'case_id', 'original', 'documents'}, 'blinding', c['case_id'])
            mapping = mappings[c['case_id']]
            query = next(q for q in queries if q['query_id'] == mapping['query_id'])
            require(c['original'] == query['original'], 'packet_identity', c['case_id'])
            require({d['id'] for d in c['documents']} == set(mapping['document_map']), 'packet_coverage', c['case_id'])
            for d in c['documents']:
                require(set(d) == {'id', 'code'}, 'blinding', d['id'])
                panel_id = mapping['document_map'][d['id']]
                panel = next(p for p in query['panels'] if p['id'] == panel_id)
                require(panel['prior_labels'] is None and d['code'] == panel['code'], 'new_panel', panel_id)
                new_panels.add((query['query_id'], panel_id))
        expected = {(q['query_id'], p['id']) for q in queries for p in q['panels'] if p['prior_labels'] is None}
        require(new_panels == expected, 'packet_coverage', reviewer)
        with zipfile.ZipFile(OUT / f'相关性补判_审阅者{reviewer}.zip') as archive:
            require(len(archive.namelist()) == 2 and all('/' not in n for n in archive.namelist()), 'zip_scope', reviewer)
            require(archive.read('代码相关性补充审阅.html') == path.read_bytes(), 'zip_content', reviewer)
        script = re.search(r'<script>(.*?)</script>', html, re.S).group(1)
        with tempfile.TemporaryDirectory(prefix='rene-relevance-js-') as temp:
            js_path = Path(temp) / 'page.js'
            js_path.write_text(script, encoding='utf-8')
            process = subprocess.run(['node', '--check', str(js_path)], capture_output=True, text=True)
            require(process.returncode == 0, 'javascript_syntax', process.stderr)
    records = list(rows(OUT / 'preview/query_bounds.jsonl'))
    result = read(OUT / 'preview/RESULTS.json')
    require(len(records) == 6144 and len(result['summary']) == 48, 'analysis_coverage', 'Counts')
    pooled = defaultdict(list)
    for row in records:
        low, high = map(Fraction, row['expanded_delta_bounds_fraction'])
        delta = Fraction(row['original_qrels_delta_fraction'])
        increment = list(map(Fraction, row['change_from_original_delta_bounds_fraction']))
        require(low <= high and increment == [low-delta, high-delta], 'paired_identity', row['query_id'])
        if row['policy'] == 'SAFE_ENTR_best3_opportunity':
            require(low >= 0, 'safe_opportunity', row['query_id'])
        query = next(q for q in queries if q['query_id'] == row['query_id'])
        if row['policy'] in query['actions'] and query['actions'][row['policy']] == 'original':
            require(low == high == 0, 'fallback_identity', row['query_id'])
        if row['cutoff'] == 1:
            require(row['missing_panels'] == 0, 'top1_existing_coverage', row['query_id'])
        pooled[row['reviewer'], row['threshold'], row['cutoff'], row['policy']].append(row)
    for r in result['summary']:
        group = pooled[r['reviewer'], r['threshold'], r['cutoff'], r['policy']]
        for i in (0, 1):
            # Independent exact weighted aggregation of saved rational endpoints.
            total = sum(Fraction(q['expanded_delta_bounds_fraction'][i]) * Fraction(q['frame_n'], q['sample_n']) for q in group)
            require(abs(float(total/8530)-r['weighted_expanded_delta_identification_bounds'][i]) < 1e-14,
                    'weighted_replay', r['policy'])
    acceptance = {'passed': True, 'created_utc': now(), 'checks': dict(checked),
                  'total_checks': sum(checked.values()), 'protocol_sha256': sha(OUT / 'PROTOCOL.json'),
                  'preview_sha256': sha(OUT / 'preview/RESULTS.json'),
                  'sample_unchanged': True, 'old_labels_unchanged': True,
                  'new_human_labels_received': 0, 'reviewer_packets_sent': False,
                  'scope': 'existing-label sensitivity, fixed supplemental review materials and import/analysis tools'}
    put(OUT / 'VALIDATION.json', acceptance)
    print({'passed': True, 'checks': sum(checked.values())})


if __name__ == '__main__':
    main()
