from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .common import OUT, frozen, now, put, put_rows, read, rows, sha


def validate_return(payload, reviewer, packet):
    if payload.get('schema_version') != 1 or payload.get('phase') != 'relevance_supplement':
        raise ValueError('Unexpected return schema')
    if payload.get('reviewer') != reviewer or payload.get('packet_id') != packet['packet_id']:
        raise ValueError('Wrong reviewer or packet')
    if payload.get('complete') is not True or payload.get('independence_confirmed') is not True:
        raise ValueError('Complete independent return required')
    cases = {c['case_id']: c for c in packet['cases']}
    answers = payload.get('answers', {})
    if set(cases) != set(answers):
        raise ValueError('Case coverage differs')
    normalized = []
    for case_id, case in cases.items():
        docs = answers[case_id].get('documents', {})
        if set(docs) != set(case['document_map']):
            raise ValueError('Code-panel coverage differs')
        for document_id, panel_id in case['document_map'].items():
            label = docs[document_id]
            if label.get('relevance') not in ('0', '1', '2', 'u') or len(str(label.get('note', '')).strip()) < 6:
                raise ValueError('Invalid judgment or missing reason')
            normalized.append({'query_id': case['query_id'], 'panel_id': panel_id,
                               'reviewer': reviewer, 'label': {'relevance': label['relevance'], 'note': label['note']}})
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviewer', required=True, choices=('A', 'B'))
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    frozen()
    packet = read(OUT / 'private/reviewer_keys.json')[args.reviewer]
    normalized = validate_return(read(args.input), args.reviewer, packet)
    pools = list(rows(OUT / 'private/query_pools.jsonl'))
    missing = {(q['query_id'], p['id']) for q in pools for p in q['panels'] if p['prior_labels'] is None}
    if {(r['query_id'], r['panel_id']) for r in normalized} != missing:
        raise ValueError('Return would replace existing labels or omit new panels')
    destination = OUT / f'returns/{args.reviewer}'
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.input, destination / 'original_return.json')
    put_rows(destination / 'labels.jsonl', normalized)
    put(destination / 'ACCEPTANCE.json', {'passed': True, 'created_utc': now(),
                                         'reviewer': args.reviewer, 'labels': len(normalized),
                                         'input_sha256': sha(args.input),
                                         'labels_sha256': sha(destination / 'labels.jsonl'),
                                         'existing_labels_overwritten': False})
    print({'reviewer': args.reviewer, 'new_labels_accepted': len(normalized)})


if __name__ == '__main__':
    main()
