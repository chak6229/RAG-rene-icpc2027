"""Exact/normalized exposure audit. Never loads any new model effect."""
from __future__ import annotations

import collections
import gc
import time
from pathlib import Path

from .common import ROOT, OUT, SOURCES, UnionFind, digest, fingerprints, now, read, record, repo, rows, sha, token_text, write, writel

CSN = ROOT / 'imports/python/final/jsonl'
KINDS = ('q_norm', 'q_lex', 'code_raw', 'code_ws', 'code_tokens', 'url', 'function', 'repo')


def source_specs():
    return [
        ('retriever_training', ROOT / 'data/codexglue/training/train.jsonl'),
        ('consumed_AdvTest_test', ROOT / 'data/codexglue/advtest/test.jsonl'),
        ('used_AdvTest_valid', ROOT / 'data/codexglue/advtest/valid.jsonl'),
    ]


def build_reference():
    seen = {k: set() for k in KINDS}
    receipts = []
    counts = {}

    def add(name, path, iterator):
        n = 0
        for r in iterator:
            n += 1
            f = fingerprints(r)
            for k in KINDS:
                seen[k].update(f[k])
        receipts.append(record(path))
        counts[name] = n
        print({'reference': name, 'rows': n}, flush=True)

    for name, path in source_specs():
        add(name, path, rows(path))
    path = ROOT / 'imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-all.json'
    add('consumed_CoSQA_all_annotations', path, iter(read(path)))
    path = ROOT / 'results/rene_cosqa_ssqr_audit_20261019/QUERY_GROUP_INVENTORY.jsonl'
    for r in rows(path):
        seen['repo'].update(s.casefold() for s in r['source_repositories'])
    receipts.append(record(path))
    for path in sorted((ROOT / 'results').glob('rene_*/fresh/queries.json')):
        if any(s in path.parent.parent.name for s in ('strict_confirmation', 'repaired_confirmation', 'uq_independent_proof')):
            add(path.parent.parent.name, path, iter(read(path)))
    # All CSN train docstrings were prepared as an early CQC corpus. Its exact historical
    # use is not fully reconstructible, so exclude it rather than invent fresh status.
    for path in sorted((CSN / 'train').glob('*.gz')):
        add('early_CSN_CQC_prepared_' + path.name, path, rows(path))
    # CODEnn has descriptions, not recoverable repository/function IDs.
    for name in ('train.txt', 'valid.txt'):
        path = ROOT / 'data/codenn_cqc' / name
        add('CQC_' + name, path, ({'query': line.strip()} for line in path.open(encoding='utf-8')))
    return seen, receipts, counts


def connected(records):
    uf = UnionFind()
    for r in records:
        anchor = r['unit_id']
        uf.find(anchor)
        for k in ('repo', 'q_lex', 'code_ws', 'code_tokens'):
            for v in r['fingerprints'][k]:
                uf.join(anchor, k + ':' + v)
    for r in records:
        r['group_id'] = digest('new-factor-group/' + uf.find(r['unit_id']))[:24]
    return records


def audit_csn(seen):
    per_split = {}
    all_records = []
    receipts = []
    for split in ('valid', 'test'):
        counts = collections.Counter()
        eligible = []
        for path in sorted((CSN / split).glob('*.gz')):
            receipts.append(record(path))
            for index, r in enumerate(rows(path)):
                counts['rows'] += 1
                f = fingerprints(r)
                hits = {k: bool(f[k] & seen[k]) for k in KINDS}
                for k, hit in hits.items():
                    counts['overlap_' + k] += hit
                query = token_text(r['docstring_tokens'])
                malformed = len(query.split()) < 3 or '<extra_id_' in query or not repo(r)
                counts['malformed_or_short'] += malformed
                identity_hit = any(hits[k] for k in KINDS if k != 'repo')
                counts['no_exact_identity_or_query_overlap'] += not identity_hit and not malformed
                if not any(hits.values()) and not malformed:
                    eligible.append({'unit_id': f'csn-{split}-{index}', 'split': split, 'source_index': index,
                                     'repo': repo(r), 'url': r['url'], 'query': query,
                                     'fingerprints': {k: sorted(v) for k, v in f.items()}})
        counts['strict_rows'] = len(eligible)
        counts['strict_repositories'] = len({r['repo'] for r in eligible})
        per_split[split] = dict(counts)
        all_records += eligible
    connected(all_records)
    sizes = collections.Counter(r['group_id'] for r in all_records)
    writel(OUT / 'CSN_ELIGIBLE_METADATA.jsonl', all_records)
    return {'splits': per_split, 'combined_rows': len(all_records), 'combined_groups': len(sizes),
            'group_sizes': dict(sorted(sizes.items())), 'files': receipts,
            'role': 'same_source_CSN_remainder; not a new web-query source',
            'scope': 'exact/normalized query, code, identity, and recovered-repository disjointness; no foundation-pretraining claim'}


def main():
    started = time.perf_counter()
    if (OUT / 'LOCAL_ELIGIBILITY.json').exists():
        raise FileExistsError('Local eligibility already sealed')
    seen, inputs, counts = build_reference()
    result = audit_csn(seen)
    # The index is only for subsequent source screening; it contains hashes and IDs.
    import pickle
    path = OUT / 'reference_fingerprints.pkl'
    with path.open('xb') as h:
        pickle.dump(seen, h, protocol=5)
    write(OUT / 'LOCAL_ELIGIBILITY.json', {'created_utc': now(), 'new_method_outcomes_read': False,
          'references': inputs, 'reference_rows': counts, 'reference_distinct': {k: len(v) for k, v in seen.items()},
          'reference_index': record(path), 'CSN': result, 'seconds': time.perf_counter() - started,
          'early_CSN_train_role': 'conservatively excluded prepared historical CQC material; not asserted to have trained current T5',
          'unrecoverable': ['foundation-model pretraining exposure', 'semantic clones', 'unlogged historic processes'],
          'models_retrained': False, 'new_confirmation': False})
    print({'CSN_rows': result['combined_rows'], 'CSN_groups': result['combined_groups']}, flush=True)


if __name__ == '__main__':
    main()
