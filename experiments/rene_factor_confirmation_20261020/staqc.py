"""Pinned author StaQC source parsing; restricted data-only pickle loading."""
from __future__ import annotations

import ast
import collections
import io
import pickle
import re
import tokenize

from .common import ROOT, OUT, SOURCES, UnionFind, digest, fingerprints, lexical, norm, now, read, record, write, writel
from .eligibility import KINDS


class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'Executable pickle global rejected: {module}.{name}')


def load_data(path):
    with path.open('rb') as h:
        return DataUnpickler(h, encoding='latin1').load()


def code_tokens(code):
    try:
        v = list(tokenize.generate_tokens(io.StringIO(code).readline))
        excluded = {tokenize.COMMENT, tokenize.ENCODING, tokenize.INDENT, tokenize.DEDENT,
                    tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER}
        return ' '.join(t.string for t in v if t.type not in excluded and t.string.strip()), False
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return ' '.join(re.findall(r'\w+|[^\w\s]', code)), True


def filename(suffix):
    matches = list(SOURCES.glob('staqc_python_how_to_do_it*' + suffix))
    if len(matches) != 1:
        raise ValueError(f'Expected one source for {suffix}, got {matches}')
    return matches[0]


def main():
    if (OUT / 'STAQC_ELIGIBILITY.json').exists():
        raise FileExistsError('StaQC eligibility already sealed')
    local = read(OUT / 'LOCAL_ELIGIBILITY.json')
    index = ROOT / local['reference_index']['path']
    if record(index) != local['reference_index']:
        raise ValueError('Reference hashes changed')
    with index.open('rb') as h:
        seen = pickle.load(h)  # Locally constructed hashes/IDs, not third-party pickle.
    paths = {key: filename(suffix) for key, suffix in {
        'multi_code': 'multiple_iid_to_code.pickle', 'multi_title': 'multiple_qid_to_title.pickle',
        'single_code': 'single_code_answer_qid_to_code.pickle',
        'single_title': 'single_code_answer_qid_to_title.pickle'}.items()}
    raw = {k: load_data(p) for k, p in paths.items()}
    if not all(isinstance(d, dict) for d in raw.values()):
        raise ValueError('Author source is not a data mapping')
    accepted = [ast.literal_eval(line) for line in (SOURCES / 'staqc_python_multi_code_iids.txt').read_text().splitlines() if line.strip()]
    if len(set(accepted)) != len(accepted) or any(type(q) is not int or type(i) is not int for q, i in accepted):
        raise ValueError('Malformed accepted code IDs')
    if any(iid not in raw['multi_code'] or iid[0] not in raw['multi_title'] for iid in accepted):
        raise ValueError('Accepted multi-code pair missing from author source')
    stats = {'single_pairs': len(raw['single_code']), 'accepted_multi_pairs': len(accepted),
             'single_missing_titles': len(set(raw['single_code']) - set(raw['single_title'])),
             'multi_source_all_pairs_not_all_accepted': len(raw['multi_code'])}
    data = []
    for kind, iterator in (
        ('single', ((qid, 0, code, raw['single_title'][qid]) for qid, code in raw['single_code'].items() if qid in raw['single_title'])),
        ('multi', ((qid, i, raw['multi_code'][qid, i], raw['multi_title'][qid]) for qid, i in accepted)),
    ):
        for qid, ci, code, title in iterator:
            if not isinstance(code, str) or not isinstance(title, str):
                raise ValueError('Non-string code/title in StaQC data')
            text, fallback = code_tokens(code)
            f = fingerprints({'query': title, 'code': code, 'code_tokens': text,
                              'url': f'https://stackoverflow.com/questions/{qid}#snippet-{ci}'})
            data.append({'unit_id': f'staqc-{qid}-{ci}', 'question_id': qid, 'snippet_index': ci,
                         'kind': kind, 'query': title, 'code_raw': code, 'code_text': text,
                         'lexical_fallback': fallback, 'fingerprints': f})
    uf = UnionFind()
    for r in data:
        anchor = f'soq:{r["question_id"]}'
        uf.find(anchor)
        for k in ('q_lex', 'code_ws', 'code_tokens'):
            for v in r['fingerprints'][k]:
                uf.join(anchor, k + ':' + v)
    bad_components = set()
    stats['overlap_pairs_by_fingerprint'] = dict.fromkeys(KINDS, 0)
    for r in data:
        hits = {k: bool(r['fingerprints'][k] & seen[k]) for k in KINDS}
        for k, v in hits.items():
            stats['overlap_pairs_by_fingerprint'][k] += v
        malformed = not r['code_text'].strip() or len(r['query'].split()) < 3 or '<extra_id_' in r['query']
        if any(hits.values()) or malformed:
            bad_components.add(uf.find(f'soq:{r["question_id"]}'))
    eligible = []
    allmeta = []
    for r in data:
        component = uf.find(f'soq:{r["question_id"]}')
        r['group_id'] = digest('staqc-connected/' + component)[:24]
        r['eligible'] = component not in bad_components
        r['fingerprints'] = {k: sorted(v) for k, v in r['fingerprints'].items()}
        allmeta.append({k: v for k, v in r.items() if k not in ('code_raw', 'code_text')})
        if r['eligible']:
            eligible.append(r)
    # Primary confirmation stays at one known accepted snippet per source question,
    # matching previous single-target RR. Restrict to the author's single-code stratum;
    # multi-code data are used to link duplicates/queries and detect exposure only.
    primary = [r for r in eligible if r['kind'] == 'single']
    sizes = collections.Counter(r['group_id'] for r in primary)
    meta = OUT / 'STAQC_PAIR_METADATA.jsonl'
    public_source = OUT / 'STAQC_ELIGIBLE_PAIRS.jsonl'
    writel(meta, allmeta)
    writel(public_source, primary)
    write(OUT / 'STAQC_ELIGIBILITY.json', {'created_utc': now(), 'new_method_outcomes_read': False,
          'source': read(SOURCES / 'staqc_PIN.json'), 'source_files': [record(p) for p in paths.values()],
          'author_label_contract': 'accepted single-code answer paired with question title; mined how-to titles, no new human relevance judgments',
          'primary_stratum_rule': 'single-code source to preserve single-target RR; all accepted multi-code entries included in overlap-connected grouping, not primary retrieval targets',
          'code_representation': 'Python tokenize lexical token strings without comments/layout; incomplete snippets use documented regex fallback',
          'stats': stats, 'eligible_all_pairs': len(eligible), 'eligible_primary_pairs': len(primary),
          'eligible_primary_questions': len({r['question_id'] for r in primary}),
          'groups': len(sizes), 'group_sizes': dict(sorted(sizes.items())),
          'metadata': record(meta), 'eligible_pairs': record(public_source),
          'reference_eligibility_sha256': record(OUT / 'LOCAL_ELIGIBILITY.json')['sha256'],
          'license': 'author repository CC-BY4.0; Stack Overflow provenance/attribution retained; artifact distributes source IDs/hash/getter, no bundled third-party snippet dataset',
          'scope': 'new locally unevaluated SO-question/snippet setting; not new Bing queries, pure repository transfer, or proof against foundation pretraining/semantic clones',
          'new_confirmation': False})
    print({'eligible_primary_pairs': len(primary), 'groups': len(sizes), 'stats': stats}, flush=True)


if __name__ == '__main__':
    main()
