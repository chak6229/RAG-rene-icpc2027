from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from transformers import AutoTokenizer

import run_t5_seed_uq_pilot as original
from crsu.core import rrf
from experiments.strong_flow_uq_20261014 import build_evidence as evidence
from experiments.rene_repaired_confirmation_20261019.common import (
    DATA, OUT, SEEDS, FAMILIES, load, save, read_json, write_json, sha, require_runtime, rr,
)
from experiments.rene_repaired_confirmation_20261019.features import union_draws, sampling_features
from vac_mqr import CandidateQualityValidator


def main():
    require_runtime()
    out = OUT / 'development'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'MANIFEST.json').exists():
        from experiments.rene_repaired_confirmation_20261019.common import check_hashes
        check_hashes(read_json(out / 'MANIFEST.json')['outputs'])
        print('Verified immutable four-seed development data', flush=True)
        return
    items = load(DATA / 'action_items.pkl')
    generations = original.read_jsonl(DATA / 'generation.jsonl')
    vectors = np.load(DATA / 'action_candidate_vectors.npy', mmap_mode='r')
    old_evidence = load(ROOT / 'results/strong_flow_uq_20261014/evidence/relevance_evidence.pkl')
    scores = load(DATA / 'ce_large_dense_4096_scores_action.pkl')
    corpus = load(DATA / 'action_code_corpus.pkl')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/codebert-base', local_files_only=True)
    code_lengths = [len(x) for x in tokenizer([e['code_text'] for e in corpus],
                       add_special_tokens=False, truncation=True, max_length=160)['input_ids']]
    query_lengths = [len(x) for x in tokenizer([e['query'] for e in items],
                        add_special_tokens=False, truncation=True, max_length=64)['input_ids']]
    result = []
    offset, feature_error, label_error, evidence_error = 0, 0., 0., 0.
    for index, (item, generated, oldz, ce) in enumerate(zip(items, generations, old_evidence, scores)):
        if not (item['query_index'] == generated['query_index'] == oldz['query_index'] == ce['query_index']):
            raise ValueError('Historical cache alignment failure')
        n = len(item['x'])
        vv = np.asarray(vectors[offset:offset + n]); offset += n
        parity = sampling_features(generated, vv, item['candidate_top100'],
                                   item['seed_appearances'], tuple(original.SEEDS))
        feature_error = max(feature_error, float(np.max(abs(parity - item['seed_x']))) if n else 0.)
        np.testing.assert_allclose(parity, item['seed_x'], atol=1e-12, rtol=0)
        selected_gen, candidates, appearances = union_draws(generated)
        lookup = {CandidateQualityValidator.canonical(c['text']): j for j, c in enumerate(item['candidates'])}
        ix = np.array([lookup[CandidateQualityValidator.canonical(c['text'])] for c in candidates], int)
        tops = [item['candidate_top100'][j].copy() for j in ix]
        seed_x = sampling_features(selected_gen, vv[ix], tops, appearances)
        e = {k: item[k] for k in ('query_index', 'source_idx', 'query', 'url', 'repo', 'original_rr')}
        e.update(x=item['x'][ix].copy(), seed_x=seed_x,
                 original_top100=item['original_top100'].copy(), candidate_top100=tops,
                 candidates=candidates, seed_appearances=appearances,
                 y={family: item['y'][family][ix].copy() for family in FAMILIES},
                 ce_query_length=query_lengths[index],
                 ce_code_lengths={d: length for d, length in enumerate(code_lengths)},
                 code_word_counts={d: len(c['code_text'].split()) for d, c in enumerate(corpus)})
        target = int(item['query_index'])
        np.testing.assert_allclose(rr(e['original_top100'], target), e['original_rr'], atol=1e-12)
        for family in FAMILIES:
            yy = [rr(top if family == 'rewrite' else rrf(e['original_top100'], top), target)
                  - rr(e['original_top100'], target) for top in tops]
            np.testing.assert_allclose(yy, e['y'][family], atol=1e-12, rtol=0)
            if len(yy):
                label_error = max(label_error, float(np.max(abs(np.array(yy) - e['y'][family]))))
        ids = evidence.unions([e])[0]
        doc_lookup = {int(d): j for j, d in enumerate(oldz['doc_ids'])}
        signals = oldz['doc_signals'][[doc_lookup[int(d)] for d in ids]].copy()
        z = dict(query_index=e['query_index'], doc_ids=ids, doc_signals=signals,
                 x=evidence.extras_for(e, ids, signals))
        # The candidate union changes; re-normalize its statistics rather than slicing x28.
        e['strong_evidence'] = z
        result.append(e)
    assert offset == len(vectors) and len(result) == 512
    assert len({e['repo'] for e in result}) == len(result)
    save(out / 'items.pkl', result)
    save(out / 'scores.pkl', scores)
    inputs = [DATA / p for p in ('action_items.pkl', 'generation.jsonl', 'action_candidate_vectors.npy',
                                'action_code_corpus.pkl', 'ce_large_dense_4096_scores_action.pkl')]
    inputs += [ROOT / 'results/strong_flow_uq_20261014/evidence/relevance_evidence.pkl']
    manifest = dict(seeds=SEEDS, queries=len(result), candidates=sum(len(e['x']) for e in result),
                    empty_queries=sum(not len(e['x']) for e in result),
                    minimum_candidates=min(len(e['x']) for e in result),
                    maximum_candidates=max(len(e['x']) for e in result),
                    eight_seed_feature_parity_max_error=feature_error,
                    retained_label_parity_max_error=label_error,
                    relevance_union_rebuilt=True, code_representation='historical token-only anonymization',
                    test_used=False, inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                    outputs={str((out / p).relative_to(ROOT)): sha(out / p) for p in ('items.pkl', 'scores.pkl')})
    write_json(out / 'MANIFEST.json', manifest, immutable=True)
    print(manifest, flush=True)


if __name__ == '__main__':
    main()
