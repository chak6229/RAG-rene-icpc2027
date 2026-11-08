"""Frozen, label-free original-query/code evidence for action512.

This module calls only the pure encoder and action-statistic helpers from the
historical evidence implementation.  Its prepare/main/freeze are never called.
All artifacts are written to a new experiment directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoTokenizer

from crsu.core import rrf
from run_bm25_acquisition_pilot import BM25, tokens
from run_relevance_selection_pilot import (
    DEPENDENCIES, FEATURES, SIGNALS, STATS, action_evidence, project_encode,
)

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/strong_flow_uq_20261014/evidence'
DATA = ROOT / 'results/uq_pilot_20261014/action_witness_uq/data'
MODEL = ROOT / 'models/codebert-base'
OLD = ROOT / 'results/uq_pilot_20261012/relevance_selection'
PROJECTION_SEED = 12012


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def load(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        pickle.dump(value, stream, protocol=4)
    temp.replace(path)


def status(stage, **kwargs):
    dump(OUT / 'STATUS.json', dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
                                 test_used=False, labels_used=False, **kwargs))


def freeze(smoke_queries):
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / 'run_relevance_selection_pilot.py',
               ROOT / 'experiments/action_witness_uq_20261014/gpu_data.py']
    sources.extend(ROOT / p for p in DEPENDENCIES)
    sources = list(dict.fromkeys(sources))
    inputs = [DATA / p for p in ('action_items.pkl', 'action_code_corpus.pkl',
                                 'GPU_PROTOCOL.json', 'FINAL_MANIFEST.json')]
    if smoke_queries:
        inputs += [DATA / 'valid_items.pkl', DATA / 'valid_code_corpus.pkl',
                   OLD / 'relevance_evidence.pkl', OLD / 'projection.npy', OLD / 'PROTOCOL.json']
    hashes = dict(source_hashes={str(p): sha(p) for p in sources},
                  input_hashes={str(p): sha(p) for p in inputs},
                  model_hashes={str(p): sha(p) for p in MODEL.iterdir() if p.is_file()})
    path = OUT / 'PROTOCOL.json'
    if path.exists():
        protocol = json.loads(path.read_text('utf8'))
        assert all(protocol[k] == hashes[k] for k in hashes), 'frozen hashes changed'
        assert protocol['valid_smoke_queries'] == smoke_queries
        return protocol
    original = json.loads((DATA / 'GPU_PROTOCOL.json').read_text('utf8'))
    original_manifest = json.loads((DATA / 'FINAL_MANIFEST.json').read_text('utf8'))
    # Validate the exact old frozen inputs being reused, without traversing or
    # opening any raw training/valid/test dataset.
    for p in (DATA / 'action_items.pkl', DATA / 'action_code_corpus.pkl'):
        expected = original_manifest['outputs'].get(str(p))
        if expected is None:
            expected = original['input_hashes'].get(str(p))
        assert expected is not None and sha(p) == expected, str(p)
    protocol = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        test_used=False, exploratory=True, labels_used=False,
        corpus='4096 existing action code_text strings only; no docstring fallback',
        query_order='exact action_items.pkl order',
        document_selection='sorted union of every original/rewrite/RRF Top10; never add target',
        semantic=dict(encoder='frozen models/codebert-base', training=False,
            query_tokens=64, code_tokens=128, projection_dim=64, projection_seed=PROJECTION_SEED,
            projection='numpy default_rng.normal / 8, float32',
            encoder_batch_size=32, autocast='bfloat16', token_storage='float16',
            valid_tokens='exclude special and padding',
            token_maxsim='mean original query token max projected cosine to code tokens',
            pooled_cosine='mean unprojected valid-token hidden states, L2 normalized'),
        lexical='BM25 over complete frozen action4096 code_text; score divided by corpus maximum; IDF coverage',
        document_signals=list(SIGNALS), action_statistics=list(STATS), features=list(FEATURES),
        valid_smoke_queries=smoke_queries,
        smoke_tolerances=dict(lexical_atol=1e-12, semantic_atol=.005,
                              old_feature_reconstruction_atol=1e-12),
        **hashes)
    dump(path, protocol)
    for source in sources:
        target = OUT / 'frozen_source' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    status('frozen')
    return protocol


def observable_items(items):
    """Explicit label boundary: y/original_rr/target/source IDs never enter evidence."""
    result = []
    for item in items:
        original = np.asarray(item['original_top100'], dtype=int).copy()
        candidates = [np.asarray(x, dtype=int).copy() for x in item['candidate_top100']]
        assert len(original) == 100 and all(len(x) == 100 for x in candidates)
        result.append(dict(query_index=int(item['query_index']), query=str(item['query']),
                           original_top100=original, candidate_top100=candidates))
    return result


def unions(items):
    per_query = []
    for item in items:
        docs = set(map(int, item['original_top100'][:10]))
        for top in item['candidate_top100']:
            docs.update(map(int, top[:10]))
            docs.update(map(int, rrf(item['original_top100'], top)[:10]))
        per_query.append(np.array(sorted(docs), dtype=int))
    return per_query


def lexical_index(code_values):
    index = BM25(code_values)
    termsets = [set(tokens(text)) for text in code_values]
    idf = {word: float(np.log1p((index.n - len(ids) + .5) / (len(ids) + .5)))
           for word, (ids, _) in index.posting.items()}
    return index, termsets, idf


def signals_for(query, ids, qt, qm, qmean, dt, dm, dmean, doc_lookup, index, termsets, idf):
    di = np.array([doc_lookup[int(d)] for d in ids])
    scores = np.zeros(index.n)
    terms = set(tokens(query))
    for term in terms:
        if term in index.posting:
            posting_ids, weights = index.posting[term]
            scores[posting_ids] += weights
    bm25 = scores[ids] / max(scores.max(), 1e-12)
    denom = sum(idf.get(t, 0) for t in terms)
    lexical = np.array([sum(idf.get(t, 0) for t in terms & termsets[int(d)]) /
                        max(denom, 1e-12) for d in ids])
    q = torch.tensor(qt[qm].astype(np.float32), device='cuda')
    d = torch.tensor(dt[di].astype(np.float32), device='cuda')
    mask = torch.tensor(dm[di], device='cuda')
    with torch.inference_mode():
        sim = torch.einsum('qh,dlh->dql', q, d)
        sim = sim.masked_fill(~mask[:, None, :], -1.)
        maxsim = sim.max(-1).values.mean(-1).cpu().numpy() if len(q) else np.zeros(len(di))
    pooled = dmean[di] @ qmean
    signals = np.column_stack((maxsim, bm25, lexical, pooled))
    assert signals.shape == (len(ids), 4) and np.isfinite(signals).all()
    return signals


def extras_for(item, ids, signals):
    extras = {}
    for family in ('rewrite', 'fusion'):
        extras[family] = np.array([
            action_evidence(item['original_top100'], top if family == 'rewrite' else
                            rrf(item['original_top100'], top), ids, signals)
            for top in item['candidate_top100']]).reshape(-1, len(FEATURES))
        assert np.isfinite(extras[family]).all()
    return extras


def smoke(model, tokenizer, projection, count):
    if not count:
        return dict(performed=False)
    cached = load(OLD / 'relevance_evidence.pkl')[:count]
    by_query = {int(e['query_index']): e for e in load(DATA / 'valid_items.pkl')}
    items = observable_items([by_query[int(z['query_index'])] for z in cached])
    codes = load(DATA / 'valid_code_corpus.pkl')
    docs = np.array(sorted({int(d) for z in cached for d in z['doc_ids']}), dtype=int)
    lookup = {int(d): i for i, d in enumerate(docs)}
    qt, qm, qmean = project_encode(model, tokenizer, [e['query'] for e in items], 64, projection)
    dt, dm, dmean = project_encode(model, tokenizer, [codes[int(d)] for d in docs], 128, projection)
    index, termsets, idf = lexical_index(codes)
    delta = np.zeros(4)
    feature_error = 0.
    for local, (item, cached_row) in enumerate(zip(items, cached)):
        ids = cached_row['doc_ids']
        signals = signals_for(item['query'], ids, qt[local], qm[local], qmean[local],
                              dt, dm, dmean, lookup, index, termsets, idf)
        delta = np.maximum(delta, np.max(np.abs(signals - cached_row['doc_signals']), axis=0))
        # Pure action-statistic parity can be checked exactly with the old
        # stored signals even when GPU batch kernels differ numerically.
        reconstructed = extras_for(item, ids, cached_row['doc_signals'])
        for family in reconstructed:
            err = float(np.max(np.abs(reconstructed[family] - cached_row['x'][family])))
            feature_error = max(feature_error, err)
            np.testing.assert_allclose(reconstructed[family], cached_row['x'][family], atol=1e-12, rtol=0)
    np.testing.assert_allclose(delta[[1, 2]], 0, atol=1e-12, rtol=0)
    assert np.max(delta[[0, 3]]) <= .005, delta.tolist()
    np.testing.assert_array_equal(projection.cpu().numpy(), np.load(OLD / 'projection.npy'))
    return dict(performed=True, queries=count, documents=len(docs),
                query_indices=[e['query_index'] for e in items],
                max_abs_error_by_signal=dict(zip(SIGNALS, map(float, delta))),
                old_x28_reconstruction_max_abs_error=feature_error,
                projection_exact=True, passed=True)


def run(protocol):
    path = OUT / 'relevance_evidence.pkl'
    if path.exists():
        manifest = json.loads((OUT / 'MANIFEST.json').read_text('utf8'))
        assert all(sha(p) == digest for p, digest in manifest['outputs'].items())
        print('Frozen evidence already complete', flush=True)
        return
    start = time.perf_counter()
    items = observable_items(load(DATA / 'action_items.pkl'))
    corpus = load(DATA / 'action_code_corpus.pkl')
    assert len(items) == 512 and len(corpus) == 4096
    assert [int(row['doc_id']) for row in corpus] == list(range(4096))
    # Deliberately only read code_text; corpus query/docstring is never used.
    codes = [str(row['code_text']) for row in corpus]
    assert all(codes)
    per_query = unions(items)
    docs = np.array(sorted({int(d) for ds in per_query for d in ds}), dtype=int)
    assert np.min(docs) >= 0 and np.max(docs) < len(codes)
    lookup = {int(d): i for i, d in enumerate(docs)}
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    projection = torch.tensor(np.random.default_rng(PROJECTION_SEED).normal(
        size=(model.config.hidden_size, 64)) / 8, dtype=torch.float32, device='cuda')
    np.save(OUT / 'projection.npy', projection.cpu().numpy())
    smoke_start = time.perf_counter()
    comparison = smoke(model, tokenizer, projection, protocol['valid_smoke_queries'])
    comparison['elapsed_seconds'] = time.perf_counter() - smoke_start
    dump(OUT / 'VALID_IMPLEMENTATION_SMOKE.json', comparison)
    encoding_start = time.perf_counter()
    print(f'Encode action evidence: {len(docs)} codes / {len(items)} queries', flush=True)
    qt, qm, qmean = project_encode(model, tokenizer, [e['query'] for e in items], 64, projection)
    dt, dm, dmean = project_encode(model, tokenizer, [codes[int(d)] for d in docs], 128, projection)
    encoding_seconds = time.perf_counter() - encoding_start
    del model
    torch.cuda.empty_cache()
    # Reusable original-query64 and original-code128 token-level caches.  The
    # code cache stores just the label-free Top10-union documents, not targets.
    np.savez(OUT / 'query64_cache.npz', query_indices=np.array([e['query_index'] for e in items]),
             projected_tokens=qt, valid_mask=qm, pooled=qmean)
    np.savez(OUT / 'code128_cache.npz', doc_ids=docs, projected_tokens=dt, valid_mask=dm, pooled=dmean)
    status('encoded', queries=len(items), documents=len(docs), encoding_seconds=encoding_seconds)
    lexical_start = time.perf_counter()
    index, termsets, idf = lexical_index(codes)
    lexical_seconds = time.perf_counter() - lexical_start
    evidence = []
    signal_start = time.perf_counter()
    for local, (item, ids) in enumerate(zip(items, per_query)):
        signals = signals_for(item['query'], ids, qt[local], qm[local], qmean[local],
                              dt, dm, dmean, lookup, index, termsets, idf)
        evidence.append(dict(query_index=item['query_index'], doc_ids=ids,
                             doc_signals=signals, x=extras_for(item, ids, signals)))
        if (local + 1) % 80 == 0:
            print(f'Evidence {local + 1}/{len(items)} {time.perf_counter() - start:.1f}s', flush=True)
    assert [e['query_index'] for e in evidence] == [e['query_index'] for e in items]
    signal_seconds = time.perf_counter() - signal_start
    save(path, evidence)
    cost = dict(completed=True, test_used=False, labels_used=False,
                queries=len(items), corpus_documents=len(codes), encoded_documents=len(docs),
                candidates=sum(len(e['candidate_top100']) for e in items),
                query_document_pairs=sum(map(len, per_query)),
                projected_dimensions=64, document_signals=list(SIGNALS), x_dimension=28,
                encoding_seconds=encoding_seconds, lexical_index_seconds=lexical_seconds,
                signal_and_features_seconds=signal_seconds,
                smoke_seconds=comparison['elapsed_seconds'], elapsed_seconds=time.perf_counter() - start,
                schema={'query_index': 'alignment key only', 'doc_ids': 'sorted observable Top10 union int64',
                        'doc_signals': '(n_documents,4) float64',
                        'x': {'rewrite': '(n_candidates,28) float64', 'fusion': '(n_candidates,28) float64'}},
                semantic_retriever_distinction='codebert-base independent evidence; not checkpoint-3000 vectors')
    dump(OUT / 'evidence_status.json', cost)
    names = ['PROTOCOL.json', 'relevance_evidence.pkl', 'projection.npy', 'query64_cache.npz',
             'code128_cache.npz', 'evidence_status.json', 'VALID_IMPLEMENTATION_SMOKE.json']
    manifest = dict(completed=True, test_used=False, labels_used=False,
                    outputs={str(OUT / n): sha(OUT / n) for n in names})
    dump(OUT / 'MANIFEST.json', manifest)
    status('complete', queries=len(items), candidates=cost['candidates'],
           query_document_pairs=cost['query_document_pairs'], elapsed_seconds=cost['elapsed_seconds'])
    print(json.dumps(cost, ensure_ascii=False), flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-queries', type=int, default=2)
    args = parser.parse_args()
    assert 0 <= args.smoke_queries <= 4
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        protocol = freeze(args.smoke_queries)
        run(protocol)


if __name__ == '__main__':
    main()
