from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np

from .common import MODEL, OUT, ROOT, deny_outcomes, now, progress, read, record, require, rows, sha, verify_design, verify_seal, write
from .encoder import encode, load_encoder, stable_order


def input_texts(cfg):
    return list(dict.fromkeys([r['query'] for r in rows(ROOT / cfg['queries'])] +
                             [r['rewrite'] for r in rows(ROOT / cfg['candidates'])]))


def encode_inputs(model, tokenizer, texts, directory, kind, max_length):
    directory.mkdir(parents=True, exist_ok=True)
    vector_path = directory / f'{kind}_vectors.npy'
    token_path = directory / f'{kind}_token_lengths.npy'
    truncation_path = directory / f'{kind}_truncated.npy'
    if vector_path.exists():
        require(np.load(vector_path, mmap_mode='r').shape == (len(texts), 768), 'Completed embedding coverage')
        require(token_path.exists() and truncation_path.exists(), 'Incomplete embedding sidecar')
        return
    chunk_dir = directory / f'{kind}_chunks'
    chunk_dir.mkdir(exist_ok=True)
    for start in range(0, len(texts), 512):
        chunk = chunk_dir / f'{start:07d}.npz'
        current = texts[start:start + 512]
        if chunk.exists():
            with np.load(chunk) as cached:
                require(cached['vectors'].shape == (len(current), 768), 'Invalid resumed vector chunk')
            continue
        vectors, lengths, truncated = encode(model, tokenizer, current, max_length, batch_size=16)
        temp = chunk.with_suffix('.partial.npz')
        np.savez(temp, vectors=vectors, lengths=np.array(lengths, np.int16), truncated=np.array(truncated, bool))
        temp.rename(chunk)
        progress('encoding', domain=directory.name, kind=kind, rows_done=min(start + 512, len(texts)), rows_total=len(texts))
    pieces = [np.load(chunk_dir / f'{start:07d}.npz') for start in range(0, len(texts), 512)]
    for path, field in [(token_path, 'lengths'), (truncation_path, 'truncated'), (vector_path, 'vectors')]:
        arr = np.concatenate([p[field] for p in pieces])
        temp = path.with_suffix('.partial.npy')
        np.save(temp, arr, allow_pickle=False)
        temp.rename(path)
    for p in pieces:
        p.close()


def rank():
    design = verify_design()
    verify_seal('ACTIONS_SEALED.json')
    if (OUT / 'RANKINGS_SEALED.json').exists():
        return verify_seal('RANKINGS_SEALED.json')
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(202610203)
    all_files = []
    model, tokenizer, _ = load_encoder(MODEL)
    for domain, cfg in design['samples'].items():
        started = time.perf_counter()
        directory = OUT / f'cache/{domain}'
        destination = OUT / f'rankings/{domain}.jsonl'
        manifest = destination.with_suffix('.manifest.json')
        if manifest.exists():
            m = read(manifest)
            require(m['design_sha256'] == sha(OUT / 'DESIGN.json'), 'Ranking manifest design changed')
            for item in m['files']:
                require(sha(ROOT / item['path']) == item['sha256'], 'Completed cache changed')
            all_files += [record(destination), record(manifest)] + m['files']
            continue
        corpus = list(rows(ROOT / cfg['corpus']))
        ids = np.array([r['doc_id'] for r in corpus], np.int64)
        with deny_outcomes():
            texts = input_texts(cfg)
            directory.mkdir(parents=True, exist_ok=True)
            if not (directory / 'texts.json').exists():
                write(directory / 'texts.json', texts)
            require(read(directory / 'texts.json') == texts, 'Unique text order changed')
            encode_inputs(model, tokenizer, [r['code_text'] for r in corpus], directory, 'code', 256)
            encode_inputs(model, tokenizer, texts, directory, 'query', 128)
            cv = np.load(directory / 'code_vectors.npy')
            qv = np.load(directory / 'query_vectors.npy')
            require(cv.shape == (len(corpus), 768) and qv.shape == (len(texts), 768), 'Vector coverage')
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix('.partial.jsonl')
            previous = sum(1 for _ in rows(partial)) if partial.exists() else 0
            if previous:
                require(all(r['text'] == texts[i] for i, r in enumerate(rows(partial))), 'Resumed ranking text order changed')
            with partial.open('a', encoding='utf-8') as h, threadpool_limits(limits=4):
                for start in range(previous, len(texts), 256):
                    scores = qv[start:start + 256] @ cv.T
                    for text, values in zip(texts[start:start + 256], scores):
                        order = stable_order(values, ids)
                        h.write(json.dumps({'text': text, 'ranked_doc_ids': ids[order].tolist(),
                                            'scores': values[order].astype(float).tolist()}, ensure_ascii=False, allow_nan=False) + '\n')
                    h.flush()
                    if start % 2048 == 0 or start + 256 >= len(texts):
                        progress('ranking', domain=domain, texts_done=min(start + 256, len(texts)), texts_total=len(texts))
            partial.rename(destination)
        cache_files = [record(p) for p in sorted(directory.glob('*')) if p.is_file()]
        write(manifest, {'created_utc': now(), 'domain': domain, 'design_sha256': sha(OUT / 'DESIGN.json'),
              'ranking_sha256': sha(destination), 'files': cache_files, 'unique_texts': len(texts),
              'documents': len(corpus), 'qrels_read': False, 'seconds': time.perf_counter() - started})
        all_files += [record(destination), record(manifest)] + cache_files
    del model
    gc.collect()
    torch.cuda.empty_cache()
    write(OUT / 'RANKINGS_SEALED.json', {'created_utc': now(), 'design_sha256': sha(OUT / 'DESIGN.json'),
          'files': all_files, 'qrels_read': False, 'new_model_effects_computed': False})


if __name__ == '__main__':
    rank()
