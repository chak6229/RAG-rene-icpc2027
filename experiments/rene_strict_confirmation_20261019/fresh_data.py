from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.rene_strict_confirmation_20261019.common import (
    DATA, OUT, SEED, SEEDS, ARMS, FAMILIES, load, save, sha, write_json, read_json,
    require_runtime, reject_gold, protected_hashes, check_hashes, stable_top, utc,
)
from experiments.rene_strict_confirmation_20261019.features import union_draws, sampling_features

HERE = Path(__file__).resolve().parent
TRAIN = ROOT / 'data/codexglue/training/train.jsonl'
FRESH = OUT / 'fresh'
T5 = ROOT / 'models/t5-cqc-codenn-1m/best'
RETRIEVER = ROOT / 'models/codebert-codexglue-2epoch/checkpoint-3000'
SEMANTIC = ROOT / 'models/codebert-base'


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def query_key(text):
    return digest(' '.join(re.findall(r'\w+', text.casefold())))


def code_key(text):
    return digest(' '.join(text.split()))


def repository(url):
    m = re.match(r'https?://github\.com/([^/]+/[^/]+)', url)
    if not m:
        raise ValueError('Unsupported repository URL: ' + url)
    return m.group(1).casefold()


def exclusion_inventory():
    paths = [DATA / 'GPU_PROTOCOL.json',
             ROOT / 'results/uq_pilot_20261012/fullchain_t5_uq/PROTOCOL.json',
             ROOT / 'results/fresh_risk_confirmation_20261015/data/ROLE_SPLIT.json',
             ROOT / 'results/rene_independent_confirmation_20261018/PROTOCOL.json']
    gpu, ce, previous, old96 = [read_json(p) for p in paths]  # Missing metadata is an error, never silently ignored.
    groups = dict(action_queries=gpu['action_query_repos'], semantic_queries=gpu['semantic_repos'],
                  all_valid=gpu['all_valid_excluded_repos'],
                  ce_training_repositories=ce['ce_training_repos'],
                  ce_actual_queries=[repository(u) for u in ce['ce_training_urls']],
                  previously_consumed_1043=previous['selection_order_repos'],
                  invalid_96=old96['query_repositories'])
    consumed_path = ROOT / 'results/rene_repaired_confirmation_20261019/fresh/queries.json'
    groups['prior_repaired_1536'] = [q['repo'] for q in read_json(consumed_path)]
    paths.append(consumed_path)
    if any(not isinstance(v, list) or not v for v in groups.values()):
        raise ValueError('Missing/empty repository exposure group')
    blocked = {str(r).casefold() for group in groups.values() for r in group}
    return blocked, dict(groups={k: len(set(v)) for k, v in groups.items()}, excluded_repositories=sorted(blocked),
                         inputs={str(p.relative_to(ROOT)): sha(p) for p in paths})


def inventory():
    from experiments.rene_strict_confirmation_20261019.code_representation import anonymized_code
    FRESH.mkdir(parents=True, exist_ok=True)
    if (FRESH / 'INVENTORY.json').exists():
        inv = read_json(FRESH / 'INVENTORY.json')
        check_hashes(inv['inputs'])
        if sha(FRESH / 'eligible_repositories.pkl') != inv['eligible_sha256']:
            raise ValueError('Eligible inventory changed')
        print({k: inv[k] for k in ('eligible_repositories', 'excluded_repository_count')}, flush=True)
        return
    blocked, exposure = exclusion_inventory()
    seen_queries, seen_codes = set(), set()
    prior_queries_path = ROOT / 'results/rene_repaired_confirmation_20261019/fresh/queries.json'
    prior_eligible_path = ROOT / 'results/rene_repaired_confirmation_20261019/fresh/eligible_repositories.pkl'
    prior_queries = read_json(prior_queries_path)
    prior_repos = {q['repo'] for q in prior_queries}
    seen_queries.update(query_key(q['query']) for q in prior_queries)
    prior_indices = {r['source_index'] for r in load(prior_eligible_path) if r['repo'] in prior_repos}
    duplicate_inputs = [DATA / 'action_code_corpus.pkl', DATA / 'semantic_code_corpus.pkl',
                        ROOT / 'results/fresh_risk_confirmation_20261015/data/fresh_rows.pkl']
    for p in duplicate_inputs:
        for row in load(p):
            if row.get('query'):
                seen_queries.add(query_key(row['query']))
            if row.get('code_text'):
                seen_codes.add(code_key(row['code_text']))
    invalid96_path = ROOT / 'results/rene_independent_confirmation_20261018/label_free_decisions.pkl'
    invalid96 = load(invalid96_path)
    for row in invalid96['items_without_labels']:
        seen_queries.add(query_key(row['query']))
    valid_path = ROOT / 'data/codexglue/advtest/valid.jsonl'
    for line in valid_path.open(encoding='utf8'):
        row = json.loads(line)
        seen_queries.add(query_key(' '.join(row.get('docstring_tokens') or [])))
    duplicate_inputs += [invalid96_path, valid_path, prior_queries_path, prior_eligible_path]
    byrepo, rejected = {}, Counter()
    eligible_rows = 0
    start = time.perf_counter()
    for source_index, line in enumerate(TRAIN.open(encoding='utf8')):
        row = json.loads(line)
        repo = str(row.get('repo') or repository(row['url'])).casefold()
        if source_index in prior_indices:
            prior_tokens = row.get('code_tokens') or row.get('function_tokens') or []
            seen_codes.add(code_key(anonymized_code(prior_tokens)))
        if repo in blocked:
            rejected['repository_exposure'] += 1
            continue
        query = ' '.join(row.get('docstring_tokens') or [])
        tokens = row.get('code_tokens') or row.get('function_tokens') or []
        if not query.strip() or len(query.split()) > 64 or len(tokens) < 8:
            rejected['length_or_empty'] += 1
            continue
        qkey = query_key(query)
        if qkey in seen_queries:
            rejected['historical_query_duplicate'] += 1
            continue
        code = anonymized_code(tokens)
        ckey = code_key(code)
        if ckey in seen_codes:
            rejected['historical_code_duplicate'] += 1
            continue
        eligible_rows += 1
        priority = digest(f'{SEED}:fresh-row:' + row['url'])
        value = dict(source_index=source_index, repo=repo, query=query, code_text=code,
                     query_hash=qkey, code_hash=ckey, priority=priority, source_url=row['url'])
        if repo not in byrepo or (priority, row['url']) < (byrepo[repo]['priority'], byrepo[repo]['source_url']):
            byrepo[repo] = value
    # Global duplicate removal is deterministic and uses no retrieval outcomes.
    chosen, qseen, cseen = [], set(), set()
    for row in sorted(byrepo.values(), key=lambda r: (digest(f'{SEED}:fresh-repo:' + r['repo']), r['repo'])):
        if row['code_hash'] in seen_codes:
            rejected['historical_code_duplicate_final'] += 1
            continue
        if row['query_hash'] in qseen or row['code_hash'] in cseen:
            rejected['cross_repository_duplicate'] += 1
            continue
        qseen.add(row['query_hash']); cseen.add(row['code_hash']); chosen.append(row)
    save(FRESH / 'eligible_repositories.pkl', chosen)
    inputs = dict(exposure['inputs'])
    inputs.update({str(p.relative_to(ROOT)): sha(p) for p in duplicate_inputs + [TRAIN]})
    write_json(FRESH / 'EXPOSURE.json', exposure, immutable=True)
    inv = dict(eligible_repositories=len(chosen), eligible_rows=eligible_rows,
               excluded_repository_count=len(blocked), rejected=dict(rejected), inputs=inputs,
               eligible_sha256=sha(FRESH / 'eligible_repositories.pkl'),
               query_duplicate_normalization='casefold + Unicode word tokens',
               code_duplicate_normalization='historical token anonymization + whitespace normalization',
               test_used=False, selection_used_rr=False, prior_outcomes_reused=False,
               upstream_CodeBERT_train_exposure=True, elapsed_seconds=time.perf_counter() - start)
    write_json(FRESH / 'INVENTORY.json', inv, immutable=True)
    print({k: inv[k] for k in ('eligible_repositories', 'eligible_rows', 'excluded_repository_count', 'elapsed_seconds')}, flush=True)


def model_assets():
    paths = []
    for directory in (T5, RETRIEVER, SEMANTIC):
        paths += [p for p in directory.iterdir() if p.is_file() and p.suffix in ('.safetensors', '.bin', '.json', '.txt', '.model')]
    paths += [DATA / 'ce_large_dense_4096.pt', DATA / 'cls_projection.npy']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def freeze():
    require_runtime(gpu=True)
    protocol_path = FRESH / 'PROTOCOL.json'
    if protocol_path.exists():
        p = read_json(protocol_path)
        check_hashes(p['binding_hashes'])
        check_hashes(p['protected_hashes'])
        return p
    inventory()
    parity = read_json(OUT / 'CODE_REPRESENTATION_PARITY.json')
    if parity['checked'] != 4096 or parity['mismatched'] != 0:
        raise ValueError('All historical code inputs must reconstruct exactly before confirmation')
    power = read_json(OUT / 'development/POWER_PRIMARY.json')
    available = load(FRESH / 'eligible_repositories.pkl')
    planned = power['confirmation_queries']
    # If the eligible frame is smaller, execute a disclosed census and report its actual MDE.
    n = min(planned, len(available))
    if n < 1024:
        raise RuntimeError('Fewer than 1024 clean repositories; confirmation design is infeasible')
    roster = available[:n]
    blocks = np.array_split(np.arange(n), max(1, math.ceil(n / 3072)))
    queries, outcomes, corpus_manifest = [], [], []
    for block, indices in enumerate(blocks):
        targets = [roster[int(i)] for i in indices]
        target_repos = {r['repo'] for r in targets}
        distractors = sorted([r for r in available if r['repo'] not in target_repos],
                             key=lambda r: digest(f'{SEED}:corpus:{block}:' + r['repo']))
        if len(targets) + len(distractors) < 4096:
            raise RuntimeError('Not enough distinct clean code documents for a 4096 corpus')
        corpus_rows = targets + distractors[:4096 - len(targets)]
        corpus_rows = sorted(corpus_rows, key=lambda r: digest(f'{SEED}:doc-order:{block}:' + r['code_hash']))
        doc_by_hash = {row['code_hash']: d for d, row in enumerate(corpus_rows)}
        if len(doc_by_hash) != 4096:
            raise ValueError('Duplicate code documents in corpus')
        corpus = [dict(doc_id=d, code_text=row['code_text']) for d, row in enumerate(corpus_rows)]
        corpus_path = FRESH / f'corpus/block{block}.pkl'
        save(corpus_path, corpus)
        corpus_manifest.append(dict(block=block, documents=4096, query_count=len(indices),
                                    path=str(corpus_path.relative_to(ROOT)), sha256=sha(corpus_path)))
        for i in indices:
            source = roster[int(i)]
            qid = 1000000 + int(i)  # Never equal to a target code document ID.
            queries.append(dict(query_index=qid, query=source['query'], repo=source['repo'],
                                block=block, url='opaque-query:' + str(qid)))
            outcomes.append(dict(query_index=qid, target_doc_id=doc_by_hash[source['code_hash']],
                                 repo=source['repo'], block=block, query_hash=source['query_hash'],
                                 source_index=source['source_index'], source_url=source['source_url']))
    write_json(FRESH / 'queries.json', queries, immutable=True)
    write_json(FRESH / 'private_outcomes.json', outcomes, immutable=True)
    runtime = require_runtime(gpu=True)
    lock = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (FRESH / 'requirements-lock.txt').write_text(lock, encoding='utf8')
    code_paths = list(HERE.glob('*.py'))
    # Freeze every Python dependency that the replay can import, rather than an incomplete hand list.
    code_paths += list(ROOT.glob('*.py')) + list((ROOT / 'crsu').rglob('*.py'))
    code_paths += list((ROOT / 'experiments').rglob('*.py'))
    code_paths = sorted(set(code_paths))
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in code_paths}
    bindings.update(model_assets())
    bindings.update(read_json(OUT / 'development/TRAINING_COMPLETE.json')['models'])
    bindings.update(read_json(OUT / 'development/WEAK_CONTROL.json')['models'])
    for p in [OUT / 'development/DESIGN.json', OUT / 'development/POWER.json', OUT / 'development/POWER_PRIMARY.json',
              OUT / 'development/MANIFEST.json', OUT / 'development/TESTS.json',
              OUT / 'development/WEAK_CONTROL.json', OUT / 'development/TRAINING_COMPLETE.json',
              OUT / 'CODE_REPRESENTATION_PARITY.json',
              OUT / 'FROZEN_HEADS_IDENTICAL.json',
              FRESH / 'queries.json', FRESH / 'INVENTORY.json',
              FRESH / 'requirements-lock.txt']:
        bindings[str(p.relative_to(ROOT))] = sha(p)
    bindings.update({r['path']: r['sha256'] for r in corpus_manifest})
    p = dict(format='rene-matched-four-seed-confirmation-v1', frozen_at_utc=utc(),
             query_count=n, planned_queries=planned, eligible_frame=len(available),
             sample_plan_complete=n >= planned, eligibility_census=n < planned,
             target_effect=.005, actual_planning_mde=power['planning_sd'] * power['z_sum'] / math.sqrt(n),
             seeds=SEEDS, query_units='one query per repository', corpus=corpus_manifest,
             generation=dict(chunk_queries=8, decoding_seeds=SEEDS, seed_schedule='seed + global chunk start',
               whitespace_positions=True, do_sample=True, top_k=50, top_p=1., temperature=1.,
               max_new_tokens=12, information_gain='mean negative generated-token entropy',
               accepted_top3='descending information_gain, quality filtering before top3'),
             feature_schema=dict(base=34, seed=12, relevance=28, supervised=10,
                                 no_seed_gain=72, seed_gain=84, no_seed_update=214, seed_update=226),
             inference_boundary='opaque query ID; no target/code source URL/labels; actual actions sealed before outcome reads',
             statistics=read_json(OUT / 'development/DESIGN.json')['multiplicity'],
             retrieval=dict(corpus_documents_per_block=4096, top_k=100,
                 ties='score descending, document ID ascending', pooling='L2 CLS', max_length=256,
                 precision='BF16 autocast, float32 vectors', code='historical token-only anonymization',
                 fusion='equal-weight original/rewrite Top100 RRF offset60, ties doc ID ascending'),
             sample_scope='random hash sample from eligible finite train-repository frame; shared frozen corpus per block',
             upstream_retriever_train_exposure=True, test_used=False, risk_certified=False,
             private_outcomes_sha256=sha(FRESH / 'private_outcomes.json'),
             runtime=runtime, binding_hashes=bindings, protected_hashes=protected_hashes())
    write_json(protocol_path, p, immutable=True)
    print('Confirmation protocol frozen: ' + str({k: p[k] for k in ('query_count', 'planned_queries', 'eligible_frame', 'sample_plan_complete')}), flush=True)
    return p


def generate():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from experiments.action_witness_uq_20261014.gpu_data import generate_chunk
    p = freeze()
    queries = read_json(FRESH / 'queries.json')
    destination = FRESH / 'generation'
    destination.mkdir(exist_ok=True)
    if (FRESH / 'GENERATION_COMPLETE.json').exists():
        check_hashes(read_json(FRESH / 'GENERATION_COMPLETE.json')['outputs'])
        print('Verified completed four-seed generation', flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(T5, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5, local_files_only=True,
                                               torch_dtype=torch.bfloat16).to('cuda').eval()
    began = time.perf_counter()
    for start in range(0, len(queries), 8):
        path = destination / f'{start:06d}.json'
        if path.exists():
            saved = read_json(path)
            if saved['protocol_sha256'] != sha(FRESH / 'PROTOCOL.json'):
                raise ValueError('Generation checkpoint bound to another protocol')
            continue
        records = generate_chunk(model, tokenizer, queries[start:start + 8], start, seeds=SEEDS)
        write_json(path, dict(protocol_sha256=sha(FRESH / 'PROTOCOL.json'), rows=records), immutable=True)
        if start == 0:
            again = generate_chunk(model, tokenizer, queries[:8], 0, seeds=SEEDS)
            if records != again:
                raise ValueError('Identical-seed generation replay did not reproduce exactly')
            write_json(FRESH / 'GENERATOR_REPLAY.json', dict(queries=8, exact=True,
                       target_access=False, seeds=SEEDS, test_used=False), immutable=True)
        if (start + 8) % 128 == 0 or start + 8 >= len(queries):
            print(f'Four-seed generation {min(start + 8, len(queries))}/{len(queries)} in {time.perf_counter() - began:.1f}s', flush=True)
            write_json(FRESH / 'GENERATION_STATUS.json', dict(completed=min(start + 8, len(queries)), total=len(queries),
                       elapsed_session_seconds=time.perf_counter() - began, seeds=SEEDS))
    del model
    torch.cuda.empty_cache()
    outputs = {str(path.relative_to(ROOT)): sha(path) for path in sorted(destination.glob('*.json'))}
    write_json(FRESH / 'GENERATION_COMPLETE.json', dict(completed=True, queries=len(queries), seeds=SEEDS,
               outputs=outputs, protocol_sha256=sha(FRESH / 'PROTOCOL.json')), immutable=True)


def retrieval():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from build_rewrite_utility_dataset import encode
    from run_t5_seed_uq_pilot import base_features
    p = freeze()
    if (FRESH / 'RETRIEVAL_COMPLETE.json').exists():
        check_hashes(read_json(FRESH / 'RETRIEVAL_COMPLETE.json')['outputs'])
        return
    check_hashes(read_json(FRESH / 'GENERATION_COMPLETE.json')['outputs'])
    rows = [row for path in sorted((FRESH / 'generation').glob('*.json')) for row in read_json(path)['rows']]
    queries = read_json(FRESH / 'queries.json')
    bygen = {r['query_index']: r for r in rows}
    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER, local_files_only=True)
    model = AutoModel.from_pretrained(RETRIEVER, local_files_only=True).to('cuda').eval()
    items = []
    began = time.perf_counter()
    for block in p['corpus']:
        path = FRESH / f'observations/block{block["block"]}_retrieval.pkl'
        if path.exists():
            items.extend(load(path))
            continue
        corpus = load(ROOT / block['path'])
        codes = [r['code_text'] for r in corpus]
        current = [q for q in queries if q['block'] == block['block']]
        unions = [union_draws(bygen[q['query_index']]) for q in current]
        cv = encode(model, tokenizer, codes, 48, 256, 'cuda')
        qv = encode(model, tokenizer, [q['query'] for q in current], 48, 256, 'cuda')
        all_candidates = [c['text'] for _, cs, _ in unions for c in cs]
        av = encode(model, tokenizer, all_candidates, 48, 256, 'cuda') if all_candidates else np.empty((0, cv.shape[1]))
        rebuilt, offset = [], 0
        for local, (q, (gen, candidates, appearances)) in enumerate(zip(current, unions)):
            original = cv @ qv[local]
            otop = stable_top(original)
            vv = av[offset:offset + len(candidates)]; offset += len(candidates)
            scores = [cv @ v for v in vv]
            tops = [stable_top(s) for s in scores]
            xx = np.array([base_features(q, c, qv[local], v, original, s, otop, top)
                           for c, v, s, top in zip(candidates, vv, scores, tops)]).reshape(-1, 34)
            e = dict(query_index=q['query_index'], query=q['query'], block=q['block'],
                     x=xx, seed_x=sampling_features(gen, vv, tops, appearances),
                     original_top100=otop, candidate_top100=tops, candidates=candidates,
                     seed_appearances=appearances)
            reject_gold(e)
            rebuilt.append(e)
        save(path, rebuilt)
        np.save(FRESH / f'observations/block{block["block"]}_code_vectors.npy', cv)
        np.save(FRESH / f'observations/block{block["block"]}_query_vectors.npy', qv)
        np.save(FRESH / f'observations/block{block["block"]}_candidate_vectors.npy', av)
        items.extend(rebuilt)
        print(f'Retrieved block {block["block"]}: {len(current)} queries, {len(all_candidates)} candidates, {time.perf_counter() - began:.1f}s', flush=True)
    del model
    torch.cuda.empty_cache()
    save(FRESH / 'retrieval_items.pkl', items)
    write_json(FRESH / 'RETRIEVAL_COMPLETE.json', dict(completed=True, queries=len(items),
               candidates=sum(len(e['x']) for e in items), empty_queries=sum(not len(e['x']) for e in items),
               outputs={str(path.relative_to(ROOT)): sha(path) for path in sorted((FRESH / 'observations').glob('*'))}
                        | {str((FRESH / 'retrieval_items.pkl').relative_to(ROOT)): sha(FRESH / 'retrieval_items.pkl')},
               labels_opened=False), immutable=True)


def relevance():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from experiments.strong_flow_uq_20261014 import build_evidence as ev
    p = freeze()
    if (FRESH / 'EVIDENCE_COMPLETE.json').exists():
        check_hashes(read_json(FRESH / 'EVIDENCE_COMPLETE.json')['outputs'])
        return
    items = load(FRESH / 'retrieval_items.pkl')
    model = AutoModel.from_pretrained(SEMANTIC, local_files_only=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(SEMANTIC, local_files_only=True)
    projection = torch.tensor(np.random.default_rng(12012).normal(size=(768, 64)) / 8,
                              dtype=torch.float32, device='cuda')
    began = time.perf_counter()
    output = []
    for block in p['corpus']:
        current = [e for e in items if e['block'] == block['block']]
        path = FRESH / f'observations/block{block["block"]}_relevance.pkl'
        if path.exists():
            output.extend(load(path)); continue
        corpus = load(ROOT / block['path'])
        codes = [r['code_text'] for r in corpus]
        unions = ev.unions(current)
        docs = np.array(sorted({int(d) for ids in unions for d in ids}), int)
        lookup = {int(d): i for i, d in enumerate(docs)}
        qt, qm, qmean = ev.project_encode(model, tokenizer, [e['query'] for e in current], 64, projection)
        dt, dm, dmean = ev.project_encode(model, tokenizer, [codes[d] for d in docs], 128, projection)
        index, termsets, idf = ev.lexical_index(codes)
        rows = []
        for i, (e, ids) in enumerate(zip(current, unions)):
            signals = ev.signals_for(e['query'], ids, qt[i], qm[i], qmean[i], dt, dm, dmean, lookup,
                                     index, termsets, idf)
            z = dict(query_index=e['query_index'], doc_ids=ids, doc_signals=signals,
                     x=ev.extras_for(e, ids, signals))
            view = dict(e, strong_evidence=z,
                        ce_query_length=len(tokenizer(e['query'], add_special_tokens=False,
                                                     truncation=True, max_length=64)['input_ids']))
            # Only IDs observed in plans are needed, but complete code-only lengths are label-free.
            view['ce_code_lengths'] = {d: len(x) for d, x in enumerate(tokenizer(codes,
                      add_special_tokens=False, truncation=True, max_length=160)['input_ids'])} if i == 0 else rows[0]['ce_code_lengths']
            reject_gold(view)
            rows.append(view)
        save(path, rows)
        output.extend(rows)
        print(f'Relevance block {block["block"]}: {len(rows)} queries, {len(docs)} observable documents, {time.perf_counter() - began:.1f}s', flush=True)
    del model
    torch.cuda.empty_cache()
    save(FRESH / 'observable_items.pkl', output)
    write_json(FRESH / 'EVIDENCE_COMPLETE.json', dict(completed=True, queries=len(output), labels_opened=False,
               outputs={str((FRESH / 'observable_items.pkl').relative_to(ROOT)): sha(FRESH / 'observable_items.pkl')}
               | {str(path.relative_to(ROOT)): sha(path) for path in sorted((FRESH / 'observations').glob('*_relevance.pkl'))}), immutable=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('inventory', 'freeze', 'generate', 'retrieve', 'relevance'), required=True)
    args = ap.parse_args()
    require_runtime(gpu=args.stage != 'inventory')
    with threadpool_limits(limits=2):
        {'inventory': inventory, 'freeze': freeze, 'generate': generate,
         'retrieve': retrieval, 'relevance': relevance}[args.stage]()


if __name__ == '__main__':
    main()
