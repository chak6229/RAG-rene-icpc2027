"""Independent reference calculations; no delegation or experimental tuning."""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np
from scipy.stats import t

from .common import COMPONENTS, MODEL, NAME, OUT, ROOT, now, progress, read, record, require, rows, sha, verify_design, verify_seal, write


def load_qrels(cfg):
    result = defaultdict(set)
    for r in rows(ROOT / cfg['qrels']):
        result[r['query_id']].update(r['relevant_doc_ids'] if 'relevant_doc_ids' in r else [r['doc_id']])
    return result


def before_effects():
    design = verify_design(private=True)
    verify_seal('ACTIONS_SEALED.json')
    verify_seal('RANKINGS_SEALED.json')
    require(not (OUT / 'OUTCOME_OPENING.json').exists(), 'Effects already opened before audit')
    if (OUT / 'PREOPEN_AUDIT.json').exists():
        require(read(OUT / 'PREOPEN_AUDIT.json')['passed'], 'Earlier preopening audit failed')
        return
    import torch
    from threadpoolctl import threadpool_limits
    from transformers import RobertaModel, RobertaTokenizer
    from .preflight import reference_encode
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = RobertaModel.from_pretrained(MODEL, local_files_only=True, attn_implementation='eager').float().cuda().eval()
    tokenizer = RobertaTokenizer.from_pretrained(MODEL, local_files_only=True)
    checks = {}
    for domain, cfg in design['samples'].items():
        corpus = list(rows(ROOT / cfg['corpus']))
        queries = list(rows(ROOT / cfg['queries']))
        ids = [r['doc_id'] for r in corpus]
        require(ids == sorted(set(ids)), 'Audit stable-sort requires unique ascending corpus IDs')
        relevant = load_qrels(cfg)
        require(set(relevant) == {q['query_id'] for q in queries}, 'Qrel/query coverage differs')
        require(all(v and v <= set(ids) for v in relevant.values()), 'Target outside corpus')
        actions = {r['query_id']: r for r in rows(ROOT / cfg['actions'])}
        pools = defaultdict(list)
        for r in rows(ROOT / cfg['candidates']):
            pools[r['query_id']].append(r)
        for q in queries:
            pool = pools[q['query_id']]
            for selector, key in [('ENTR', lambda r: (r['full_vocab_entropy'], r['position'])),
                                  ('PROB', lambda r: (-r['label_mean_log_probability'], r['position']))]:
                require(actions[q['query_id']]['orders'][selector] == [r['position'] for r in sorted(pool, key=key)], 'Independent selector disagrees')
        cache = OUT / f'cache/{domain}'
        texts = read(cache / 'texts.json')
        expected = list(dict.fromkeys([q['query'] for q in queries] + [r['rewrite'] for r in rows(ROOT / cfg['candidates'])]))
        require(texts == expected, 'Retrieval text coverage/order')
        cv, qv = np.load(cache / 'code_vectors.npy'), np.load(cache / 'query_vectors.npy')
        require(cv.shape == (len(corpus), 768) and qv.shape == (len(texts), 768), 'Embedding shape')
        require(np.isfinite(cv).all() and np.isfinite(qv).all(), 'Nonfinite embeddings')
        norm_error = max(float(np.max(np.abs(np.linalg.norm(a, axis=1) - 1))) for a in (cv, qv))
        require(norm_error <= 2e-6, 'Embedding normalization')
        stream = iter(rows(OUT / f'rankings/{domain}.jsonl'))
        ranking_count = score_count = 0
        score_error = 0.
        with threadpool_limits(limits=4):
            for start in range(0, len(texts), 256):
                values = qv[start:start + 256].dot(cv.T)
                # Independent stable sort: corpus is already ascending by doc ID.
                orders = np.argsort(-values, axis=1, kind='stable')[:, :100]
                for j, order in enumerate(orders):
                    saved = next(stream)
                    require(saved['text'] == texts[start + j], 'Ranking stream reordered')
                    require(saved['ranked_doc_ids'] == [ids[i] for i in order], 'Independent full-corpus ranking mismatch')
                    score_error = max(score_error, float(np.max(np.abs(np.asarray(saved['scores']) - values[j, order]))))
                    require(score_error <= 1e-7, 'Saved retrieval scores differ')
                    ranking_count += 1
                score_count += values.size
                if start % 8192 == 0:
                    progress('preopen_ranking_audit', domain=domain, rows_done=min(start + 256, len(texts)), rows_total=len(texts))
        require(next(stream, None) is None, 'Extra ranking rows')
        model_checks, max_model_error = {}, 0.
        for kind, data, vectors, length in [('query', texts, qv, 128), ('code', [r['code_text'] for r in corpus], cv, 256)]:
            # Fixed evenly spaced indices, independent of scores/labels.
            selected = np.linspace(0, len(data) - 1, 16, dtype=int).tolist()
            rv, _ = reference_encode(model, tokenizer, [data[i] for i in selected], length, code=kind == 'code')
            err = float(np.max(np.abs(rv - vectors[selected])))
            require(err <= 5e-6, 'Actual model reference forward disagrees')
            max_model_error = max(max_model_error, err)
            model_checks[kind] = {'indices': selected, 'max_abs_error': err, 'fresh_official_model_forward': True}
        checks[domain] = {'queries': len(queries), 'groups': len({r['group'] for r in actions.values()}),
            'candidate_positions': sum(map(len, pools.values())), 'documents': len(corpus),
            'full_rankings_checked': ranking_count, 'full_scores_recomputed': score_count,
            'top100_score_max_error': score_error, 'unit_norm_max_error': norm_error,
            'actual_model_checks': model_checks, 'actual_model_max_abs_error': max_model_error,
            'truncation': {kind: int(np.load(cache / f'{kind}_truncated.npy').sum()) for kind in ('query', 'code')}}
    del model
    torch.cuda.empty_cache()
    verify_design(private=True)
    write(OUT / 'PREOPEN_AUDIT.json', {'created_utc': now(), 'passed': True,
          'design_sha256': sha(OUT / 'DESIGN.json'), 'new_effects_computed': False,
          'qrels_used_for_coverage_only': True, 'checks': checks})
    print({'preopening_audit_passed': True, 'checks': checks}, flush=True)


def random_max_cdf(values, k):
    k = min(k, len(values))
    denominator = math.comb(len(values), k)
    total, below = Fraction(0), 0
    for value, count in sorted(Counter(values).items()):
        at = below + count
        total += value * Fraction((math.comb(at, k) if at >= k else 0) - (math.comb(below, k) if below >= k else 0), denominator)
        below = at
    return total


def independent_values(position_rr, orders, original):
    values = {'NoRewrite': original}
    uniform = sum(position_rr.values(), Fraction(0)) / len(position_rr)
    for k in (1, 2, 3):
        for selector in ('ENTR', 'PROB', 'RAND_exact'):
            if selector == 'RAND_exact':
                first = mean = uniform
                maximum = random_max_cdf(list(position_rr.values()), k)
            else:
                current = [position_rr[p] for p in orders[selector][:k]]
                first, mean, maximum = current[0], sum(current, Fraction(0)) / len(current), max(current)
            values.update({f'{selector}_k{k}_first': first, f'{selector}_k{k}_mean': mean, f'{selector}_k{k}_max': maximum})
    components = {}
    for p in ('ENTR', 'PROB'):
        b = (values[f'{p}_k1_mean'] - values['RAND_exact_k1_mean']) - (values[f'{p}_k3_mean'] - values['RAND_exact_k3_mean'])
        g = (values[f'{p}_k3_mean'] - values['RAND_exact_k3_mean']) - (values[f'{p}_k3_max'] - values['RAND_exact_k3_max'])
        components.update({f'{p}_B_budget_mean_1_to_3': b, f'{p}_G_aggregation_at3_mean_to_max': g, f'{p}_J_exact': b + g})
    return values, components


def independent_intervals(matrix, names, groups):
    # Group sums via explicit original-row memberships, rather than production add.at.
    x = np.asarray(matrix, dtype=np.float64)
    labels = sorted(set(groups))
    indices = defaultdict(list)
    for i, g in enumerate(groups):
        indices[g].append(i)
    sizes = np.array([len(indices[g]) for g in labels], dtype=float)
    sums = np.array([x[indices[g]].sum(0) for g in labels])
    mean = x.mean(0)
    residual = sums - sizes[:, None] * mean
    variance = (residual.T @ residual).diagonal() * len(labels) / (len(labels) - 1) / len(x)**2
    se = np.sqrt(variance)
    rng = np.random.default_rng(202610203)
    boot = np.empty((10000, len(names)))
    for start in range(0, 10000, 25):
        draws = rng.integers(0, len(labels), (min(25, 10000 - start), len(labels)))
        boot[start:start + len(draws)] = sums[draws].sum(1) / sizes[draws].sum(1)[:, None]
    quant = np.quantile(boot, [.025, .975, .025 / 12, 1 - .025 / 12], axis=0)
    out = {}
    for i, name in enumerate(names):
        c, fc = t.ppf(.975, len(labels) - 1), t.ppf(1 - .025 / 12, len(labels) - 1)
        out[name] = {'mean': float(mean[i]), 'se_cluster': float(se[i]),
                     'ci95_cluster_t': [mean[i] - c * se[i], mean[i] + c * se[i]],
                     'ci95_group_bootstrap': quant[:2, i],
                     'family12_ci_cluster_t': [mean[i] - fc * se[i], mean[i] + fc * se[i]],
                     'family12_ci_group_bootstrap': quant[2:, i]}
    return out


def after_effects():
    design = verify_design(private=True)
    verify_seal('RANKINGS_SEALED.json')
    result = read(OUT / 'RESULTS.json')
    ledger = {(r['domain'], r['query_id']): r for r in rows(OUT / 'per_query.jsonl')}
    require(len(ledger) == sum(c['queries_expected'] for c in design['samples'].values()), 'Per-query output coverage')
    audits = {}
    for domain, cfg in design['samples'].items():
        queries = list(rows(ROOT / cfg['queries']))
        actions = {r['query_id']: r for r in rows(ROOT / cfg['actions'])}
        rel = load_qrels(cfg)
        targets = defaultdict(list)
        for q in queries:
            targets[q['query']].append((q['query_id'], 0))
        for r in rows(ROOT / cfg['candidates']):
            targets[r['rewrite']].append((r['query_id'], r['position']))
        exact_rr = defaultdict(dict)
        rr_count = 0
        for ranking in rows(OUT / f'rankings/{domain}.jsonl'):
            ranks = {doc: i + 1 for i, doc in enumerate(ranking['ranked_doc_ids'])}
            for qid, p in targets[ranking['text']]:
                first = min((ranks[d] for d in rel[qid] if d in ranks), default=0)
                exact_rr[qid][p] = Fraction(1, first) if first else Fraction(0)
                rr_count += 1
        independently = {}
        field_count, max_error = 0, 0.
        for q in queries:
            qid = q['query_id']
            original = exact_rr[qid].pop(0)
            values, components = independent_values(exact_rr[qid], actions[qid]['orders'], original)
            saved = ledger[domain, qid]
            for p, value in exact_rr[qid].items():
                err = abs(float(value) - saved['position_rr'][str(p)])
                require(err < 1e-14, 'Independent Fraction candidate RR mismatch')
                max_error = max(max_error, err)
                field_count += 1
            for name, value in values.items():
                err = abs(float(value) - saved['values'][name])
                require(err < 1e-13, 'Independent Fraction factorial value mismatch')
                max_error = max(max_error, err)
                field_count += 1
            for name, value in components.items():
                err = abs(float(value) - saved['contrasts'][name])
                require(err < 1e-13, 'Independent Fraction B/G mismatch')
                max_error = max(max_error, err)
                field_count += 1
            independently[qid] = components
        reference = {ret: {} for ret in ('CodeBERT', 'BM25')}
        for r in rows(ROOT / cfg['reference_rows']):
            if r.get('domain', domain) == domain and r['query_id'] in independently:
                reference[r['retriever']][r['query_id']] = r
        names = [f'{NAME}::{c}' for c in COMPONENTS]
        names += [f'{NAME}_minus_{ret}::{c}' for ret in reference for c in COMPONENTS]
        matrix = []
        for q in queries:
            qid = q['query_id']
            matrix.append([float(independently[qid][c]) for c in COMPONENTS] +
                [float(independently[qid][c]) - reference[ret][qid]['contrasts'][c] for ret in reference for c in COMPONENTS])
        groups = [actions[q['query_id']]['group'] for q in queries]
        summaries = independent_intervals(matrix, names, groups)
        statistic_error = 0.
        for name, expected in summaries.items():
            ret, component = name.split('::')
            if ret == NAME:
                saved = result['domains'][domain]['views'][NAME]['summaries'][component]
            else:
                saved = result['domains'][domain]['paired_differences'][ret.split('_minus_')[1]][component]
            for field, value in expected.items():
                err = float(np.max(np.abs(np.asarray(value) - np.asarray(saved[field]))))
                require(err < 2e-12, 'Independent primary statistic mismatch: ' + name + ':' + field)
                statistic_error = max(statistic_error, err)
                field_count += np.asarray(value).size
            lower = min(saved['family12_ci_cluster_t'][0], saved['family12_ci_group_bootstrap'][0])
            upper = max(saved['family12_ci_cluster_t'][1], saved['family12_ci_group_bootstrap'][1])
            expected_direction = 'positive' if lower > 0 and saved['groups_with_nonzero'] >= 20 else 'negative' if upper < 0 and saved['groups_with_nonzero'] >= 20 else 'unresolved'
            require(saved['direction'] == expected_direction, 'Primary decision mismatch')
        audits[domain] = {'queries': len(queries), 'Fraction_RR_comparisons': rr_count,
                         'independent_numerical_fields': field_count, 'max_factorial_error': max_error,
                         'max_primary_statistic_error': statistic_error, 'primary_intervals_independently_recomputed': len(names),
                         'random_max_uses_separate_CDF_formula': True}
        progress('post_effect_audit', domain=domain, passed=True, **audits[domain])
    verify_design(private=True)
    write(OUT / 'FINAL_ACCEPTANCE.json', {'created_utc': now(), 'passed': True,
          'design_sha256': sha(OUT / 'DESIGN.json'), 'preopen_audit': record(OUT / 'PREOPEN_AUDIT.json'),
          'results': record(OUT / 'RESULTS.json'), 'per_query': record(OUT / 'per_query.jsonl'),
          'protected_states_unchanged': True, 'new_confirmation': False, 'risk_certified': False,
          'audits': audits})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--after', action='store_true')
    args = parser.parse_args()
    after_effects() if args.after else before_effects()
