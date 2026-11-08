"""Independent Fraction/CDF and scalar-statistics replay; no production imports."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import t


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**23), b''):
            h.update(block)
    return h.hexdigest()


def tid(text):
    return hashlib.sha256(text.encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def fraction_rr(rank, target):
    check(len(rank) == 100 and len(set(rank)) == 100, 'Full unique Top100 required')
    return Fraction(1, rank.index(target) + 1) if target in rank else Fraction(0)


def random_max_cdf(values, count):
    n, k = len(values), min(count, len(values))
    denominator = math.comb(n, k)
    probability_before, expectation = Fraction(0), Fraction(0)
    for value in sorted(set(values)):
        available = sum(v <= value for v in values)
        cumulative = Fraction(math.comb(available, k), denominator) if available >= k else Fraction(0)
        expectation += value * (cumulative - probability_before)
        probability_before = cumulative
    check(probability_before == 1, 'CDF not normalized')
    return expectation


def independent_fields(scores, orders, original):
    values = {'NoRewrite': original}
    all_rr = list(scores.values())
    for policy in ('ENTR', 'PROB', 'RAND_exact'):
        for k in (1, 2, 3):
            if policy == 'RAND_exact':
                mean = sum(all_rr, Fraction(0)) / len(all_rr)
                best = random_max_cdf(all_rr, k)
            else:
                chosen = [scores[p] for p in orders[policy][:k]]
                mean = sum(chosen, Fraction(0)) / len(chosen)
                best = max(chosen)
            values[f'{policy}_k{k}_mean'] = mean
            values[f'{policy}_k{k}_max'] = best
    contrasts = {'delta_' + key: value - original for key, value in values.items() if key != 'NoRewrite'}
    for policy in ('ENTR', 'PROB'):
        for k in (1, 2, 3):
            for a in ('mean', 'max'):
                contrasts[f'S_{policy}_k{k}_{a}'] = values[f'{policy}_k{k}_{a}'] - values[f'RAND_exact_k{k}_{a}']
        # Algebraically expanded separately from the production S-path implementation.
        contrasts[f'{policy}_B'] = values[f'{policy}_k1_mean'] - values[f'{policy}_k3_mean']
        contrasts[f'{policy}_G'] = (values[f'RAND_exact_k3_max'] - values['RAND_exact_k3_mean']) - (values[f'{policy}_k3_max'] - values[f'{policy}_k3_mean'])
        contrasts[f'{policy}_J'] = contrasts[f'{policy}_B'] + contrasts[f'{policy}_G']
    for k in (1, 2, 3):
        for a in ('mean', 'max'):
            contrasts[f'ENTR_minus_PROB_k{k}_{a}'] = values[f'ENTR_k{k}_{a}'] - values[f'PROB_k{k}_{a}']
    for component in ('B', 'G', 'J'):
        contrasts['ENTR_minus_PROB_' + component] = contrasts['ENTR_' + component] - contrasts['PROB_' + component]
    return {'values': values, 'contrasts': contrasts}


def independent_statistics(matrix, names, labels, design):
    # Dictionary group membership, explicit centering, and bootstrap frequency weights.
    membership = defaultdict(list)
    for i, group in enumerate(labels):
        membership[group].append(i)
    groups = sorted(membership)
    n, g = len(labels), len(groups)
    sums = np.stack([np.array([math.fsum(matrix[i, j] for i in membership[group]) for j in range(len(names))]) for group in groups])
    sizes = np.array([len(membership[group]) for group in groups])
    mean = np.array([math.fsum(matrix[:, j]) / n for j in range(len(names))])
    se = np.array([math.sqrt(g / (g - 1) * math.fsum((sums[i, j] - sizes[i] * mean[j])**2 for i in range(g)) / n**2) for j in range(len(names))])
    repetitions = design['statistics']['repetitions']
    rng = np.random.default_rng(design['statistics']['seed'])
    boot = np.empty((repetitions, len(names)))
    for b in range(repetitions):
        counts = np.bincount(rng.integers(0, g, g), minlength=g)
        boot[b] = counts @ sums / (counts @ sizes)
    f = design['primary_family']['size']
    q = np.quantile(boot, [.025, .975, .025 / f, 1 - .025 / f], axis=0)
    crit, family_crit = float(t.ppf(.975, g - 1)), float(t.ppf(1 - .025 / f, g - 1))
    out = {}
    for j, name in enumerate(names):
        nonzero_groups = sum(any(abs(matrix[i, j]) > 1e-12 for i in membership[group]) for group in groups)
        ci_t = [mean[j] - family_crit * se[j], mean[j] + family_crit * se[j]]
        ci_b = q[2:, j].tolist()
        direction = 'positive' if min(ci_t[0], ci_b[0]) > 0 and nonzero_groups >= 20 else 'negative' if max(ci_t[1], ci_b[1]) < 0 and nonzero_groups >= 20 else 'unresolved'
        out[name] = {'mean': float(mean[j]), 'se_cluster': float(se[j]), 'queries': n, 'groups': g,
                     'changed_queries': int(np.count_nonzero(np.abs(matrix[:, j]) > 1e-12)), 'groups_with_nonzero': nonzero_groups,
                     'ci95_cluster_t': [float(mean[j] - crit * se[j]), float(mean[j] + crit * se[j])],
                     'ci95_group_bootstrap': q[:2, j].tolist(), 'family24_ci_cluster_t': [float(v) for v in ci_t],
                     'family24_ci_group_bootstrap': ci_b, 'direction': direction}
    return out


def audit():
    root = Path(__file__).resolve().parents[2]
    out = root / 'results/rene_t5_weight_factors_20261020'
    old = root / 'results/rene_envelope_completion_20261020'
    bridge = root / 'results/rene_ssqr_bridge_20261019'
    design, result = read(out / 'DESIGN.json'), read(out / 'RESULTS.json')
    counts, maximum = Counter(), defaultdict(float)

    def near(actual, expected, category, tolerance=2e-12):
        error = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
        maximum[category] = max(maximum[category], error)
        counts[category] += int(np.asarray(actual).size)
        check(error < tolerance, category + ' mismatch')

    for item in design['bound_files'] + design['protected_files']:
        check(digest(root / item['path']) == item['sha256'], 'Bound/protected file changed')
        counts['bound_protected_files'] += 1
    check(result['design_sha256'] == digest(out / 'DESIGN.json'), 'Design digest mismatch')
    cache_rows = rows(old / 'candidate_RR.jsonl')
    cache = {(r['domain'], r['rewriter'], r['query_id'], r['position']): r for r in cache_rows}
    saved_rows = rows(out / 'per_query.jsonl')
    saved = {(r['domain'], r['view'], r['query_id']): r for r in saved_rows}
    paired_rows = rows(out / 'paired_per_query.jsonl')
    saved_pair = {(r['domain'], r['query_id']): r for r in paired_rows}
    check(len(cache) == len(cache_rows) == 18680 and len(saved) == len(saved_rows) == 2048 and len(saved_pair) == len(paired_rows) == 1024, 'Export scope mismatch')
    all_computed = {}
    for domain in ('source', 'cosqa'):
        sample = design['samples'][domain]
        ids = sample['query_ids']
        queries = {r['query_id']: r['query'] for r in rows(bridge / f'data/{domain}/queries.jsonl')}
        target = {r['query_id']: r['doc_id'] for r in rows(bridge / f'data/private/{domain}_qrels.jsonl')}
        computed = {}
        for arm in ('local', 't5base'):
            candidates_path = root / f'results/rene_ssqr_rewriter_envelope_20261020/candidates/{domain}/t5base.jsonl' if arm == 't5base' else bridge / 'candidates/source/baseline.jsonl' if domain == 'source' else root / 'results/rene_cosqa_development_20261019/candidates.jsonl'
            ranks_path = old / f'rankings/{domain}/Top100.jsonl' if arm == 't5base' else bridge / f'rankings/{domain}/CodeBERT.jsonl'
            ranks = {r['text_id']: r for r in rows(ranks_path)}
            pools = defaultdict(list)
            for c in rows(candidates_path):
                if c['query_id'] in set(ids):
                    pools[c['query_id']].append(c)
            computed[arm] = []
            for qid in ids:
                pool = sorted(pools[qid], key=lambda c: c['position'])
                check([c['position'] for c in pool] == list(range(1, len(queries[qid].split()) + 1)), 'Pool scope mismatch')
                orders = {'ENTR': [c['position'] for c in sorted(pool, key=lambda c: (c['full_vocab_entropy'], c['position']))],
                          'PROB': [c['position'] for c in sorted(pool, key=lambda c: (-c['label_mean_log_probability'], c['position']))]}
                scores = {}
                for c in pool:
                    key = domain, arm, qid, c['position']
                    check(cache[key]['text_id'] == tid(c['rewrite']) and cache[key]['full_vocab_entropy'] == c['full_vocab_entropy'], 'Candidate identity differs')
                    scores[c['position']] = fraction_rr(ranks[tid(c['rewrite'])]['ranked_doc_ids'], target[qid])
                    near(cache[key]['rr'], float(scores[c['position']]), 'Fraction_candidate_RR')
                original = fraction_rr(ranks[tid(queries[qid])]['ranked_doc_ids'], target[qid])
                fields = independent_fields(scores, orders, original)
                computed[arm].append(fields)
                s = saved[domain, arm, qid]
                check(s['orders'] == orders and s['group'] == sample['groups'][qid] and s['positions'] == len(pool), 'Exported query/actions differ')
                for section in fields:
                    check(set(s[section]) == set(fields[section]), 'Factor keys differ')
                    for key in fields[section]:
                        near(s[section][key], float(fields[section][key]), 'Fraction_factor_fields')
                near(s['values']['NoRewrite'], float(original), 'Fraction_original_RR')
                counts['query_rows'] += 1
        computed['base_minus_local'] = []
        for i, qid in enumerate(ids):
            fields = {section: {key: computed['t5base'][i][section][key] - computed['local'][i][section][key] for key in computed['local'][i][section]} for section in ('values', 'contrasts')}
            check(fields['values']['NoRewrite'] == 0, 'Original controls not paired')
            s = saved_pair[domain, qid]
            check(s['group'] == sample['groups'][qid], 'Paired group mismatch')
            for section in fields:
                check(set(s[section]) == set(fields[section]), 'Paired keys differ')
                for key in fields[section]:
                    near(s[section][key], float(fields[section][key]), 'Fraction_paired_fields')
            computed['base_minus_local'].append(fields)
        for view, qrows in computed.items():
            for section in ('values', 'contrasts'):
                for key in qrows[0][section]:
                    near(result['domains'][domain]['views'][view][section][key]['mean'], float(sum((r[section][key] for r in qrows), Fraction(0)) / len(qrows)), 'all_summary_means')
        names = design['primary_family']['per_domain_names']
        matrix = np.array([[float(computed[view][i][section][key]) for view, section, key in (name.split('::') for name in names)] for i in range(len(ids))])
        independent = independent_statistics(matrix, names, [sample['groups'][qid] for qid in ids], design)
        for name, r in independent.items():
            view, section, key = name.split('::')
            actual = result['domains'][domain]['views'][view][section][key]
            for field, value in r.items():
                if field == 'direction':
                    check(actual[field] == value, 'Direction mismatch')
                else:
                    near(actual[field], value, 'independent_primary_statistics')
            counts['primary_intervals'] += 1
        all_computed[domain] = computed
        print({'stage': 'independent_audit', 'domain': domain, 'primary_intervals': len(names)}, flush=True)
    check(counts['primary_intervals'] == design['primary_family']['size'] == 24, 'Primary family incomplete')
    acceptance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
                  'independent_of_production_metrics_and_loaders': True, 'random_max_method': 'exact Fraction CDF over RR levels',
                  'statistics_method': 'explicit group membership, scalar centering, bootstrap frequency weights',
                  'counts': dict(counts), 'max_errors': dict(maximum),
                  'design_sha256': digest(out / 'DESIGN.json'), 'results_sha256': digest(out / 'RESULTS.json'),
                  'prior_accepted_rankings_reused': True, 'new_model_inference': False}
    with (out / 'AUDIT.json').open('x', encoding='utf-8') as stream:
        json.dump(acceptance, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(acceptance, flush=True)


if __name__ == '__main__':
    audit()
