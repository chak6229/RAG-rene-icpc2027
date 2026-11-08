from __future__ import annotations

import math

import numpy as np
from scipy.stats import t

from .common import FAMILY_SIZE, POLICIES, REPETITIONS, SEED


def order_candidates(pool, policy):
    positions = [c['position'] for c in pool]
    if not pool or len(set(positions)) != len(positions):
        raise ValueError('Nonempty unique candidate positions required')
    if not all(math.isfinite(c[key]) for c in pool for key in ('full_vocab_entropy', 'label_mean_log_probability')):
        raise ValueError('Nonfinite selector score')
    if policy == 'ENTR':
        key = lambda c: (c['full_vocab_entropy'], c['position'])
    elif policy == 'PROB':
        key = lambda c: (-c['label_mean_log_probability'], c['position'])
    else:
        raise ValueError('Unknown policy')
    return [c['position'] for c in sorted(pool, key=key)]


def exact_uniform_max(values, k):
    if not values or k < 1:
        raise ValueError('Invalid pool/count')
    n, k = len(values), min(k, len(values))
    denominator = math.comb(n, k)
    return math.fsum(value * math.comb(i, k - 1) / denominator
                     for i, value in enumerate(sorted(values)) if i >= k - 1)


def factorial(scores, orders, baseline):
    if not scores or set(orders) != {'ENTR', 'PROB'}:
        raise ValueError('Missing scores/selector')
    if any(len(order) != len(scores) or set(order) != set(scores) for order in orders.values()):
        raise ValueError('Score/position mismatch')
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in [baseline, *scores.values()]):
        raise ValueError('Invalid RR')
    values = {'NoRewrite': float(baseline)}
    uniform = math.fsum(scores.values()) / len(scores)
    for policy in POLICIES:
        for k in (1, 2, 3):
            if policy == 'RAND_exact':
                mean, best = uniform, exact_uniform_max(list(scores.values()), k)
            else:
                selected = [scores[p] for p in orders[policy][:min(k, len(scores))]]
                mean, best = math.fsum(selected) / len(selected), max(selected)
            values[f'{policy}_k{k}_mean'] = mean
            values[f'{policy}_k{k}_max'] = best
    contrasts = {f'delta_{name}': value - baseline for name, value in values.items() if name != 'NoRewrite'}
    for policy in ('ENTR', 'PROB'):
        for k in (1, 2, 3):
            for a in ('mean', 'max'):
                contrasts[f'S_{policy}_k{k}_{a}'] = values[f'{policy}_k{k}_{a}'] - values[f'RAND_exact_k{k}_{a}']
        contrasts[f'{policy}_B'] = contrasts[f'S_{policy}_k1_mean'] - contrasts[f'S_{policy}_k3_mean']
        contrasts[f'{policy}_G'] = contrasts[f'S_{policy}_k3_mean'] - contrasts[f'S_{policy}_k3_max']
        contrasts[f'{policy}_J'] = contrasts[f'S_{policy}_k1_mean'] - contrasts[f'S_{policy}_k3_max']
        if abs(contrasts[f'{policy}_B'] + contrasts[f'{policy}_G'] - contrasts[f'{policy}_J']) > 1e-12:
            raise AssertionError('B+G != J')
    for k in (1, 2, 3):
        for a in ('mean', 'max'):
            contrasts[f'ENTR_minus_PROB_k{k}_{a}'] = values[f'ENTR_k{k}_{a}'] - values[f'PROB_k{k}_{a}']
    for c in ('B', 'G', 'J'):
        contrasts[f'ENTR_minus_PROB_{c}'] = contrasts[f'ENTR_{c}'] - contrasts[f'PROB_{c}']
    return values, contrasts


def paired_difference(local, base):
    if (local['query_id'], local['group'], local['positions']) != (base['query_id'], base['group'], base['positions']):
        raise ValueError('Mismatched weight pair')
    if local['values']['NoRewrite'] != base['values']['NoRewrite']:
        raise ValueError('Original control differs')
    out = {'domain': local['domain'], 'view': 'base_minus_local', 'query_id': local['query_id'],
           'group': local['group'], 'positions': local['positions']}
    for section in ('values', 'contrasts'):
        if set(local[section]) != set(base[section]):
            raise ValueError('Weight metric keys differ')
        out[section] = {name: base[section][name] - local[section][name] for name in local[section]}
    return out


def summarize(matrix, names, groups, family_names):
    x = np.asarray(matrix, dtype=np.float64)
    labels = sorted(set(groups))
    lookup = {label: i for i, label in enumerate(labels)}
    codes = np.array([lookup[g] for g in groups])
    if x.shape != (len(groups), len(names)) or len(labels) < 2 or len(set(names)) != len(names) or not np.isfinite(x).all():
        raise ValueError('Invalid grouped matrix')
    sizes = np.bincount(codes, minlength=len(labels)).astype(float)
    sums = np.zeros((len(labels), len(names)))
    np.add.at(sums, codes, x)
    means = x.mean(0)
    se = np.sqrt(len(labels) / (len(labels) - 1) * np.square(sums - sizes[:, None] * means).sum(0) / len(x)**2)
    # A single draw stream per domain is shared across weights, selectors, and differences.
    rng = np.random.default_rng(SEED)
    boot = np.empty((REPETITIONS, len(names)))
    for start in range(0, REPETITIONS, 16):
        draw = rng.integers(0, len(labels), (min(16, REPETITIONS - start), len(labels)))
        boot[start:start + len(draw)] = sums[draw].sum(1) / sizes[draw].sum(1)[:, None]
    q = np.quantile(boot, [.025, .975, .025 / FAMILY_SIZE, 1 - .025 / FAMILY_SIZE], axis=0)
    c = float(t.ppf(.975, len(labels) - 1))
    fc = float(t.ppf(1 - .025 / FAMILY_SIZE, len(labels) - 1))
    changed = np.zeros((len(labels), len(names)), dtype=bool)
    np.logical_or.at(changed, codes, np.abs(x) > 1e-12)
    summary = {}
    for i, name in enumerate(names):
        r = {'mean': float(means[i]), 'se_cluster': float(se[i]), 'queries': len(x), 'groups': len(labels),
             'changed_queries': int(np.count_nonzero(np.abs(x[:, i]) > 1e-12)),
             'groups_with_nonzero': int(changed[:, i].sum()),
             'ci95_cluster_t': [float(means[i] - c * se[i]), float(means[i] + c * se[i])],
             'ci95_group_bootstrap': q[:2, i].tolist()}
        if name in family_names:
            r['family24_ci_cluster_t'] = [float(means[i] - fc * se[i]), float(means[i] + fc * se[i])]
            r['family24_ci_group_bootstrap'] = q[2:, i].tolist()
            lo = min(r['family24_ci_cluster_t'][0], r['family24_ci_group_bootstrap'][0])
            hi = max(r['family24_ci_cluster_t'][1], r['family24_ci_group_bootstrap'][1])
            r['direction'] = 'positive' if lo > 0 and r['groups_with_nonzero'] >= 20 else 'negative' if hi < 0 and r['groups_with_nonzero'] >= 20 else 'unresolved'
        summary[name] = r
    return summary
