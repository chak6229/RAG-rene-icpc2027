from __future__ import annotations

import numpy as np
from scipy.stats import norm, t


def grouped_arrays(matrix, groups):
    x = np.asarray(matrix, dtype=np.float64)
    labels = sorted(set(groups))
    lookup = {label: i for i, label in enumerate(labels)}
    codes = np.array([lookup[g] for g in groups])
    if x.ndim != 2 or len(x) != len(groups) or len(labels) < 2 or not np.isfinite(x).all():
        raise ValueError('Invalid cluster matrix')
    sizes = np.bincount(codes, minlength=len(labels)).astype(float)
    sums = np.zeros((len(labels), x.shape[1]))
    np.add.at(sums, codes, x)
    return x, labels, codes, sizes, sums


def summarize(matrix, names, groups, *, family_names=(), family_size=12,
              repetitions=10000, seed=202610203):
    x, labels, codes, sizes, sums = grouped_arrays(matrix, groups)
    if x.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('Contrast names mismatch')
    mean = x.mean(0)
    se = np.sqrt(len(labels) / (len(labels) - 1) * np.square(sums - sizes[:, None] * mean).sum(0) / len(x)**2)
    rng = np.random.default_rng(seed)
    boot = np.empty((repetitions, len(names)))
    for start in range(0, repetitions, 16):
        draws = rng.integers(0, len(labels), (min(16, repetitions - start), len(labels)))
        boot[start:start + len(draws)] = sums[draws].sum(1) / sizes[draws].sum(1)[:, None]
    quant = np.quantile(boot, [.025, .975, .025 / family_size, 1 - .025 / family_size], axis=0)
    c = t.ppf(.975, len(labels) - 1)
    fc = t.ppf(1 - .025 / family_size, len(labels) - 1)
    changed = np.zeros((len(labels), len(names)), dtype=bool)
    np.logical_or.at(changed, codes, np.abs(x) > 1e-12)
    out = {}
    for i, name in enumerate(names):
        r = {'mean': float(mean[i]), 'se_cluster': float(se[i]),
             'queries': len(x), 'groups': len(labels), 'changed_queries': int(np.count_nonzero(np.abs(x[:, i]) > 1e-12)),
             'groups_with_nonzero': int(changed[:, i].sum()),
             'ci95_cluster_t': [float(mean[i] - c * se[i]), float(mean[i] + c * se[i])],
             'ci95_group_bootstrap': quant[:2, i].tolist()}
        if name in family_names:
            r.update(family12_ci_cluster_t=[float(mean[i] - fc * se[i]), float(mean[i] + fc * se[i])],
                     family12_ci_group_bootstrap=quant[2:, i].tolist())
            lo = min(r['family12_ci_cluster_t'][0], r['family12_ci_group_bootstrap'][0])
            hi = max(r['family12_ci_cluster_t'][1], r['family12_ci_group_bootstrap'][1])
            r['direction'] = 'positive' if lo > 0 and r['groups_with_nonzero'] >= 20 else 'negative' if hi < 0 and r['groups_with_nonzero'] >= 20 else 'unresolved'
        out[name] = r
    return out


def precision_scenario(values, groups, *, family_size=12):
    x, labels, _, sizes, sums = grouped_arrays(np.asarray(values)[:, None], groups)
    mean = x.mean()
    se = float(np.sqrt(len(labels) / (len(labels) - 1) * np.square(sums[:, 0] - sizes * mean).sum() / len(x)**2))
    critical = float(t.ppf(1 - .025 / family_size, len(labels) - 1))
    return {'queries': len(x), 'groups': len(labels), 'old_se': se,
            'family12_halfwidth_if_se_transfers': critical * se,
            'mde80_normal_if_se_transfers': (float(norm.ppf(1 - .025 / family_size)) + float(norm.ppf(.8))) * se,
            'mde90_normal_if_se_transfers': (float(norm.ppf(1 - .025 / family_size)) + float(norm.ppf(.9))) * se,
            'assumption': 'Old retriever variance transferred to new model; scenario, not a power guarantee',
            'fixed_sample_no_adaptive_expansion': True}
