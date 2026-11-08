from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata, t


def rr(order, target):
    if len(order) != 100 or len(set(order)) != 100:
        raise ValueError("Expected 100 unique document identities")
    return next((1.0 / rank for rank, doc in enumerate(order, 1) if doc == target), 0.0)


def uniform_max(values, k=3):
    n = len(values)
    if n < 1 or k < 1:
        raise ValueError("A nonempty position pool is required")
    k = min(k, n)
    return math.fsum(value * math.comb(index, k - 1) / math.comb(n, k)
                     for index, value in enumerate(sorted(values)) if index >= k - 1)


def correlation(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1])


def summarize(values, groups, *, repetitions=10000, seed=20261020, family=8):
    values = np.asarray(values, dtype=np.float64)
    labels = sorted(set(groups))
    if len(values) != len(groups) or len(labels) < 2 or not np.isfinite(values).all():
        raise ValueError("Invalid grouped outcomes")
    index = {label: i for i, label in enumerate(labels)}
    codes = np.array([index[g] for g in groups])
    counts = np.bincount(codes).astype(float)
    sums = np.bincount(codes, weights=values)
    mean = float(values.mean())
    se = float(np.sqrt(len(labels) / (len(labels) - 1) * np.square(sums - counts * mean).sum() / len(values)**2))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), (repetitions, len(labels)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    critical = float(t.ppf(.975, len(labels) - 1))
    family_critical = float(t.ppf(1 - .05 / (2 * family), len(labels) - 1))
    return {
        "mean": mean, "queries": len(values), "groups": len(labels), "se_cluster": se,
        "ci95_group_bootstrap": np.quantile(boot, [.025, .975]).tolist(),
        "ci95_cluster_t": [mean - critical * se, mean + critical * se],
        "eight_effect_family95_bonferroni_bootstrap": np.quantile(boot, [.05 / (2 * family), 1 - .05 / (2 * family)]).tolist(),
        "eight_effect_family95_bonferroni_t": [mean - family_critical * se, mean + family_critical * se],
    }
