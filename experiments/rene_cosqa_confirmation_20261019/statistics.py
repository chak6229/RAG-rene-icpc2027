"""Paired connected-group inference; generation repetitions are never samples."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, t

from .common import require


def aggregate(values, groups):
    values = np.asarray(values, float)
    require(len(values) == len(groups) and np.isfinite(values).all() and len(values) > 0, "invalid metric vector")
    labels = sorted(set(groups))
    index = {g: i for i, g in enumerate(labels)}
    codes = np.array([index[g] for g in groups])
    return (np.bincount(codes, weights=values, minlength=len(labels)),
            np.bincount(codes, minlength=len(labels)).astype(float))


def influence_variance(sums, sizes):
    n = len(sums)
    require(n > 1 and np.all(sizes > 0), "independent groups required")
    mean = float(np.sum(sums) / np.sum(sizes))
    return n * n / (n - 1) * float(np.square(sums - mean * sizes).sum()) / float(sizes.sum() ** 2)


def design_factor(sizes):
    sizes = np.asarray(sizes, float)
    return float(len(sizes) * np.square(sizes).sum() / sizes.sum() ** 2)


def plan(values, groups, future_sizes, repetitions=10000):
    sums, sizes = aggregate(values, groups)
    n = len(sums)
    rng = np.random.default_rng(20261019)
    estimates = []
    for _ in range(0, repetitions, 250):
        idx = rng.integers(0, n, (min(250, repetitions - len(estimates)), n))
        ss, nn = sums[idx], sizes[idx]
        mm = ss.sum(1) / nn.sum(1)
        estimates.extend((n * n / (n - 1) * np.square(ss - mm[:, None] * nn).sum(1) / nn.sum(1) ** 2).tolist())
    upper = float(np.quantile(estimates, .95))
    require(upper > 0, "zero estimated variance cannot authorize a power claim")
    ratio = max(1., design_factor(future_sizes) / design_factor(sizes))
    capacity = len(future_sizes)
    sigma2 = upper * ratio
    z = float(t.isf(.025, capacity - 1))
    required = math.ceil((z + norm.ppf(.8)) ** 2 * sigma2 / .005 ** 2)
    ncp = .005 / math.sqrt(sigma2 / capacity)
    power = float(norm.sf(z - ncp) + norm.cdf(-z - ncp))
    return {"development_queries": int(sizes.sum()), "development_groups": n,
            "changed_development_groups": int(np.count_nonzero(np.abs(sums) > 1e-10)),
            "development_variance": influence_variance(sums, sizes),
            "variance_bootstrap_upper95": upper, "variance_bootstrap_repetitions": repetitions,
            "D_design_factor": design_factor(sizes), "H_design_factor": design_factor(future_sizes),
            "size_imbalance_multiplier_at_least1": ratio, "planning_variance": sigma2,
            "alpha": .05, "two_sided": True, "target_effect": .005, "target_power": .8,
            "required_groups": required, "available_groups": capacity,
            "planned_power_at_capacity": power,
            "mde80": float((z + norm.ppf(.8)) * math.sqrt(sigma2 / capacity)),
            "assumptions": "independent representative connected groups; D-to-H outcome variance transport; normal power approximation with t critical value; size sensitivity is not a universal variance bound"}


def interval(values, groups, repetitions=10000, seed=261019):
    sums, sizes = aggregate(values, groups)
    n = len(sums)
    mean = float(sums.sum() / sizes.sum())
    se = math.sqrt(influence_variance(sums, sizes) / n)
    half = float(t.isf(.025, n - 1)) * se
    rng = np.random.default_rng(seed)
    boot = []
    for offset in range(0, repetitions, 200):
        idx = rng.integers(0, n, (min(200, repetitions - offset), n))
        boot.extend((sums[idx].sum(1) / sizes[idx].sum(1)).tolist())
    p = float(2 * t.sf(abs(mean / se), n - 1)) if se > 0 else (1. if mean == 0 else 0.)
    return {"mean": mean, "queries": int(sizes.sum()), "groups": n,
            "changed_groups": int(np.count_nonzero(np.abs(sums) > 1e-10)),
            "standard_error_cluster": se, "ci95_cluster_t": [mean - half, mean + half],
            "ci95_group_bootstrap": np.quantile(boot, [.025, .975]).tolist(),
            "p_two_sided_cluster_t": p}


def ratio_interval(numerator, denominator, groups, repetitions=10000):
    a, sizes = aggregate(numerator, groups)
    b, other = aggregate(denominator, groups)
    require(np.array_equal(sizes, other), "ratio denominator population differs")
    if b.sum() <= 0:
        return {"value": None, "ci95_group_bootstrap": None, "reason": "zero denominator"}
    rng = np.random.default_rng(261020)
    boot = []
    for offset in range(0, repetitions, 200):
        idx = rng.integers(0, len(a), (min(200, repetitions - offset), len(a)))
        aa, bb = a[idx].sum(1), b[idx].sum(1)
        boot.extend((aa[bb > 0] / bb[bb > 0]).tolist())
    return {"value": float(a.sum() / b.sum()),
            "ci95_group_bootstrap": np.quantile(boot, [.025, .975]).tolist() if boot else None,
            "not_a_risk_certificate": True}
