from __future__ import annotations

import math

import numpy as np
from scipy.stats import t

CONTRASTS = {
    "a_ENTR_single_minus_uniform": {"ENTR_top1": 1, "Uniform_single_exact": -1},
    "b_ENTR_best3_minus_RAND20": {"ENTR_best3": 1, "RAND20_best3": -1},
    "j_ENTR_contract_difference": {"ENTR_top1": 1, "Uniform_single_exact": -1, "ENTR_best3": -1, "RAND20_best3": 1},
    "d_ENTR_single_minus_NoRewrite": {"ENTR_top1": 1, "NoRewrite": -1},
    "a_PROB_single_minus_uniform": {"PROB_top1": 1, "Uniform_single_exact": -1},
    "b_PROB_best3_minus_RAND20": {"PROB_best3": 1, "RAND20_best3": -1},
    "j_PROB_contract_difference": {"PROB_top1": 1, "Uniform_single_exact": -1, "PROB_best3": -1, "RAND20_best3": 1},
    "d_PROB_single_minus_NoRewrite": {"PROB_top1": 1, "NoRewrite": -1},
    "ENTR_minus_PROB_single": {"ENTR_top1": 1, "PROB_top1": -1},
    "ENTR_minus_PROB_best3": {"ENTR_best3": 1, "PROB_best3": -1},
    "ENTR_minus_PROB_contract_difference": {"ENTR_top1": 1, "PROB_top1": -1, "ENTR_best3": -1, "PROB_best3": 1},
    "ENTR_best3_minus_NoRewrite": {"ENTR_best3": 1, "NoRewrite": -1},
    "uniform_single_minus_NoRewrite": {"Uniform_single_exact": 1, "NoRewrite": -1},
    "RAND20_best3_minus_NoRewrite": {"RAND20_best3": 1, "NoRewrite": -1},
    "b_ENTR_best3_minus_uniform3_exact": {"ENTR_best3": 1, "Uniform_best3_exact": -1},
    "j_ENTR_contract_difference_exact_random3": {"ENTR_top1": 1, "Uniform_single_exact": -1, "ENTR_best3": -1, "Uniform_best3_exact": 1},
    "b_PROB_best3_minus_uniform3_exact": {"PROB_best3": 1, "Uniform_best3_exact": -1},
    "j_PROB_contract_difference_exact_random3": {"PROB_top1": 1, "Uniform_single_exact": -1, "PROB_best3": -1, "Uniform_best3_exact": 1},
    "RAND20_minus_uniform3_exact": {"RAND20_best3": 1, "Uniform_best3_exact": -1},
}


def rr(order: list[int], relevant: set[int]) -> float:
    if len(order) != 100 or len(set(order)) != 100:
        raise ValueError("Ranking must contain 100 distinct document identities")
    return next((1 / rank for rank, doc in enumerate(order, 1) if doc in relevant), 0.0)


def exact_random_max(values: list[float], k: int) -> float:
    """Uniform sampling of k distinct positions; duplicate texts retain position mass."""
    n = len(values)
    if n < 1 or k < 1:
        raise ValueError("Nonempty pool and positive selection budget required")
    k = min(k, n)
    ordered = sorted(values)
    denominator = math.comb(n, k)
    return math.fsum(ordered[i] * math.comb(i, k - 1) / denominator for i in range(k - 1, n))


def effects(values: dict[str, float]) -> dict[str, float]:
    return {name: math.fsum(values[key] * coefficient for key, coefficient in terms.items())
            for name, terms in CONTRASTS.items()}


def clustered_summary(matrix, groups, names, *, repetitions, seed, family_size=4):
    """Query-weighted paired means; resample whole groups with a shared draw."""
    x = np.asarray(matrix, dtype=np.float64)
    labels = sorted(set(groups))
    if x.ndim != 2 or len(x) != len(groups) or x.shape[1] != len(names) or len(labels) < 2 or not np.isfinite(x).all():
        raise ValueError("Invalid clustered comparison matrix")
    group_codes = {label: index for index, label in enumerate(labels)}
    codes = np.array([group_codes[label] for label in groups])
    sizes = np.bincount(codes, minlength=len(labels)).astype(float)
    sums = np.zeros((len(labels), x.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, x)
    means = x.mean(axis=0)
    influence = sums - sizes[:, None] * means
    se = np.sqrt(len(labels) / (len(labels) - 1) * np.square(influence).sum(axis=0) / len(x) ** 2)
    rng = np.random.default_rng(seed)
    boot = np.empty((repetitions, x.shape[1]), dtype=np.float64)
    for start in range(0, repetitions, 100):
        count = min(100, repetitions - start)
        draw = rng.integers(0, len(labels), (count, len(labels)))
        boot[start:start + count] = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)[:, None]
    point_critical = float(t.ppf(0.975, len(labels) - 1))
    family_critical = float(t.ppf(1 - 0.05 / (2 * family_size), len(labels) - 1))
    point_quantiles = np.quantile(boot, [0.025, 0.975], axis=0)
    family_quantiles = np.quantile(boot, [0.05 / (2 * family_size), 1 - 0.05 / (2 * family_size)], axis=0)
    changed = np.zeros((len(labels), x.shape[1]), dtype=bool)
    np.logical_or.at(changed, codes, np.abs(x) > 1e-12)
    result = {}
    for index, name in enumerate(names):
        result[name] = {
            "mean": float(means[index]),
            "standard_error_cluster": float(se[index]),
            "ci95_cluster_t": [float(means[index] - point_critical * se[index]), float(means[index] + point_critical * se[index])],
            "ci95_group_bootstrap": point_quantiles[:, index].tolist(),
            "changed_queries": int(np.count_nonzero(np.abs(x[:, index]) > 1e-12)),
            "groups_with_any_nonzero_query": int(changed[:, index].sum()),
        }
        if name == "j_ENTR_contract_difference":
            result[name]["four_cell_family95_bonferroni_cluster_t"] = [float(means[index] - family_critical * se[index]), float(means[index] + family_critical * se[index])]
            result[name]["four_cell_family95_bonferroni_group_bootstrap"] = family_quantiles[:, index].tolist()
    return result
