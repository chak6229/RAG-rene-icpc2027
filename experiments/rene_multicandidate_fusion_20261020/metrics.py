from __future__ import annotations

import numpy as np
from scipy.stats import t

CONTRASTS = {
    "RRF3-NoRewrite": ("RRF3", "NoRewrite"),
    "RRF4-NoRewrite": ("RRF4", "NoRewrite"),
    "ENTRTop1-NoRewrite": ("ENTRTop1", "NoRewrite"),
    "ENTRBest3-NoRewrite": ("ENTRBest3", "NoRewrite"),
    "RRF4-RRF3": ("RRF4", "RRF3"),
    "RRF3-ENTRTop1": ("RRF3", "ENTRTop1"),
    "RRF4-ENTRTop1": ("RRF4", "ENTRTop1"),
    "ENTRBest3-ENTRTop1": ("ENTRBest3", "ENTRTop1"),
    "RRF3-ENTRBest3": ("RRF3", "ENTRBest3"),
    "RRF4-ENTRBest3": ("RRF4", "ENTRBest3"),
}
PRIMARY = ("RRF3-NoRewrite", "RRF4-NoRewrite")


def rr(order, relevant):
    return next((1.0 / i for i, doc in enumerate(order, 1) if doc in relevant), 0.0)


def intervals(matrix, groups, names, *, family_names=(), seed=20261020, repetitions=10000):
    x = np.asarray(matrix, dtype=np.float64)
    groups = np.asarray(groups)
    labels = sorted(set(groups))
    if x.shape != (len(groups), len(names)) or len(labels) < 2 or not np.isfinite(x).all():
        raise ValueError("Invalid paired cluster input")
    lookup = {g: i for i, g in enumerate(labels)}
    codes = np.asarray([lookup[g] for g in groups])
    sizes = np.bincount(codes, minlength=len(labels)).astype(float)
    sums = np.zeros((len(labels), x.shape[1]))
    np.add.at(sums, codes, x)
    point = x.mean(axis=0)
    influence = sums-sizes[:, None]*point
    se = np.sqrt(len(labels)/(len(labels)-1)*np.square(influence).sum(axis=0)/len(x)**2)
    rng = np.random.default_rng(seed)
    sample = np.empty((repetitions, x.shape[1]))
    for start in range(0, repetitions, 100):
        count = min(100, repetitions-start)
        draw = rng.integers(0, len(labels), size=(count, len(labels)))
        sample[start:start+count] = sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:, None]
    p95 = np.quantile(sample, [.025, .975], axis=0)
    f95 = np.quantile(sample, [.05/8, 1-.05/8], axis=0)
    result = {}
    for i, name in enumerate(names):
        c = float(t.ppf(.975, len(labels)-1))
        result[name] = {"mean": float(point[i]), "standard_error_cluster": float(se[i]),
                        "ci95_cluster_t": [float(point[i]-c*se[i]), float(point[i]+c*se[i])],
                        "ci95_group_bootstrap": p95[:, i].tolist(),
                        "changed_queries": int(np.count_nonzero(np.abs(x[:, i]) > 1e-12)),
                        "groups_with_nonzero_queries": len(set(groups[np.abs(x[:, i]) > 1e-12]))}
        if name in family_names:
            critical = float(t.ppf(1-.05/8, len(labels)-1))
            result[name]["four_effect_family95_bonferroni_cluster_t"] = [float(point[i]-critical*se[i]), float(point[i]+critical*se[i])]
            result[name]["four_effect_family95_bonferroni_group_bootstrap"] = f95[:, i].tolist()
    return result


def utility(values, baseline, changed):
    gain = values-baseline
    positive, negative = gain > 0, gain < 0
    return {"mrr": float(values.mean()), "delta_mrr": float(gain.mean()),
            "beneficial_queries": int(positive.sum()), "harmful_queries": int(negative.sum()),
            "zero_gain_queries": int((gain == 0).sum()), "harm_rate_all_queries": float(negative.mean()),
            "benefit_mass_per_query": float(np.maximum(gain, 0).mean()),
            "harm_mass_per_query": float(np.maximum(-gain, 0).mean()),
            "mean_loss_among_harms": float(-gain[negative].mean()) if negative.any() else None,
            "max_loss": float(-gain[negative].min()) if negative.any() else 0.0,
            "ranking_changed_queries": int(changed.sum()) if changed is not None else None,
            "harm_rate_ranking_changed": float(negative.sum()/changed.sum()) if changed is not None and changed.any() else None}
