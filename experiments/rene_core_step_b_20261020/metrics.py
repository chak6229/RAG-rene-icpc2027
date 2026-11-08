from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, t


def clustered(matrix, groups, names, *, family_names=(), repetitions=10000, seed=20261020):
    """Fixed-OOF prediction diagnostic: paired, query-weighted group bootstrap."""
    x = np.asarray(matrix, dtype=float)
    labels = sorted(set(groups))
    if x.ndim != 2 or x.shape != (len(groups), len(names)) or len(labels) < 2 or not np.isfinite(x).all():
        raise ValueError("Invalid cluster input")
    lookup = {g: i for i, g in enumerate(labels)}
    codes = np.asarray([lookup[g] for g in groups])
    sizes = np.bincount(codes, minlength=len(labels)).astype(float)
    sums = np.zeros((len(labels), x.shape[1]))
    np.add.at(sums, codes, x)
    means = x.mean(axis=0)
    residuals = sums - sizes[:, None] * means
    se = np.sqrt(len(labels) / (len(labels) - 1) * np.square(residuals).sum(axis=0) / len(x) ** 2)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((repetitions, len(names)))
    for start in range(0, repetitions, 100):
        count = min(100, repetitions - start)
        draw = rng.integers(0, len(labels), (count, len(labels)))
        bootstrap[start:start + count] = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)[:, None]
    point = np.quantile(bootstrap, [.025, .975], axis=0)
    # Family has eight contrasts over two retrievers; each retriever has four.
    family = np.quantile(bootstrap, [.05 / 16, 1 - .05 / 16], axis=0)
    result = {}
    for i, name in enumerate(names):
        result[name] = {"mean": float(means[i]), "standard_error_cluster": float(se[i]),
                        "ci95_cluster_t": [float(means[i] - t.ppf(.975, len(labels)-1) * se[i]),
                                           float(means[i] + t.ppf(.975, len(labels)-1) * se[i])],
                        "ci95_group_bootstrap": point[:, i].tolist(),
                        "nonzero_queries": int(np.count_nonzero(np.abs(x[:, i]) > 1e-12)),
                        "groups_with_nonzero_queries": len(set(groups[np.abs(x[:, i]) > 1e-12]))}
        if name in family_names:
            critical = t.ppf(1 - .05 / 16, len(labels)-1)
            result[name]["family95_bonferroni_cluster_t"] = [float(means[i] - critical * se[i]), float(means[i] + critical * se[i])]
            result[name]["family95_bonferroni_group_bootstrap"] = family[:, i].tolist()
    return result


def utility(y, baseline, adopted, folds):
    gain = np.asarray(y) * adopted
    harms = adopted & (y < 0)
    benefits = adopted & (y > 0)
    positive_mass = np.maximum(y, 0).sum()
    return {"mrr": float(np.mean(baseline + gain)), "delta_mrr": float(gain.mean()),
            "adopted": int(adopted.sum()), "coverage": float(adopted.mean()),
            "beneficial": int(benefits.sum()), "harmful": int(harms.sum()),
            "zero_gain_adoptions": int((adopted & (y == 0)).sum()),
            "harm_rate_among_adopted": float(harms.sum() / adopted.sum()) if adopted.any() else None,
            "benefit_mass_per_query": float(gain[benefits].sum() / len(y)),
            "harm_mass_per_query": float(-gain[harms].sum() / len(y)),
            "mean_loss_among_harms": float(-y[harms].mean()) if harms.any() else None,
            "max_loss": float(-y[harms].min()) if harms.any() else 0.0,
            "retained_fixed_action_positive_mass": float(gain[benefits].sum() / positive_mass) if positive_mass > 0 else None,
            "per_fold": [{"fold": int(f), "queries": int((folds == f).sum()),
                          "adopted": int((adopted & (folds == f)).sum()),
                          "harmful": int((harms & (folds == f)).sum()),
                          "delta_mrr": float(gain[folds == f].mean())} for f in sorted(set(folds))]}


def prediction_quality(y, prediction, qids):
    error = prediction - y
    # Fixed OOF-score quintiles are descriptive calibration, not fitted calibrators.
    order = sorted(range(len(y)), key=lambda i: (float(prediction[i]), qids[i]))
    bins = []
    for b, subset in enumerate(np.array_split(order, 5)):
        bins.append({"bin": b + 1, "queries": len(subset),
                     "mean_prediction": float(prediction[subset].mean()), "observed_mean_delta": float(y[subset].mean()),
                     "harm_rate": float((y[subset] < 0).mean()), "positive_rate": float((y[subset] > 0).mean())})
    corr = spearmanr(prediction, y).statistic
    return {"mse": float(np.mean(error ** 2)), "mae": float(np.abs(error).mean()),
            "mean_prediction": float(prediction.mean()), "mean_observed": float(y.mean()),
            "bias": float(error.mean()), "prediction_spearman": float(corr) if np.isfinite(corr) else None,
            "negative_predictions": int((prediction < 0).sum()), "positive_predictions": int((prediction > 0).sum()),
            "outside_target_range": int((np.abs(prediction) > 1).sum()),
            "calibration_quintiles": bins,
            "mean_absolute_quintile_calibration_gap": float(sum(b["queries"] * abs(b["mean_prediction"] - b["observed_mean_delta"]) for b in bins) / len(y))}
