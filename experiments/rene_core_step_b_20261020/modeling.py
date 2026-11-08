from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import BLOCKS, require


def fit_fold(x, y, folds, head, held_fold):
    """Only training labels enter both preprocessing and supervised fitting."""
    train, held = folds != held_fold, folds == held_fold
    columns = BLOCKS[head]
    require(train.any() and held.any(), "Empty fold")
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="svd", fit_intercept=True))
    model.fit(x[train][:, columns], y[train])
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    saved = {"head": head, "held_fold": int(held_fold), "columns": columns,
             "alpha": 1.0, "solver": "svd", "fit_intercept": True,
             "train_rows": int(train.sum()), "held_rows": int(held.sum()),
             "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
             "scaler_variance": scaler.var_.tolist(), "coefficient": ridge.coef_.tolist(),
             "intercept": float(ridge.intercept_)}
    prediction = model.predict(x[held][:, columns])
    require(np.isfinite(prediction).all(), "Nonfinite prediction")
    return saved, prediction


def predict_saved(model, x):
    z = (x[:, model["columns"]] - np.asarray(model["scaler_mean"])) / np.asarray(model["scaler_scale"])
    return z @ np.asarray(model["coefficient"]) + model["intercept"]


def decide(prediction, qids, folds, eligible, policy):
    """No labels; quota is floor(eligible_in_fold / 5), including negative scores."""
    prediction, eligible = np.asarray(prediction), np.asarray(eligible, dtype=bool)
    require(np.isfinite(prediction).all() and len(prediction) == len(qids), "Invalid decision scores")
    require(len(qids) == len(set(qids)), "Duplicate query IDs")
    if policy == "positive":
        return eligible & (prediction > 0.0)
    require(policy == "fixed20", "Unknown decision policy")
    adopted = np.zeros(len(qids), dtype=bool)
    for fold in sorted(set(folds)):
        pool = np.flatnonzero((folds == fold) & eligible)
        order = sorted(pool, key=lambda i: (-float(prediction[i]), qids[i]))
        adopted[order[:len(pool) // 5]] = True
    return adopted
