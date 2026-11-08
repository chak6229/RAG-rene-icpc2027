"""Train-only transformations; calibration preserves the raw positive winner."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import BLOCKS, CALIBRATORS, require


def assert_partition(data, train, held):
    require(len(train) > 0 and len(held) > 0, "empty training/held partition")
    require(not set(train) & set(held), "rows overlap")
    require(not set(data.groups[train]) & set(data.groups[held]), "connected groups overlap")


def fit_base(data, train, held, algorithm, block, hgb_parameters):
    assert_partition(data, train, held)
    columns = BLOCKS[block]
    x = data.x[np.ix_(train, columns)]
    # Exactly three candidates per query; inverse-query-count weights normalize to one.
    weights = np.ones(len(train))
    if algorithm == "RIDGE":
        model = Pipeline([("scale", StandardScaler()), ("regressor", Ridge(alpha=10.0))])
        model.fit(x, data.y[train], scale__sample_weight=weights, regressor__sample_weight=weights)
    else:
        model = HistGradientBoostingRegressor(**hgb_parameters).fit(x, data.y[train], sample_weight=weights)
    return model, np.clip(model.predict(data.x[np.ix_(held, columns)]), -1, 1)


def inner_oof(data, outer, algorithm, hgb_parameters, emit=None):
    """Other original outer labels become four inner folds, all inside outer A."""
    prediction = np.full(len(data.rows), np.nan)
    for inner in sorted(set(data.folds) - {outer}):
        train = np.flatnonzero((data.folds != outer) & (data.folds != inner))
        held = np.flatnonzero(data.folds == inner)
        require(not np.any(data.folds[train] == outer) and not np.any(data.folds[held] == outer), "outer fold in inner fit")
        model, values = fit_base(data, train, held, algorithm, "FULL", hgb_parameters)
        prediction[held] = values
        if emit:
            emit(int(inner), train, held, model)
    require(np.isfinite(prediction[data.folds != outer]).all(), "inner OOF incomplete")
    require(np.isnan(prediction[data.folds == outer]).all(), "outer predictions leaked into inner data")
    return prediction


def winners(data, indices, prediction):
    indices = set(int(i) for i in indices)
    result = []
    for q in data.qids:
        ids = data.query_rows[q]
        if not any(i in indices for i in ids):
            continue
        require(all(i in indices for i in ids), "partial query in partition")
        require(all(np.isfinite(prediction[i]) for i in ids), "missing candidate predictions")
        scores = {i: 0.0 if data.rows[i]["structural_zero"] else float(np.clip(prediction[i], -1, 1)) for i in ids}
        order = sorted(ids, key=lambda i: (-scores[i], data.rows[i]["entr_rank"], data.rows[i]["position"]))
        best = order[0]
        second = max(0.0, scores[order[1]])  # NoRewrite is an explicit zero-score competitor.
        result.append({"query_id": q, "index": best, "position": data.rows[best]["position"],
                       "fold": int(data.folds[best]), "score": scores[best], "runner_up": second,
                       "gap": scores[best] - second,
                       "eligible": bool(scores[best] > 0 and not data.rows[best]["structural_zero"]),
                       "structural_zero": bool(data.rows[best]["structural_zero"])})
    return result


def calibration_matrix(data, selected, method):
    if method in ("AFFINE", "ISOTONIC"):
        return np.asarray([[r["score"]] for r in selected], dtype=float).reshape(len(selected), 1)
    # Three prediction/competition signals plus 52 retrieval/anchor/lexical signals.
    return np.asarray([[r["score"], r["runner_up"], r["gap"], *data.x[r["index"], 19:71]]
                       for r in selected], dtype=float).reshape(len(selected), 55)


@dataclass
class ConstantCalibration:
    value: float

    def predict(self, x):
        return np.full(len(x), self.value)


def fit_calibrator(data, training_winners, method):
    require(method in CALIBRATORS, "unknown calibration method")
    eligible = [r for r in training_winners if r["eligible"]]
    x = calibration_matrix(data, eligible, method)
    y = np.asarray([data.y[r["index"]] for r in eligible], dtype=float)
    # A fixed fallback for low-event folds, never a fitted held-out threshold.
    if len(y) < 30:
        return ConstantCalibration(float(y.mean()) if len(y) else 0.0), eligible
    if method == "ISOTONIC":
        model = IsotonicRegression(increasing=True, y_min=-1, y_max=1, out_of_bounds="clip")
        model.fit(x[:, 0], y)
    else:
        model = Pipeline([("scale", StandardScaler()), ("regressor", Ridge(alpha=10.0))])
        model.fit(x, y)
    return model, eligible


def predict_calibrator(model, data, selected, method):
    x = calibration_matrix(data, selected, method)
    if not len(x):
        return np.empty(0)
    values = model.predict(x[:, 0] if method == "ISOTONIC" and isinstance(model, IsotonicRegression) else x)
    return np.clip(values, -1, 1)


def calibrated_winners(data, outer, outer_prediction, inner_prediction):
    train = winners(data, np.flatnonzero(data.folds != outer), inner_prediction)
    held = winners(data, np.flatnonzero(data.folds == outer), outer_prediction)
    require(not {r["query_id"] for r in train} & {r["query_id"] for r in held}, "calibration query leak")
    models = {}
    for method in CALIBRATORS:
        model, eligible = fit_calibrator(data, train, method)
        values = predict_calibrator(model, data, held, method)
        models[method] = (model, eligible)
        for row, value in zip(held, values):
            row.setdefault("calibrated", {})[method] = float(value) if row["eligible"] else 0.0
    return held, train, models


def quota(selected, score_key, fraction, *, eligible_only):
    """Fixed count within each outer fold; query-ID tie, no outcome argument."""
    adopted = set()
    for fold in sorted({r["fold"] for r in selected}):
        current = [r for r in selected if r["fold"] == fold]
        pool = [r for r in current if r["eligible"]] if eligible_only else [r for r in current if not r["structural_zero"]]
        denominator = len(pool) if eligible_only else len(current)
        count = int(np.floor(fraction * denominator))
        require(len(pool) >= count, "not enough changed-ranking winners for fixed coverage")
        # Isotonic plateaus must not invent a ranking improvement through ties.
        ordered = sorted(pool, key=lambda r: (-r[score_key], -r["score"], r["query_id"]))
        adopted.update(r["query_id"] for r in ordered[:count])
    return adopted


def make_decision(data, winner, policy, score, accept):
    q = winner["query_id"]
    position = winner["position"] if accept else None
    original = data.actions[q, None]["ranked_doc_ids"]
    ranking = data.actions[q, position]["ranked_doc_ids"]
    adopted = ranking != original
    if not adopted:
        position = None
        ranking = original
    return {"query_id": q, "policy": policy, "fold": data.fold_map[q],
            "winner_position": winner["position"], "selected_position": position,
            "predicted_gain": None if score is None else float(score), "adopted": adopted,
            "fallback_restores_full_ranking": not adopted, "used_gold_at_prediction": False,
            "ranking_reference": {"retriever": data.retriever, "query_id": q, "position": position},
            "candidate_reselected_after_calibration": False}
