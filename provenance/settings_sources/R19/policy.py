"""One sequential CoSQA study adapter, trained in three disjoint group roles.

Uses current real 71-column CoSQA inputs, rather than padding missing historical
seed columns. The old formal strong84/fusion policy is not changed or impersonated.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.rene_cosqa_development_20261019.freeze import HGB
from .common import TOL, require

CE_NAMES = (
    "ce_max", "ce_min", "ce_mean", "ce_std", "ce_original_top1_available", "ce_original_top1_logit",
    "ce_candidate_top1_available", "ce_candidate_top1_logit", "ce_query_original_top1_mass",
    "ce_query_candidate_top1_mass", "ce_weighted_delta_rr", "ce_weighted_positive_mass",
    "ce_weighted_negative_mass", "ce_unobserved_negative_mass_fraction", "ce_anchor_promoted_conflict",
    "ce_original_top3_mass", "ce_candidate_top3_mass", "ce_observed_docs",
)
RISK_NAMES = (
    "verified", "logit_native_harm", "predicted_mean", "winner_gap", "worst_rr_loss", "top1_changed",
    "top10_change_fraction", "positive_rank_mass", "negative_rank_mass", "ce_available",
    "ce_down_probability", "ce_weighted_rr_loss", "ce_weighted_rr_gain", "ce_max_logit",
    "original_top1_drop", "original_rank2_10_loss_mass", "unobserved_down_mass_fraction",
    "ce_up_down_pair_available", "ce_top1_drop_conflict", "ce_rank2_10_loss_conflict",
)


def regression(x, y, weights=None):
    model = Pipeline([("scale", StandardScaler()), ("regressor", Ridge(alpha=10.0))])
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    model.fit(x, y, scale__sample_weight=weights, regressor__sample_weight=weights)
    return model


def binary_fit(x, y, weights):
    prior = float((np.dot(weights, y) + .5) / (weights.sum() + 1.))
    if len(np.unique(y)) < 2:
        return {"constant": prior, "model": None}
    model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(
        C=.1, solver="lbfgs", max_iter=2000, random_state=261019))])
    model.fit(x, y, scale__sample_weight=weights, classifier__sample_weight=weights)
    return {"constant": prior, "model": model}


def binary_predict(model, x):
    return np.full(len(x), model["constant"]) if model["model"] is None else model["model"].predict_proba(x)[:, 1]


def fit_gain(rows, labels):
    require(len(rows) == len(labels) > 0, "gain training data invalid")
    counts = Counter(r["query_id"] for r in rows)
    weights = np.asarray([1. / counts[r["query_id"]] for r in rows])
    weights /= weights.mean()
    x = np.asarray([r["values"] for r in rows], float)
    y = np.asarray(labels, float)
    return {"mean": regression(x, y, weights),
            "harm": binary_fit(x, (y < -TOL).astype(int), weights),
            "hgb_control": HistGradientBoostingRegressor(**HGB).fit(x, y, sample_weight=weights),
            "fit_query_ids": sorted(counts), "label_count": len(y)}


def gains(model, rows):
    x = np.asarray([r["values"] for r in rows], float)
    mu = np.clip(model["mean"].predict(x), -1., 1.)
    control = np.clip(model["hgb_control"].predict(x), -1., 1.)
    harm = np.clip(binary_predict(model["harm"], x), 1e-6, 1. - 1e-6)
    for i, r in enumerate(rows):
        if r["structural_zero"]:
            mu[i] = control[i] = 0.
    return mu, harm, control


def order(rows, values):
    return sorted(range(len(rows)), key=lambda i: (-float(values[i]), rows[i]["entr_rank"], rows[i]["position"]))


def rr_map(ranking):
    return {int(d): 1. / (i + 1) for i, d in enumerate(ranking)}


def verification_plan(original, rankings, rows, mu):
    ordered = order(rows, mu)
    if not ordered or mu[ordered[0]] <= 0 or rows[ordered[0]]["structural_zero"]:
        return None
    winner = ordered[0]
    rival = ordered[1] if len(ordered) > 1 else winner
    old = rr_map(original)
    new = rr_map(rankings[winner])
    # Original success anchors, selected winner and rival, then largest actual position changes.
    seeds = [original[0], rankings[winner][0], original[1], rankings[rival][0], original[2]]
    change = sorted(set(old) | set(new), key=lambda d: (-abs(new.get(d, 0.) - old.get(d, 0.)), d))
    tail = [d for k in ordered for d in rankings[k][:10]] + original[:10]
    docs = list(dict.fromkeys(seeds + change + tail))[:8]
    require(len(docs) == 8 and len(set(docs)) == 8, "eight real unique verification documents required")
    return {"doc_ids": docs, "winner": winner, "rival": rival,
            "priority": float(mu[winner]), "used_gold": False}


def ce_features(original, ranking, plan, scores):
    if plan is None:
        return np.zeros(len(CE_NAMES))
    docs = plan["doc_ids"]
    require(set(docs) <= set(scores), "missing requested real CE evidence")
    logits = np.array([scores[d] for d in docs], float)
    require(np.isfinite(logits).all(), "nonfinite CE evidence")
    prob = np.exp(logits - logsumexp(logits))
    known = dict(zip(docs, logits))
    masses = dict(zip(docs, prob))
    old, new = rr_map(original), rr_map(ranking)
    delta = {d: new.get(d, 0.) - old.get(d, 0.) for d in set(old) | set(new)}
    down = {d: -v for d, v in delta.items() if v < -TOL}
    up = {d for d, v in delta.items() if v > TOL}
    observed_up = [known[d] for d in up if d in known]
    total_down = sum(down.values())
    missing = sum(v for d, v in down.items() if d not in known) / total_down if total_down else 0.
    anchor_conflict = float(expit(known[original[0]] - max(observed_up))) if original[0] in known and observed_up else 0.
    return np.array([logits.max(), logits.min(), logits.mean(), logits.std(),
                     float(original[0] in known), known.get(original[0], 0.),
                     float(ranking[0] in known), known.get(ranking[0], 0.),
                     masses.get(original[0], 0.), masses.get(ranking[0], 0.),
                     sum(masses[d] * delta.get(d, 0.) for d in docs),
                     sum(masses[d] * max(delta.get(d, 0.), 0.) for d in docs),
                     sum(masses[d] * max(-delta.get(d, 0.), 0.) for d in docs),
                     missing, anchor_conflict, sum(masses.get(d, 0.) for d in original[:3]),
                     sum(masses.get(d, 0.) for d in ranking[:3]), float(len(docs))], float)


def witness_matrix(original, rankings, rows, mu, plan, scores):
    best = float(max(mu))
    return np.stack([np.r_[r["values"], float(mu[i]), best - float(mu[i]),
                           ce_features(original, rankings[i], plan, scores)] for i, r in enumerate(rows)])


def fit_update(training):
    # Each tuple is a separate-role query with honest base predictions and real CE scores.
    xx, yy, qq, counts = [], [], [], []
    for item in training:
        rows, mu = item["rows"], item["mu"]
        if item["plan"] is None:
            continue
        x = witness_matrix(item["original"], item["rankings"], rows, mu, item["plan"], item["ce"])
        xx.extend(x)
        yy.extend(np.asarray(item["labels"]) - mu)
        qq.extend([item["query_id"]] * len(rows))
        counts.extend([1. / len(rows)] * len(rows))
    if len(yy) < 30:
        return {"model": None, "fit_query_ids": sorted(set(qq)), "training_rows": len(yy), "fallback": "zero residual"}
    return {"model": regression(np.asarray(xx), np.asarray(yy), np.asarray(counts) / np.mean(counts)),
            "fit_query_ids": sorted(set(qq)), "training_rows": len(yy), "fallback": None}


def allocation(items):
    eligible = [q for q, item in items.items() if item["plan"] is not None]
    chosen = sorted(eligible, key=lambda q: (-items[q]["plan"]["priority"], q))[:len(items) // 4]
    return set(chosen)


def trace(item, update, verified):
    rows, original = item["rows"], item["original"]
    mu = np.asarray(item["mu"]).copy()
    if verified:
        require(item["plan"] is not None, "verification without eligible plan")
        x = witness_matrix(original, item["rankings"], rows, mu, item["plan"], item["ce"])
        if update["model"] is not None:
            mu = np.clip(mu + .5 * update["model"].predict(x), -1., 1.)
        for i, row in enumerate(rows):
            if row["structural_zero"]:
                mu[i] = 0.
    ordering = order(rows, mu)
    best = ordering[0]
    accepted = bool(mu[best] > 0 and not rows[best]["structural_zero"])
    ranking = item["rankings"][best] if accepted else original
    accepted = bool(accepted and ranking != original)
    return {"query_id": item["query_id"], "chosen": best if accepted else None,
            "position": rows[best]["position"] if accepted else None, "ranking": list(ranking if accepted else original),
            "accepted": accepted, "verified": bool(verified), "predicted_gain": float(mu[best]) if accepted else 0.,
            "gap": float(mu[best] - max(0., mu[ordering[1]] if len(ordering) > 1 else 0.)),
            "actual_pairs": 8 if verified else 0,
            "verification_doc_ids": list(item["plan"]["doc_ids"]) if verified else [],
            "final_scores": mu.tolist()}


def risk_features(item, t):
    require(t["accepted"], "risk fit/evaluation requires an actual nonidentity action")
    j = t["chosen"]
    old, new = rr_map(item["original"]), rr_map(t["ranking"])
    delta = {d: new.get(d, 0.) - old.get(d, 0.) for d in set(old) | set(new)}
    down = {d: -v for d, v in delta.items() if v < -TOL}
    up = {d for d, v in delta.items() if v > TOL}
    verified = t["verified"]
    docs = item["plan"]["doc_ids"] if verified else []
    known = {d: item["ce"][d] for d in docs}
    logits = np.array(list(known.values()), float)
    mass = dict(zip(docs, np.exp(logits - logsumexp(logits)))) if len(logits) else {}
    totaldown = sum(down.values())
    missing = sum(v for d, v in down.items() if d not in known) / totaldown if totaldown else 0.
    promoted = [known[d] for d in up if d in known]
    demoted = [d for d in down if d in known]
    pair = bool(promoted and demoted)
    band = set(item["original"][1:10])
    bandmass = sum(v for d, v in down.items() if d in band)
    topdrop = down.get(item["original"][0], 0.)
    conflict1 = topdrop * float(expit(known[item["original"][0]] - max(promoted))) if pair and item["original"][0] in known else 0.
    conflictband = (sum(down[d] * float(expit(known[d] - max(promoted))) for d in demoted if d in band)
                    / max(bandmass, 1e-12)) if pair else 0.
    native = float(np.clip(item["native_harm"][j], 1e-6, 1. - 1e-6))
    x = np.array([float(verified), logit(native), t["predicted_gain"], t["gap"],
                  max(down.values(), default=0.), float(item["original"][0] != t["ranking"][0]),
                  1. - len(set(item["original"][:10]) & set(t["ranking"][:10])) / 10.,
                  sum(v for v in delta.values() if v > 0), totaldown, float(bool(known)),
                  sum(mass.get(d, 0.) for d in down), sum(mass.get(d, 0.) * v for d, v in down.items()),
                  sum(mass.get(d, 0.) * delta[d] for d in up), max(known.values(), default=0.),
                  topdrop, bandmass, missing, float(pair), conflict1, conflictband], float)
    require(x.shape == (len(RISK_NAMES),) and np.isfinite(x).all(), "invalid risk evidence")
    return x, native


def offsets(native, positive_given_nonharm):
    native = np.clip(np.asarray(native), 1e-6, 1. - 1e-6)
    q = float(np.clip(positive_given_nonharm, 1e-6, 1. - 1e-6))
    return np.stack([np.log(native), np.log1p(-native) + np.log1p(-q), np.log1p(-native) + np.log(q)], axis=1)


def fit_risk(x, native, deltas, fit_qids):
    x, native, deltas = np.asarray(x, float).reshape(-1, len(RISK_NAMES)), np.asarray(native, float), np.asarray(deltas, float)
    labels = np.where(deltas < -TOL, 0, np.where(deltas > TOL, 2, 1))
    counts = np.bincount(labels, minlength=3)
    prior = (counts + .5) / (len(labels) + 1.5)
    mask = np.ones(len(RISK_NAMES))
    mask[9] = 0.  # CE availability exactly duplicates the verified stage indicator.
    mean = x.mean(0) if len(x) else np.zeros(len(RISK_NAMES))
    scale = x.std(0) if len(x) else np.ones(len(RISK_NAMES))
    scale[scale < 1e-6] = 1.
    result = {"mean": mean, "scale": scale, "mask": mask, "class_prior": prior, "counts": counts,
              "positive_given_nonharm": float((counts[2] + .5) / (counts[1] + counts[2] + 1.)),
              "fit_query_ids": list(fit_qids), "beta": None, "training_actions": len(labels)}
    if len(labels) < 30 or np.count_nonzero(counts) < 2:
        return result
    design = np.c_[np.ones(len(x)), (x - mean) / scale * mask]
    penalty = np.r_[1., np.full(len(RISK_NAMES), 10.)]
    penalty[1] = 50.
    off = offsets(native, result["positive_given_nonharm"])

    def objective(flat):
        beta = flat.reshape(2, design.shape[1])
        eta = off.copy()
        eta[:, [0, 2]] += design @ beta.T
        logs = eta - logsumexp(eta, axis=1, keepdims=True)
        loss = -logs[np.arange(len(labels)), labels].sum() + .5 * np.sum(beta * beta * penalty)
        residual = np.exp(logs)
        residual[np.arange(len(labels)), labels] -= 1.
        gradient = residual[:, [0, 2]].T @ design + beta * penalty
        return float(loss), gradient.ravel()

    fit = minimize(objective, np.zeros(2 * design.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-7})
    require(fit.success or np.max(np.abs(fit.jac)) < 1e-4, "threeway optimization did not converge")
    result.update(beta=fit.x.reshape(2, design.shape[1]), optimizer_success=bool(fit.success),
                  gradient_max=float(np.max(np.abs(fit.jac))))
    return result


def risk_predict(model, x, native):
    x = np.asarray(x, float).reshape(-1, len(RISK_NAMES))
    if model["beta"] is None:
        return np.tile(model["class_prior"], (len(x), 1))
    design = np.c_[np.ones(len(x)), (x - model["mean"]) / model["scale"] * model["mask"]]
    eta = offsets(native, model["positive_given_nonharm"])
    eta[:, [0, 2]] += design @ model["beta"].T
    prob = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
    require(np.isfinite(prob).all() and np.allclose(prob.sum(1), 1.), "invalid threeway probability")
    return prob


def apply_risk(item, t, model):
    result = {**t, "class_probabilities": None, "risk_kept": False}
    if not t["accepted"]:
        return result
    x, native = risk_features(item, t)
    prob = risk_predict(model, [x], [native])[0]
    result["class_probabilities"] = prob.tolist()
    result["risk_kept"] = bool(prob[0] <= .075)
    if not result["risk_kept"]:
        result.update(accepted=False, position=None, chosen=None, ranking=list(item["original"]), predicted_gain=0.)
    require(result["actual_pairs"] == t["actual_pairs"], "risk fallback cannot refund verification cost")
    return result


def group_items(query_ids, rows, actions, gain_model):
    pooled = defaultdict(list)
    for i, row in enumerate(rows):
        pooled[row["query_id"]].append(i)
    all_mu, all_native, all_control = gains(gain_model, rows)
    result = {}
    for q in query_ids:
        indices = sorted(pooled[q], key=lambda i: rows[i]["entr_rank"])
        current = [rows[i] for i in indices]
        require(current, "query missing candidate features")
        mu, native, control = all_mu[indices], all_native[indices], all_control[indices]
        original = actions[q, None]["ranked_doc_ids"]
        rankings = [actions[q, r["position"]]["ranked_doc_ids"] for r in current]
        plan = verification_plan(original, rankings, current, mu)
        result[q] = {"query_id": q, "rows": current, "original": original, "rankings": rankings,
                     "mu": mu, "native_harm": native, "control": control, "plan": plan, "ce": {}}
    return result


def all_decisions(items, update, risk):
    allocated = allocation(items)
    outputs = []
    for q, item in items.items():
        base = trace(item, update, False)
        verified = trace(item, update, q in allocated)
        gated = apply_risk(item, verified, risk)
        rows = item["rows"]
        native = min(range(len(rows)), key=lambda i: rows[i]["entr_rank"])
        control = order(rows, item["control"])[0]
        prob = min(range(len(rows)), key=lambda i: (-rows[i]["values"][1], rows[i]["position"]))
        for name, j in (("NoRewrite", None), ("ENTR_top1", native), ("PROB_in_ENTR3", prob),
                        ("HGB_gain_control", control if item["control"][control] > 0 else None)):
            ranking = item["original"] if j is None else item["rankings"][j]
            accepted = ranking != item["original"]
            outputs.append({"query_id": q, "policy": name, "position": rows[j]["position"] if accepted else None,
                            "ranking": list(ranking), "accepted": accepted, "verified": False, "actual_pairs": 0,
                            "predicted_gain": 0. if j is None else float(item["control"][j]) if name == "HGB_gain_control" else None})
        for name, t in (("RIDGE_gain", base), ("RIDGE_verified", verified), ("RIDGE_risk", gated)):
            outputs.append({**t, "policy": name, "used_gold": False})
    return outputs, allocated
