"""Selected-action risk heads; all inference inputs are observable."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from experiments.rene_strict_confirmation_20261019 import model as upstream
from experiments.rene_strict_confirmation_20261019.common import TOL, reject_gold
from run_t5_seed_uq_pilot import SEED_FEATURES

HEADS = ("common", "support", "full")
SUPPORT = (0, 1, 3, 5, 7, 11)
DISPERSION = (2, 4, 6, 8, 9, 10)
COMMON_NAMES = tuple(upstream.risk.FEATURES)
ALL_NAMES = COMMON_NAMES + tuple(SEED_FEATURES)
COSTS = ("billed_slots", "actual_pairs", "executed_ce_tokens")
POLICIES = ("fixed25", "fixed50", "fixed75", "all", "threshold075")
RATES = dict(fixed25=.25, fixed50=.5, fixed75=.75, all=1.)


def inputs(views, traces):
    if len(views) != len(traces):
        raise ValueError("View/trace length mismatch")
    common, seed, native, queries = [], [], [], []
    for e, t in zip(views, traces):
        reject_gold(e)
        reject_gold(t)
        if int(e["query_index"]) != int(t["query_index"]):
            raise ValueError("View/trace query mismatch")
        original = tuple(map(int, e["original_top100"]))
        changed = tuple(map(int, t["ranking"])) != original
        if bool(t["accepted"]) != changed:
            raise ValueError("Adoption must match executed ranking identity")
        if not t["accepted"]:
            continue
        j = int(t["chosen"])
        if not 0 <= j < len(e["seed_x"]):
            raise ValueError("Invalid selected candidate")
        common.append(upstream.risk_x(e, t))
        seed.append(np.asarray(e["seed_x"][j], float))
        native.append(float(t["harm"]))
        queries.append(int(e["query_index"]))
    n = len(queries)
    result = dict(
        common=np.asarray(common, float).reshape(n, len(COMMON_NAMES)),
        seed=np.asarray(seed, float).reshape(n, len(SEED_FEATURES)),
        native=np.asarray(native, float), q=queries,
    )
    if not all(np.isfinite(result[k]).all() for k in ("common", "seed", "native")):
        raise ValueError("Nonfinite observable inputs")
    return result


def design_x(data, head):
    if head not in HEADS:
        raise ValueError("Unknown risk head")
    x = data["common"] if head == "common" else np.c_[data["common"], data["seed"]]
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite risk inputs")
    return x


def feature_mask(head):
    mask = upstream.risk.model_mask("threeway_anchor")
    if head == "common":
        return mask
    extra = np.zeros(12)
    extra[list(SUPPORT) if head == "support" else list(range(12))] = 1.
    return np.r_[mask, extra]


def fit(data, head):
    """Same historical likelihood, offset and fixed penalties; genuine refits."""
    x = design_x(data, head)
    delta = np.asarray(data["delta"], float)
    if len(x) != len(delta) or len(data["q"]) != len(delta):
        raise ValueError("Training alignment")
    n = len(delta)
    common_data = dict(
        x=x, y=(delta < -TOL).astype(float), delta=delta,
        native=data["native"], q=data["q"], repo=data["repo"],
    )
    if head == "common":
        result = upstream.risk.fit_model(common_data, "threeway_anchor")
        result.update(head=head, feature_names=COMMON_NAMES)
        return result
    counts = np.array([
        np.sum(delta < -TOL), np.sum(np.abs(delta) <= TOL), np.sum(delta > TOL)
    ], int)
    mean = x.mean(0) if n else np.zeros(x.shape[1])
    scale = x.std(0) if n else np.ones(x.shape[1])
    scale[scale < 1e-6] = 1.
    penalty = np.r_[1., np.full(x.shape[1], 10.)]
    penalty[1] = 50.
    model = dict(
        arm="threeway_anchor", head=head, kind="threeway",
        fit_query_ids=list(data["q"]), fit_repos=list(data["repo"]),
        n=n, counts=counts, mask=feature_mask(head), mean=mean, scale=scale,
        penalty=penalty, feature_names=ALL_NAMES,
        fit_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        fit_labels_sha256=hashlib.sha256(delta.tobytes()).hexdigest(),
        class_prior=(counts + .5) / (n + 1.5),
        positive_given_nonharm=float((counts[2] + .5) / (counts[1] + counts[2] + 1.)),
        negative_magnitude=float(np.mean(-delta[delta < -TOL])) if counts[0] else 0.,
        positive_magnitude=float(np.mean(delta[delta > TOL])) if counts[2] else 0.,
        fallback=np.count_nonzero(counts) < 2,
    )
    if model["fallback"]:
        return model
    z = upstream.risk.make_design(model, x)
    native = np.clip(data["native"], 1e-6, 1 - 1e-6)
    offsets = upstream.risk.threeway_offsets(native, model["positive_given_nonharm"])
    labels = np.where(delta < -TOL, 0, np.where(delta > TOL, 2, 1))

    def objective(flat):
        beta = flat.reshape(2, z.shape[1])
        eta = offsets.copy()
        eta[:, [0, 2]] += z @ beta.T
        logs = eta - logsumexp(eta, axis=1, keepdims=True)
        loss = -logs[np.arange(n), labels].sum() + .5 * np.sum(beta * beta * penalty)
        residual = np.exp(logs)
        residual[np.arange(n), labels] -= 1.
        gradient = residual[:, [0, 2]].T @ z + beta * penalty
        return float(loss), gradient.ravel()

    opt = minimize(
        objective, np.zeros(2 * z.shape[1]), method="L-BFGS-B", jac=True,
        options=dict(maxiter=1000, ftol=1e-13, gtol=1e-8),
    )
    if not opt.success:
        raise RuntimeError(str(opt.message))
    model.update(beta=opt.x.reshape(2, z.shape[1]), loss=float(opt.fun),
                 optimizer_success=True)
    return model


def predict(model, data):
    # Labels, if supplied accidentally, are rejected at this narrower boundary.
    forbidden = {"delta", "y", "repo", "labels", "target", "original_rr"}
    if forbidden & set(data):
        raise ValueError("Risk inference must receive observable arrays only")
    if set(data) != {"common", "seed", "native", "q"}:
        raise ValueError("Unexpected inference fields")
    x = design_x(data, model["head"])
    p, classes, gain = upstream.risk.probabilities(
        model, dict(x=x, native=data["native"])
    )
    if not np.isfinite(classes).all() or not np.allclose(classes.sum(1), 1., atol=1e-12):
        raise ValueError("Invalid class probabilities")
    return dict(harm=p, classes=classes, expected_gain=gain)


def adoption(queries, probabilities, policy):
    queries = np.asarray(queries, int)
    probabilities = np.asarray(probabilities, float)
    if len(queries) != len(probabilities) or len(set(queries)) != len(queries):
        raise ValueError("Invalid adoption query alignment")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid harm probability")
    if policy == "threshold075":
        return probabilities <= .075
    if policy not in RATES:
        raise ValueError("Unknown adoption policy")
    count = int(np.floor(RATES[policy] * len(queries)))
    selected = np.zeros(len(queries), bool)
    selected[np.lexsort((queries, probabilities))[:count]] = True
    return selected
