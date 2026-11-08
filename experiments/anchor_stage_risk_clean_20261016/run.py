"""Fixed-action, CPU-only stage shrinkage and observable anchor risk models."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("anchor_previous", ROOT / "experiments/selected_action_risk_deduplicated_20261016/run.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
SOURCE = s.SOURCE
OLD = s.OUT
OUT = ROOT / "results/anchor_stage_risk_clean_20261016"
PRIMARY = "anchor_shrunk"
ARMS = ("baseline", "previous_structural", "stage_shrunk", PRIMARY,
        "anchor_shrunk_uq", "threeway_anchor", "threeway_positive")
NEW = ARMS[2:-1]
ANCHORS = ("original_top1_drop", "original_rank2_10_loss_mass", "unobserved_down_mass_fraction",
           "ce_up_down_pair_available", "ce_top1_drop_conflict", "ce_rank2_10_loss_conflict")
FEATURES = s.FEATURES + ANCHORS
TAU, TOL, ALPHA = .075, 1e-10, .05 / 8


def freeze():
    inputs = [Path(__file__), HERE / "DESIGN.md", Path(s.__file__), OLD / "PROTOCOL.json", OLD / "AUDIT.json"]
    inputs += [OLD / f"{fam}_fold{f}.pkl" for fam in ("rewrite", "fusion") for f in range(4)]
    old = json.loads((OLD / "PROTOCOL.json").read_text("utf8"))
    inputs += [Path(p) for p in old["source_hashes"]]
    for p, h in {**old["source_hashes"], **old["protected_hashes"]}.items():
        assert s.sha(p) == h, p
    z = dict(primary=PRIMARY, arms=ARMS, features=FEATURES, base_tau=.1, post_tau=TAU,
             stage_standardized_coefficient_ridge=50., other_coefficient_ridge=10., intercept_ridge=1.,
             fit="complete inner pipeline OOF192 adopted actions, no class reweighting",
             same_actions=True, same_verification=True, same_cost=True, reject_then_reselect=False,
             alpha_per_method_check=ALPHA, risk_target=.1, hyperparameter_selection=False,
             audit_correction="mask pair_available column20; it duplicates stage in every fit; diagnostic retained",
             historical_development=True, certificate_issued=False, test_used=False,
             new_t5=False, new_retrieval=False, new_ce=False, neural_model_calls=False,
             source_hashes={str(p): s.sha(p) for p in inputs}, protected_hashes=old["protected_hashes"])
    path = OUT / "PROTOCOL.json"
    if path.exists():
        assert json.loads(path.read_text("utf8")) == json.loads(json.dumps(z))
    else:
        s.dump(path, z)
    return z


def anchor_features(e, t):
    original = tuple(map(int, e["original_top100"]))
    action = tuple(map(int, t["ranking"]))
    r0 = {d: 1. / (i + 1) for i, d in enumerate(original)}
    ra = {d: 1. / (i + 1) for i, d in enumerate(action)}
    down = {d: r - ra.get(d, 0.) for d, r in r0.items() if r - ra.get(d, 0.) > TOL}
    up = {d for d, r in ra.items() if r - r0.get(d, 0.) > TOL}
    top1 = down.get(original[0], 0.)
    band = set(original[1:10])
    bandmass = sum(v for d, v in down.items() if d in band)
    total = sum(down.values())
    known = {}
    if t["verified"]:
        chosen = int(t["chosen"])
        u = next(u for u in t["updated"] if int(u["candidate"]) == chosen and u["arm"] == "full")
        x = np.asarray(u["x"])
        pad = x[90:122].reshape(8, 4)
        docs = list(map(int, t["plan"]["doc_ids"]))
        assert len(docs) == int(np.sum(pad[:, 0] > 0))
        known = dict(zip(docs, map(float, x[122:122 + len(docs)])))
    missing = sum(v for d, v in down.items() if d not in known) / total if total > 0 else 0.
    promoted = [known[d] for d in up if d in known]
    demoted = [d for d in down if d in known]
    pair = bool(promoted and demoted)
    conflict1 = conflictband = 0.
    if pair:
        bestup = max(promoted)
        if original[0] in known:
            conflict1 = top1 * float(expit(known[original[0]] - bestup))
        conflictband = sum(down[d] * float(expit(known[d] - bestup)) for d in demoted if d in band) / max(bandmass, 1e-12)
    z = np.array([top1, bandmass, missing, float(pair), conflict1, conflictband])
    assert z.shape == (6,) and np.isfinite(z).all() and 0 <= missing <= 1 + 1e-10
    return z


def rows(byid, traces):
    data = s.rows(byid, traces)
    selected = [t for t in traces if t["accepted"]]
    a = np.stack([anchor_features(byid[t["query_index"]], t) for t in selected]) if selected else np.empty((0, 6))
    data["x"] = np.c_[data["x"], a]
    data["delta"] = np.array([t["delta_rr"] for t in selected])
    return data


def model_mask(arm):
    mask = np.zeros(len(FEATURES))
    mask[:14] = 1.
    mask[9] = 0.
    if arm != "stage_shrunk":
        mask[17:] = 1.
    if arm == "anchor_shrunk_uq":
        mask[14:17] = 1.
    # Availability is retained in artifacts, but duplicates the stage indicator in this pipeline.
    mask[20] = 0.
    return mask


def make_design(model, x):
    return np.c_[np.ones(len(x)), (x - model["mean"]) / model["scale"] * model["mask"]]


def fit_model(data, arm):
    x, y = data["x"], data["y"]
    n = len(y)
    counts = np.array([np.sum(data["delta"] < -TOL), np.sum(np.abs(data["delta"]) <= TOL), np.sum(data["delta"] > TOL)], int)
    prior = (counts + .5) / (n + 1.5)
    mean = x.mean(0) if n else np.zeros(len(FEATURES))
    scale = x.std(0) if n else np.ones(len(FEATURES))
    scale[scale < 1e-6] = 1.
    penalty = np.r_[1., np.full(len(FEATURES), 10.)]
    penalty[1] = 50.
    model = dict(arm=arm, fit_query_ids=data["q"], fit_repos=data["repo"], n=n, counts=counts,
                 mask=model_mask(arm), mean=mean, scale=scale, penalty=penalty,
                 fit_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                 fit_labels_sha256=hashlib.sha256(data["delta"].tobytes()).hexdigest(),
                 class_prior=prior, positive_given_nonharm=float((counts[2] + .5) / (counts[1] + counts[2] + 1.)),
                 negative_magnitude=float(np.mean(-data["delta"][data["delta"] < -TOL])) if counts[0] else 0.,
                 positive_magnitude=float(np.mean(data["delta"][data["delta"] > TOL])) if counts[2] else 0.)
    design = make_design(model, x)
    native = np.clip(data["native"], 1e-6, 1 - 1e-6)
    if arm == "threeway_anchor":
        model["kind"] = "threeway"
        model["fallback"] = np.count_nonzero(counts) < 2
        if model["fallback"]:
            return model
        labels = np.where(data["delta"] < -TOL, 0, np.where(data["delta"] > TOL, 2, 1))
        offsets = threeway_offsets(native, model["positive_given_nonharm"])

        def objective(flat):
            beta = flat.reshape(2, design.shape[1])
            eta = offsets.copy()
            eta[:, [0, 2]] += design @ beta.T
            logs = eta - logsumexp(eta, axis=1, keepdims=True)
            loss = -logs[np.arange(n), labels].sum() + .5 * np.sum(beta * beta * penalty)
            residual = np.exp(logs)
            residual[np.arange(n), labels] -= 1.
            grad = residual[:, [0, 2]].T @ design + beta * penalty
            return float(loss), grad.ravel()

        result = minimize(objective, np.zeros(2 * design.shape[1]), method="L-BFGS-B", jac=True,
                          options=dict(maxiter=1000, ftol=1e-13, gtol=1e-8))
        model["beta"] = result.x.reshape(2, design.shape[1])
    else:
        model["kind"] = "binary"
        model["fallback"] = n == 0 or len(np.unique(y)) < 2
        model["binary_prior"] = float((y.sum() + .5) / (n + 1))
        if model["fallback"]:
            return model
        offset = logit(native)

        def objective(beta):
            eta = offset + design @ beta
            loss = np.logaddexp(0, eta).sum() - y @ eta + .5 * np.dot(penalty * beta, beta)
            grad = design.T @ (expit(eta) - y) + penalty * beta
            return float(loss), grad

        result = minimize(objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
                          options=dict(maxiter=1000, ftol=1e-13, gtol=1e-8))
        model["beta"] = result.x
    assert result.success, result.message
    model.update(loss=float(result.fun), optimizer_success=True)
    return model


def threeway_offsets(native, positive_given_nonharm):
    q = positive_given_nonharm
    return np.log(np.c_[native, (1 - native) * (1 - q), (1 - native) * q])


def probabilities(model, data):
    n = len(data["x"])
    if model["kind"] == "threeway":
        if model["fallback"]:
            p = np.tile(model["class_prior"], (n, 1))
        else:
            eta = threeway_offsets(np.clip(data["native"], 1e-6, 1 - 1e-6), model["positive_given_nonharm"])
            eta[:, [0, 2]] += make_design(model, data["x"]) @ model["beta"].T
            p = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
        gain = p[:, 2] * model["positive_magnitude"] - p[:, 0] * model["negative_magnitude"]
        return p[:, 0], p, gain
    if model["fallback"]:
        p = np.full(n, model["binary_prior"])
    else:
        p = expit(logit(np.clip(data["native"], 1e-6, 1 - 1e-6)) + make_design(model, data["x"]) @ model["beta"])
    return p, None, None


def postgate(traces, data, p, byid, gain=None):
    effective = p.copy()
    if gain is not None:
        effective[gain <= 0] = 1.
    result = s.postgate(traces, data, effective, byid, "new")
    lookup = dict(zip(data["q"], p))
    gains = dict(zip(data["q"], gain)) if gain is not None else {}
    for t in result:
        t["postselection_probability"] = float(lookup[t["query_index"]]) if t["query_index"] in lookup else None
        t["postselection_expected_gain"] = float(gains[t["query_index"]]) if t["query_index"] in gains else None
    return result


def main():
    frozen = freeze()
    began = time.perf_counter()
    byid = {int(e["query_index"]): e for e in s.read(s.r.DATA / "action_items.pkl")}
    results = {}
    for family in ("rewrite", "fusion"):
        for fold in range(4):
            key = f"{family}_fold{fold}"
            source = s.read(SOURCE / f"{key}.pkl")
            old = s.read(OLD / f"{key}.pkl")
            fit_traces = source["selection"]["tables"]["gain_priority"][.1]
            fit = rows(byid, fit_traces)
            assert fit["q"] == old["fit_data"]["q"]
            np.testing.assert_array_equal(fit["x"][:, :17], old["fit_data"]["x"])
            models = {arm: fit_model(fit, arm) for arm in NEW}
            outputs = {}
            for role in ("risk", "held"):
                baseline = source["outputs"][role]["gain_priority"]["traces"]
                assert not set(t["repo"] for t in baseline) & set(t["repo"] for t in fit_traces)
                data = rows(byid, baseline)
                outputs[role] = {"data": data}
                predictions = {arm: probabilities(model, data) for arm, model in models.items()}
                for arm in ARMS:
                    class_p = expected_gain = None
                    if arm in ("baseline", "previous_structural"):
                        previous_arm = "baseline" if arm == "baseline" else "selected_structural"
                        p = old["outputs"][role][previous_arm]["probabilities"]
                        traces = copy.deepcopy(old["outputs"][role][previous_arm]["traces"])
                    else:
                        modelarm = "threeway_anchor" if arm == "threeway_positive" else arm
                        p, class_p, expected_gain = predictions[modelarm]
                        traces = postgate(baseline, data, p, byid, expected_gain if arm == "threeway_positive" else None)
                    for a, b in zip(baseline, traces):
                        assert a["verified"] == b["verified"] and a["plan"] == b["plan"]
                        assert all(a[c] == b[c] for c in s.COSTS)
                        assert not b["accepted"] or (a["accepted"] and a["ranking"] == b["ranking"])
                    outputs[role][arm] = dict(traces=traces, probabilities=p, class_probabilities=class_p,
                                               expected_gain=expected_gain, summary=s.summary(traces),
                                               probability_metrics=s.probability_metrics(data, p))
            cps = {arm: s.cp_diagnostic(outputs["risk"][arm]["traces"]) for arm in ARMS}
            z = dict(key=key, family=family, fold=fold, models=models, fit_data=fit,
                     fit_query_ids=[t["query_index"] for t in fit_traces], outputs=outputs, certificates=cps,
                     protocol_sha256=s.sha(OUT / "PROTOCOL.json"))
            s.save(OUT / f"{key}.pkl", z)
            results[key] = z
            print(json.dumps(dict(key=key, held={a: outputs["held"][a]["summary"] for a in ARMS}), ensure_ascii=False), flush=True)
    families = {}
    for family in ("rewrite", "fusion"):
        byarm = {a: [t for f in range(4) for t in results[f"{family}_fold{f}"]["outputs"]["held"][a]["traces"]] for a in ARMS}
        data = s.join_data([results[f"{family}_fold{f}"]["outputs"]["held"]["data"] for f in range(4)])
        families[family] = dict(methods={a: s.summary(t) for a, t in byarm.items()},
                                contrasts_vs_previous={a: s.contrast(byarm["previous_structural"], t) for a, t in byarm.items() if a != "previous_structural"},
                                contrasts_vs_baseline={a: s.contrast(byarm["baseline"], t) for a, t in byarm.items() if a != "baseline"},
                                uq_increment=s.contrast(byarm[PRIMARY], byarm["anchor_shrunk_uq"]),
                                probability_quality={a: s.probability_metrics(data, np.concatenate([results[f"{family}_fold{f}"]["outputs"]["held"][a]["probabilities"] for f in range(4)])) for a in ARMS},
                                certificates={a: [results[f"{family}_fold{f}"]["certificates"][a] for f in range(4)] for a in ARMS})
    for p, h in frozen["protected_hashes"].items():
        assert s.sha(p) == h
    s.dump(OUT / "summary.json", dict(completed=True, primary=PRIMARY, families=families,
            passed={a: sum(z["certificates"][a]["statistical_pass"] for z in results.values()) for a in ARMS},
            certificate_issued=False, historical_development=True, new_t5=False, new_retrieval=False,
            new_ce=False, neural_model_calls=False, protected_unchanged=True,
            calibrator_fits=32, elapsed_seconds=time.perf_counter() - began))


if __name__ == "__main__":
    main()
