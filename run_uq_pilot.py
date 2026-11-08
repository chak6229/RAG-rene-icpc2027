"""Frozen, valid-only UQ pilot. Never reads test or updates V5 stable.

Three existing repository-disjoint development folds; fixed hyperparameters.
Coverage-matched curves are descriptive evaluation rankings, not fitted gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error

from prepare_vac_mqr_cv_folds import repository
from vac_mqr import ENHANCED_RETRIEVAL_FEATURES, UTILITY_FEATURES


FEATURES = list(ENHANCED_RETRIEVAL_FEATURES)
ROLES = {"utility_train": 1200, "lcb_calibration": 400,
         "selection_valid": 300, "risk_calibration": 500, "new_holdout": 600}
ALPHA = .10


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def grouped_folds(repo, count=3, seed=101):
    rng = np.random.default_rng(seed)
    names = np.unique(repo); rng.shuffle(names)
    sizes = Counter(repo); bins = [[] for _ in range(count)]; loads = [0] * count
    for name in sorted(names, key=lambda n: -sizes[n]):
        f = int(np.argmin(loads)); bins[f].append(name); loads[f] += sizes[name]
    return [np.flatnonzero(np.isin(repo, group)) for group in bins]


def finite_quantile(values, alpha=ALPHA):
    values = np.sort(np.asarray(values))
    k = math.ceil((len(values) + 1) * (1 - alpha))
    return float(values[k - 1]) if 0 < k <= len(values) else float("inf")


def sample_fold(path, seed):
    rows = [json.loads(s) for s in path.open(encoding="utf-8") if s.strip()]
    assert not any(r["split_role"] == "test" for r in rows)
    rng = np.random.default_rng(seed)
    sampled = {}
    for role, count in ROLES.items():
        pool = [r for r in rows if r["split_role"] == role]
        indices = rng.choice(len(pool), min(count, len(pool)), replace=False)
        sampled[role] = [pool[i] for i in sorted(indices)]
    repos = {k: {repository(r["url"]) for r in v} for k, v in sampled.items()}
    assert all(not repos[a] & repos[b] for a in repos for b in repos if a != b)
    return sampled


def flatten(rows, action):
    x, y, q, rep, base_rr, rank = [], [], [], [], [], []
    for row in rows:
        for c in row["candidates"]:
            if not c["quality_pass"]:
                continue
            x.append([c["features"][f] for f in FEATURES])
            rr = c["replacement_rr_at_100"] if action == "rewrite" else c["rr_at_100"]
            y.append(rr - row["original_rr_at_100"])
            q.append(row["query_index"]); rep.append(repository(row["url"]))
            base_rr.append(row["original_rr_at_100"]); rank.append(c["rank"])
    return {"x": np.asarray(x, dtype=float), "y": np.asarray(y),
            "q": np.asarray(q), "repo": np.asarray(rep), "rank": np.asarray(rank),
            "base_rr": np.asarray(base_rr), "query_count": len(rows),
            "query_repos": [repository(r["url"]) for r in rows],
            "query_ids": [r["query_index"] for r in rows]}


def booster(x, y, seed=101, objective="regression", weight=None):
    params = dict(objective=objective, verbosity=-1, num_leaves=7,
                  min_data_in_leaf=30, learning_rate=.04, lambda_l2=1,
                  num_threads=2, seed=seed, deterministic=True, force_col_wise=True)
    if objective == "multiclass":
        params["num_class"] = 3
    return lgb.train(params, lgb.Dataset(x, label=y, weight=weight), num_boost_round=120)


class Signed:
    def fit(self, x, y, repo, seed):
        oof = np.zeros(len(y)); folds = grouped_folds(repo, seed=seed)
        for val in folds:
            tr = np.setdiff1d(np.arange(len(y)), val)
            oof[val] = booster(x[tr], y[tr], seed).predict(x[val])
        self.mean = booster(x, y, seed)
        self.scale = booster(x, np.log(np.abs(y - oof) + .001), seed)
        self.harm = booster(x, (y < 0).astype(int), seed, "binary")
        self.constant = max(float(np.mean(np.abs(y - oof))), .001)
        self.oof = oof
        return self

    def predict(self, x, constant=False):
        mu = self.mean.predict(x)
        s = np.full(len(x), self.constant) if constant else np.exp(self.scale.predict(x))
        return dict(mu=mu, s=np.maximum(s, .001), harm=self.harm.predict(x))


class Hurdle:
    def fit(self, x, y, seed, weight=None):
        self.sign = booster(x, np.sign(y).astype(int) + 1, seed, "multiclass", weight)
        self.mag = []; self.var = []
        for sign in (-1, 1):
            keep = sign * y > 0
            model = booster(x[keep], np.abs(y[keep]), seed,
                            weight=weight[keep] if weight is not None else None)
            self.mag.append(model)
            self.var.append(float(np.mean((np.abs(y[keep]) - model.predict(x[keep])) ** 2)))
        return self

    def predict(self, x):
        p = self.sign.predict(x)
        neg, pos = [np.clip(m.predict(x), 0, 1) for m in self.mag]
        mu = p[:, 2] * pos - p[:, 0] * neg
        second = p[:, 2] * (pos ** 2 + self.var[1]) + p[:, 0] * (neg ** 2 + self.var[0])
        return dict(mu=mu, s=np.sqrt(np.maximum(second - mu ** 2, 1e-6)),
                    harm=p[:, 0], sign=p)


class Ensemble:
    def fit(self, x, y, repo, seed):
        rng = np.random.default_rng(seed); names = np.unique(repo); self.models = []
        for member in range(5):
            counts = Counter(rng.choice(names, len(names), replace=True))
            weight = np.asarray([counts[r] for r in repo], dtype=float)
            keep = weight > 0
            self.models.append(Hurdle().fit(x[keep], y[keep], seed + member, weight[keep]))
        return self

    def predict(self, x):
        p = [m.predict(x) for m in self.models]
        means = np.stack([v["mu"] for v in p]); mu = means.mean(axis=0)
        epistemic = means.std(axis=0)
        variance = np.mean([v["s"] ** 2 for v in p], axis=0) + epistemic ** 2
        # Disagreement over the sign of net expected gain, not candidate diversity.
        fraction = np.mean(means > 0, axis=0)
        return dict(mu=mu, s=np.sqrt(variance), harm=np.mean([v["harm"] for v in p], axis=0),
                    epistemic=epistemic, sign_disagreement=4 * fraction * (1 - fraction),
                    members=means)


class QuantileForest:
    """Leaf-weighted empirical outcome distribution (not quantiles of tree means)."""
    def fit(self, x, y, seed):
        self.model = RandomForestRegressor(n_estimators=48, max_depth=8,
                                           min_samples_leaf=20, n_jobs=2, random_state=seed)
        self.model.fit(x, y); self.y = y; self.sort = np.argsort(y)
        leaves = self.model.apply(x); self.maps = []
        for col in range(leaves.shape[1]):
            self.maps.append({v: np.flatnonzero(leaves[:, col] == v) for v in np.unique(leaves[:, col])})
        return self

    def predict(self, x):
        leaves = self.model.apply(x); out = {k: [] for k in ("mu", "s", "harm", "raw_lower")}
        for leaf in leaves:
            weight = np.zeros(len(self.y))
            for t, node in enumerate(leaf):
                indices = self.maps[t][node]
                weight[indices] += 1 / (len(self.maps) * len(indices))
            mu = float(weight @ self.y)
            out["mu"].append(mu)
            out["s"].append(max(float(np.sqrt(weight @ (self.y - mu) ** 2)), .001))
            out["harm"].append(float(weight @ (self.y < 0)))
            cdf = np.cumsum(weight[self.sort])
            out["raw_lower"].append(float(self.y[self.sort[min(np.searchsorted(cdf, .1), len(cdf)-1)]]))
        return {k: np.asarray(v) for k, v in out.items()}


class NeuralDistribution:
    def fit(self, x, y, kind, seed, directory):
        import torch
        from torch import nn
        from torch.nn import functional as F
        torch.set_num_threads(2); torch.manual_seed(seed)
        self.kind = kind; self.mean = x.mean(axis=0); self.std = x.std(axis=0)
        self.std[self.std < 1e-6] = 1
        self.target_scale = max(float(y.std()), .05)
        self.net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.Tanh(),
                                 nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 4))
        opt = torch.optim.Adam(self.net.parameters(), lr=.001, weight_decay=1e-4)
        tx = torch.tensor((x-self.mean)/self.std, dtype=torch.float32)
        ty = torch.tensor(y/self.target_scale, dtype=torch.float32)
        rng = np.random.default_rng(seed); history = []
        for epoch in range(100):
            order = rng.permutation(len(y)); losses = []
            for start in range(0, len(order), 256):
                ix = order[start:start+256]; raw = self.net(tx[ix]); location = raw[:, 0]
                if kind == "gaussian":
                    scale = F.softplus(raw[:, 1]) + .01
                    loss = -torch.distributions.Normal(location, scale).log_prob(ty[ix])
                elif kind == "student_t":
                    scale = F.softplus(raw[:, 1]) + .01; df = F.softplus(raw[:, 2]) + 2.1
                    loss = -torch.distributions.StudentT(df, location, scale).log_prob(ty[ix])
                else:
                    nu = F.softplus(raw[:, 1]) + .01
                    alpha = F.softplus(raw[:, 2]) + 1.1
                    beta = F.softplus(raw[:, 3]) + .01
                    scale = torch.sqrt(beta*(1+nu)/(alpha*nu))
                    loss = -torch.distributions.StudentT(2*alpha, location, scale).log_prob(ty[ix])
                    if kind == "nig_regularized":
                        loss = loss + .01 * torch.abs(ty[ix]-location) * (2*nu+alpha)
                loss = loss.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss in {kind}")
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 5)
                opt.step(); losses.append(float(loss.detach()))
            if epoch in (0, 24, 49, 74, 99):
                history.append({"epoch": epoch+1, "training_loss": float(np.mean(losses))})
        torch.save(dict(state_dict=self.net.state_dict(), feature_names=FEATURES,
                        input_mean=self.mean, input_std=self.std, target_scale=self.target_scale,
                        kind=kind, seed=seed, history=history), directory / f"{kind}.pt")
        self.history = history
        return self

    def predict(self, x):
        import torch
        from torch.nn import functional as F
        with torch.no_grad():
            raw = self.net(torch.tensor((x-self.mean)/self.std, dtype=torch.float32))
            mu = raw[:, 0].numpy()*self.target_scale
            if self.kind == "gaussian":
                s = (F.softplus(raw[:, 1])+.01).numpy()*self.target_scale
                dist = stats.norm(loc=mu, scale=s); variance = s**2
                extra = {}
            elif self.kind == "student_t":
                scale = (F.softplus(raw[:, 1])+.01).numpy()*self.target_scale
                df = (F.softplus(raw[:, 2])+2.1).numpy()
                dist = stats.t(df, loc=mu, scale=scale); variance = scale**2*df/(df-2)
                extra = {}
            else:
                nu = (F.softplus(raw[:, 1])+.01).numpy()
                alpha = (F.softplus(raw[:, 2])+1.1).numpy()
                beta = (F.softplus(raw[:, 3])+.01).numpy()*self.target_scale**2
                scale = np.sqrt(beta*(1+nu)/(alpha*nu)); df = 2*alpha
                dist = stats.t(df, loc=mu, scale=scale)
                aleatoric = beta/(alpha-1); epistemic = aleatoric/nu
                variance = aleatoric+epistemic
                extra = dict(nig_aleatoric_variance=aleatoric, nig_epistemic_variance=epistemic)
        return dict(mu=mu, s=np.sqrt(variance), harm=dist.cdf(0),
                    raw_lower=dist.ppf(.1), outside_global_support=dist.cdf(-1)+dist.sf(1), **extra)


def pick(data, pred, score="mu"):
    # Select a single frozen predicted first choice; no trying another after rejection.
    return np.asarray([ix[np.argmax(pred[score][ix])]
                       for q in np.unique(data["q"])
                       for ix in [np.flatnonzero(data["q"] == q)]], dtype=int)


def evaluate_prediction(data, pred):
    y = data["y"]; mu = pred["mu"]; selected = pick(data, pred)
    def ap(label, scores):
        return float(average_precision_score(label, scores)) if np.any(label) else None
    def correlation(a, b):
        v = stats.spearmanr(a, b).statistic
        return float(v) if np.isfinite(v) else None
    out = dict(mae=float(mean_absolute_error(y, mu)),
               positive_auprc=ap(y > 0, mu), harmful_auprc=ap(y < 0, pred["harm"]),
               harmful_brier=float(np.mean((pred["harm"]-(y < 0))**2)),
               scale_error_spearman=correlation(pred["s"], np.abs(y-mu)),
               first_choice_actual_mean_gain=float(np.mean(y[selected])))
    if "raw_lower" in pred:
        out["raw_lower_violation_all"] = float(np.mean(y < pred["raw_lower"]))
        out["raw_lower_violation_first"] = float(np.mean(y[selected] < pred["raw_lower"][selected]))
    if "epistemic" in pred:
        out["ensemble_disagreement_error_spearman"] = correlation(pred["epistemic"], np.abs(y-mu))
    if "outside_global_support" in pred:
        out["mean_probability_outside_minus1_plus1"] = float(np.mean(pred["outside_global_support"]))
    return out


def clustered_ci(data, query_values, seed, samples=500):
    sums = {}; counts = Counter(data["query_repos"])
    for q, repo in zip(data["query_ids"], data["query_repos"]):
        sums[repo] = sums.get(repo, 0.) + query_values.get(q, 0.)
    names = list(counts); rng = np.random.default_rng(seed); means = []
    for _ in range(samples):
        drawn = rng.choice(names, len(names), replace=True)
        means.append(sum(sums[r] for r in drawn)/sum(counts[r] for r in drawn))
    return [float(v) for v in np.quantile(means, [.025, .975])]


def decision_metrics(data, indices, seed):
    indices = np.asarray(indices, dtype=int); values = data["y"][indices]
    selected_values = {int(data["q"][i]): float(data["y"][i]) for i in indices}
    return dict(accepted=int(len(indices)), coverage=len(indices)/data["query_count"],
                delta_mrr=float(values.sum()/data["query_count"]),
                delta_mrr_repo_bootstrap_95_ci=clustered_ci(data, selected_values, seed),
                accepted_mean_gain=float(values.mean()) if len(values) else None,
                harmful_rate=float(np.mean(values < 0)) if len(values) else None,
                mean_loss_per_accepted=float(np.maximum(-values, 0).mean()) if len(values) else None)


def calibrate_and_evaluate(cal, pred_cal, ev, pred_ev, seed):
    selected = pick(cal, pred_cal)
    residual = (pred_cal["mu"]-cal["y"])/np.maximum(pred_cal["s"], .001)
    # First-choice calibration matches the first-choice-only deployed policy.
    k = max(finite_quantile(residual[selected]), 0.)
    ev_selected = pick(ev, pred_ev)
    lower = pred_ev["mu"]-k*np.maximum(pred_ev["s"], .001)
    pred_ev["lcb"] = lower
    out = dict(normalized_quantile=k,
               lower_violation_first=float(np.mean(ev["y"][ev_selected] < lower[ev_selected])),
               mean_first_bound_width=float(np.mean(pred_ev["mu"][ev_selected]-lower[ev_selected])),
               lcb_positive=decision_metrics(ev, ev_selected[lower[ev_selected] > 0], seed),
               coverage_curves={})
    # Rank first choices at fixed coverage without using evaluation labels.
    # These are descriptive budget curves, not thresholds learned from the eval set.
    for score in ("mu", "lcb", "negative_harm"):
        values = -pred_ev["harm"] if score == "negative_harm" else pred_ev[score]
        ordered = ev_selected[np.argsort(-values[ev_selected], kind="stable")]
        out["coverage_curves"][score] = {}
        for coverage in (.02, .05, .10, .20):
            n = min(len(ordered), max(1, round(ev["query_count"]*coverage)))
            out["coverage_curves"][score][str(coverage)] = decision_metrics(ev, ordered[:n], seed)
    return out


def evidence_acquisition(train, ev, seed, directory):
    base_cols = [FEATURES.index(f) for f in UTILITY_FEATURES]
    x, y, repo = train["x"], train["y"], train["repo"]
    base_oof = np.zeros(len(y)); full_oof = np.zeros(len(y)); scale_oof = np.zeros(len(y))
    for val in grouped_folds(repo, seed=seed):
        tr = np.setdiff1d(np.arange(len(y)), val)
        b = Hurdle().fit(x[tr][:, base_cols], y[tr], seed)
        f = Hurdle().fit(x[tr], y[tr], seed)
        pb = b.predict(x[val][:, base_cols]); pf = f.predict(x[val])
        base_oof[val] = pb["mu"]; scale_oof[val] = pb["s"]; full_oof[val] = pf["mu"]
    def query_features(data, base_pred):
        feats, indices = [], []
        for q in np.unique(data["q"]):
            ix = np.flatnonzero(data["q"] == q)
            order = ix[np.argsort(-base_pred["mu"][ix])]; best = order[0]
            feats.append(np.r_[data["x"][best, base_cols], base_pred["mu"][best],
                               base_pred["s"][best], base_pred["mu"][best] -
                               (base_pred["mu"][order[1]] if len(order)>1 else 0)])
            indices.append((q, best, ix))
        return np.asarray(feats), indices
    meta_x, tr_indices = query_features(train, dict(mu=base_oof, s=scale_oof))
    labels = []
    for _, best, ix in tr_indices:
        full_best = ix[np.argmax(full_oof[ix])]
        base_value = y[best] if base_oof[best] > 0 else 0.
        full_value = y[full_best] if full_oof[full_best] > 0 else 0.
        labels.append(full_value-base_value)
    acquisition = booster(meta_x, np.asarray(labels), seed)
    base = Hurdle().fit(x[:, base_cols], y, seed); full = Hurdle().fit(x, y, seed)
    pb = base.predict(ev["x"][:, base_cols]); pf = full.predict(ev["x"])
    meta_ev, ev_indices = query_features(ev, pb); score = acquisition.predict(meta_ev)
    rng = np.random.default_rng(seed); random_order = rng.permutation(len(ev_indices))
    uncertainty = np.asarray([pb["s"][best] for _, best, _ in ev_indices])
    rankings = dict(learned_expected_gain=np.argsort(-score),
                    uncertainty=np.argsort(-uncertainty), random=random_order)
    result = {}
    for mode in ("never", "always", *rankings):
        fractions = (0.,) if mode == "never" else (1.,) if mode == "always" else (.2, .5)
        result[mode] = {}
        for fraction in fractions:
            acquired = set(range(len(ev_indices))) if mode == "always" else set()
            if mode in rankings:
                acquired = set(rankings[mode][:round(len(ev_indices)*fraction)].tolist())
            chosen = []; oracle_improvement = 0
            for local, (_, best, ix) in enumerate(ev_indices):
                if local in acquired:
                    best = ix[np.argmax(pf["mu"][ix])]; mu = pf["mu"][best]
                else:
                    mu = pb["mu"][best]
                if mu > 0:
                    chosen.append(best)
            result[mode][str(fraction)] = dict(
                acquisition_fraction=len(acquired)/ev["query_count"],
                **decision_metrics(ev, chosen, seed))
    joblib.dump(dict(base=base, full=full, acquisition=acquisition), directory / "acquisition.joblib")
    return dict(proxy="Reveal cached enhanced Top-100 features, never new retrieval; no latency claim",
                policy="positive predicted mean with original fallback; no safety certification",
                results=result)


def run_action(sampled, action, seed, directory):
    directory.mkdir(parents=True, exist_ok=True)
    data = {role: flatten(rows, action) for role, rows in sampled.items()}
    tr = data["utility_train"]; cal = data["lcb_calibration"]; ev = data["new_holdout"]
    x, y, repo = tr["x"], tr["y"], tr["repo"]
    preds = {}; models = {}
    signed = Signed().fit(x, y, repo, seed)
    models["signed_constant"] = signed; models["signed_oof_scale"] = signed
    models["hurdle"] = Hurdle().fit(x, y, seed)
    models["hurdle_ensemble"] = Ensemble().fit(x, y, repo, seed)
    models["quantile_forest"] = QuantileForest().fit(x, y, seed)
    for kind in ("gaussian", "student_t", "nig_unregularized", "nig_regularized"):
        print(f"  {action} fit {kind}", flush=True)
        models[kind] = NeuralDistribution().fit(x, y, kind, seed, directory)
    result = {}
    for name, model in models.items():
        print(f"  {action} evaluate {name}", flush=True)
        pred = {role: (model.predict(d["x"], constant=name == "signed_constant")
                      if isinstance(model, Signed) else model.predict(d["x"]))
                for role, d in data.items() if role != "utility_train"}
        metrics = evaluate_prediction(ev, pred["new_holdout"])
        bounds = calibrate_and_evaluate(cal, pred["lcb_calibration"], ev, pred["new_holdout"], seed)
        # Preselect ONE gate on selection-valid; inspect it once on independent risk role.
        k = bounds["normalized_quantile"]
        for role in ("selection_valid", "risk_calibration"):
            pred[role]["lcb"] = pred[role]["mu"] - k * np.maximum(pred[role]["s"], .001)
        selection = data["selection_valid"]; p = pred["selection_valid"]
        first = pick(selection, p); choices = []
        for threshold in (.05, .10, .20, .40, 1.):
            accept = first[(p["lcb"][first] > 0) & (p["harm"][first] <= threshold)]
            gain = float(selection["y"][accept].sum()/selection["query_count"])
            harm = float(np.mean(selection["y"][accept] < 0)) if len(accept) else 0.
            if harm <= .1 and gain >= 0:
                choices.append((gain, len(accept), threshold))
        threshold = max(choices)[2] if choices else -1.
        risk = data["risk_calibration"]; pr = pred["risk_calibration"]; ri = pick(risk, pr)
        ra = ri[(pr["lcb"][ri] > 0) & (pr["harm"][ri] <= threshold)]
        harms = int(np.sum(risk["y"][ra] < 0)); n = len(ra)
        cp = float(stats.beta.ppf(.95, harms+1, n-harms)) if n > harms else 1.
        permitted = n > 0 and cp <= .1
        ei = pick(ev, pred["new_holdout"]); pe = pred["new_holdout"]
        ea = ei[(pe["lcb"][ei] > 0) & (pe["harm"][ei] <= threshold)] if permitted else []
        result[name] = dict(prediction=metrics, calibrated=bounds,
                            risk_diagnostic=dict(frozen_harm_threshold=threshold,
                                calibration_accepted=n, calibration_harms=harms,
                                query_binomial_cp95_upper=cp, passes_query_iid_diagnostic=permitted,
                                caveat="repository dependence: not a certified risk guarantee",
                                evaluation=decision_metrics(ev, ea, seed)))
        if isinstance(model, NeuralDistribution):
            result[name]["training_history"] = model.history
        else:
            joblib.dump(model, directory / f"{name}.joblib")
        preds[name] = pred["new_holdout"]
    # Evaluation predictions contain labels solely for analysis; inference features exclude them.
    arrays = dict(query_index=ev["q"], candidate_rank=ev["rank"], delta_rr=ev["y"])
    for name, pred in preds.items():
        for key in ("mu", "s", "harm", "lcb", "epistemic", "sign_disagreement"):
            if key in pred:
                arrays[f"{name}__{key}"] = pred[key]
    np.savez_compressed(directory / "evaluation_predictions.npz", **arrays)
    oracle = []
    for q in np.unique(ev["q"]):
        ix = np.flatnonzero(ev["q"] == q); best = ix[np.argmax(ev["y"][ix])]
        if ev["y"][best] > 0:
            oracle.append(best)
    result["oracle"] = decision_metrics(ev, oracle, seed)
    result["acquisition_proxy"] = evidence_acquisition(tr, ev, seed, directory)
    dump(directory / "metrics.json", result)
    return result


def make_report(summary, directory):
    text = ["# VAC-MQR small-sample UQ pilot", "",
            "Development-only, reused valid-derived folds. Not an independent test or risk guarantee.",
            "Fixed Top-3; rewrite and fusion labels are evaluated separately. No T5/retriever training.",
            "", "## Mean results across three samples", "",
            "| Action | Method | MAE | Error-scale Spearman | Delta-MRR at 10% coverage, mean ranking | Harm at 10% | Delta-MRR at 10%, LCB ranking | LCB first-choice violation | LCB>0 coverage |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    aggregates = {}
    names = ["signed_constant", "signed_oof_scale", "hurdle", "hurdle_ensemble",
             "quantile_forest", "gaussian", "student_t", "nig_unregularized", "nig_regularized"]
    for action in ("rewrite", "fusion"):
        aggregates[action] = {}
        for name in names:
            values = [v[action][name] for v in summary["runs"].values()]
            def mean(path):
                out=[]
                for v in values:
                    for k in path: v=v[k]
                    if v is not None: out.append(v)
                return float(np.mean(out)) if out else None
            metrics = dict(mae=mean(["prediction", "mae"]),
                error_scale_spearman=mean(["prediction", "scale_error_spearman"]),
                delta_mrr_at10_mean=mean(["calibrated", "coverage_curves", "mu", "0.1", "delta_mrr"]),
                harm_at10_mean=mean(["calibrated", "coverage_curves", "mu", "0.1", "harmful_rate"]),
                delta_mrr_at10_lcb=mean(["calibrated", "coverage_curves", "lcb", "0.1", "delta_mrr"]),
                first_lcb_violation=mean(["calibrated", "lower_violation_first"]),
                lcb_positive_coverage=mean(["calibrated", "lcb_positive", "coverage"]),
                lcb_positive_delta=mean(["calibrated", "lcb_positive", "delta_mrr"]))
            aggregates[action][name] = metrics
            numbers = [metrics[k] for k in ("mae", "error_scale_spearman", "delta_mrr_at10_mean",
                "harm_at10_mean", "delta_mrr_at10_lcb", "first_lcb_violation", "lcb_positive_coverage")]
            text.append(f"| {action} | {name} | " + " | ".join("NA" if x is None else f"{x:.6f}" for x in numbers)+" |")
    text += ["", "## Limits", "",
             "- 1,200 train / 400 bound-calibration / 300 selection / 500 risk / 600 evaluation queries per sample.",
             "- Three existing folds and three seeds are confounded; this is not a seed-only stability study.",
             "- Evaluation samples can overlap between runs; runs are not independent replications.",
             "- All hyperparameters fixed before evaluation; no post-result tuning.",
             "- Risk diagnostic uses ONE selection-valid-chosen policy on a separate role, but its binomial bound does not handle repository dependence.",
             "- Zero acceptance yields undefined accepted harm, not zero observed risk.",
             "- Acquisition reveals cached Top-100 features; it does not validate actual retrieval costs or safety.",
             "- Bootstrap intervals cluster repositories; evaluation coverage curves use label-free rankings, not deployment-certified thresholds."]
    (directory / "REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    dump(directory / "aggregate.json", aggregates)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise RuntimeError("Completed result exists; choose a new output directory")
    started=time.time(); summary=dict(test_used=False, development_only=True, runs={})
    config=dict(seed_by_fold={0:101,1:202,2:303}, role_caps=ROLES, features=FEATURES,
                fixed_hyperparameters=True, lcb_alpha=ALPHA, lcb_policy="predicted-first-only",
                label_policies=["rewrite_only", "original_rewrite_equal_weight_rrf"],
                environment=".venv-models", protocol="small-sample-dev-v1")
    dump(args.output / "config.json", config)
    for fold, seed in ((0,101),(1,202),(2,303)):
        print(f"FOLD {fold} seed {seed}", flush=True)
        path=Path(f"results/vac_mqr_v5/enhanced_cv_folds/fold_{fold}.jsonl")
        sampled=sample_fold(path, seed); directory=args.output / f"fold_{fold}"
        directory.mkdir(exist_ok=True)
        manifest=dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      seed=seed, roles={k:[r["query_index"] for r in v] for k,v in sampled.items()},
                      repository_counts={k:len({repository(r["url"]) for r in v}) for k,v in sampled.items()},
                      repository_overlap=0, test_used=False)
        dump(directory / "sample_manifest.json", manifest)
        run={}
        for action in ("rewrite", "fusion"):
            dump(args.output / "status.json", dict(status="running", fold=fold, action=action,
                                                   elapsed_seconds=time.time()-started, test_used=False))
            run[action]=run_action(sampled, action, seed, directory / action)
        summary["runs"][str(fold)]=run
        dump(args.output / "checkpoint.json", summary)
    summary["elapsed_seconds"]=time.time()-started
    make_report(summary,args.output); dump(args.output / "summary.json", summary)
    dump(args.output / "status.json", dict(status="completed", elapsed_seconds=time.time()-started,
                                           test_used=False))
    print(f"COMPLETED {summary['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
