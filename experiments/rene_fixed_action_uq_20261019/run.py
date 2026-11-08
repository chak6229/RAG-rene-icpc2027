"""Frozen cached-development experiment. No neural model inference."""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy import stats
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from experiments.rene_strict_confirmation_20261019 import model as upstream
from experiments.rene_strict_confirmation_20261019.common import (
    assert_disjoint, check_hashes, load, observable, protected_hashes,
    read_json, rr, save, sha, utc, write_json, canonical_bytes,
)
from experiments.rene_strict_confirmation_20261019.development import compact
from experiments.rene_fixed_action_uq_20261019 import model as heads

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/rene_fixed_action_uq_20261019"
DEV = ROOT / "results/rene_strict_confirmation_20261019/development"
FAMILIES = ("fusion", "rewrite")
UPSTREAMS = ("no_seed_uq", "seed_uq")
SEED, TOL = 261019, 1e-10


def event(stage, **kwargs):
    path = OUT / "EVENTS.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(dict(utc=utc(), stage=stage, **kwargs), ensure_ascii=False) + "\n")


def source_files():
    paths = set(HERE.glob("*.py")) | {HERE / "DESIGN.md"}
    for mod in list(sys.modules.values()):
        name = getattr(mod, "__file__", None)
        if not name:
            continue
        p = Path(name).resolve()
        if p.suffix == ".py" and p.is_file() and p.is_relative_to(ROOT):
            relative = p.relative_to(ROOT)
            if not any(part.startswith(".venv") for part in relative.parts):
                paths.add(p)
    return sorted(paths)


def freeze():
    path = OUT / "PROTOCOL.json"
    if path.exists():
        protocol = read_json(path)
        check_hashes(protocol["source_hashes"])
        check_hashes(protocol["input_hashes"])
        check_hashes(protocol["protected_hashes"])
        return protocol
    inputs = [DEV / s for s in ("items.pkl", "scores.pkl", "DESIGN.json", "MANIFEST.json")]
    inputs += [DEV / f"folds/{f}_{a}_{k}.pkl"
               for f in FAMILIES for a in UPSTREAMS for k in range(4)]
    runtime = dict(python=sys.version, executable=sys.executable,
                   numpy=np.__version__, scipy=scipy.__version__)
    protocol = dict(
        frozen_at_utc=utc(), study="fixed_action_raw_seed_risk_heads",
        historical_development=True, confirmation=False, prior_outcome_exposure=True,
        primary=dict(family="fusion", upstream="no_seed_uq", policy="fixed50",
                     contrast="full_minus_common", metric="delta_mrr"),
        support_control="support_minus_common; full_minus_support",
        robustness="seed_uq upstream; rewrite secondary",
        families=FAMILIES, upstreams=UPSTREAMS, heads=heads.HEADS, policies=heads.POLICIES,
        common_features=heads.COMMON_NAMES, full_features=heads.ALL_NAMES,
        support_columns=heads.SUPPORT, dispersion_columns=heads.DISPERSION,
        active_features={h: int(heads.feature_mask(h).sum()) for h in heads.HEADS},
        risk_threshold=.075, native_threshold=.1, thresholds_tuned=False,
        fixed_adoption="per outer fold floor(eligible*rate); ascending p_harm, then query_index",
        objective="threeway with native-harm offset; identical priors and fixed ridge",
        ridge=dict(intercept=1., verified=50., other=10.),
        intervals="descriptive, conditional fitted-model paired repository bootstrap95",
        bootstrap_repeats=10000, bootstrap_seed=SEED, reference_sesoi=.005,
        same_actions=True, same_ce=True, same_cost=True, reselect=False,
        fresh_1536_read=False, old_test_read=False, neural_model_calls=False,
        new_t5=False, new_retrieval=False, new_ce=False,
        risk_certified=False, runtime=runtime,
        source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in source_files()},
        input_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        protected_hashes=protected_hashes(),
    )
    write_json(path, protocol, immutable=True)
    event("protocol_frozen", protocol_sha256=sha(path))
    return read_json(path)


def train_data(items, inputs, traces, family):
    lookup = {e["query_index"]: e for e in items}
    ts = {t["query_index"]: t for t in traces}
    delta = np.array([lookup[q]["y"][family][ts[q]["chosen"]] for q in inputs["q"]])
    return dict(inputs, delta=delta, repo=[lookup[q]["repo"] for q in inputs["q"]])


def fixed_traces(model, items, scores):
    views = [observable(e) for e in items]
    predictions = upstream.predict_gain(model["gain"], views)
    sessions = [upstream.prepare(e, p, model["family"], model["cut"])
                for e, p in zip(views, predictions)]
    traces = [upstream.choose_trace(e, s, model["update"], score)
              for e, s, score in zip(views, sessions, scores)]
    return traces, heads.inputs(views, traces)


def check_upstream_roles(model, roles, byid):
    role_repos = {r: [byid[q]["repo"] for q in ids] for r, ids in roles.items()}
    assert_disjoint(role_repos)
    for name in ("gain", "update", "risk"):
        if set(model["roles"][name]) != set(roles[name]):
            raise ValueError("Checkpoint role mismatch")
    for name in ("gain", "update"):
        if not set(model[name]["fit_repos"]) <= set(role_repos[name]):
            raise ValueError("Upstream fit repository leakage")
        if not set(model[name]["fit_queries"]) <= set(roles[name]):
            raise ValueError("Upstream fit query leakage")


def fit_batch(family, upstream_arm, fold, roles, byid, scoreby, protocol_sha):
    key = f"{family}_{upstream_arm}_{fold}"
    path = OUT / f"folds/{key}.pkl"
    if path.exists():
        batch = load(path)
        if batch["protocol_sha256"] != protocol_sha:
            raise ValueError("Saved fold protocol mismatch")
        return batch
    checkpoint = load(DEV / f"folds/{key}.pkl")
    if checkpoint["design_sha256"] != sha(DEV / "DESIGN.json"):
        raise ValueError("Source checkpoint design mismatch")
    pipeline = checkpoint["model"]
    check_upstream_roles(pipeline, roles, byid)
    fixed, arrays = {}, {}
    for role in ("risk", "held"):
        es = [byid[q] for q in roles[role]]
        ss = [scoreby[q] for q in roles[role]]
        fixed[role], arrays[role] = fixed_traces(pipeline, es, ss)
    if [compact(t) for t in fixed["held"]] != checkpoint["outputs"]["verified"]:
        raise ValueError("Regenerated upstream action or cost mismatch")
    data = train_data([byid[q] for q in roles["risk"]], arrays["risk"], fixed["risk"], family)
    models = {h: heads.fit(data, h) for h in heads.HEADS}
    common = models["common"]
    for key_hash in ("fit_x_sha256", "fit_labels_sha256"):
        if common[key_hash] != pipeline["risk"][key_hash]:
            raise ValueError("Common head does not reproduce original fit data")
    predictions = {
        r: {h: heads.predict(models[h], arrays[r]) for h in heads.HEADS}
        for r in ("risk", "held")
    }
    old = upstream.risk.probabilities(
        pipeline["risk"], dict(x=arrays["held"]["common"], native=arrays["held"]["native"])
    )
    error = float(np.max(np.abs(old[1] - predictions["held"]["common"]["classes"]))) if len(old[0]) else 0.
    if error > 1e-10:
        raise ValueError("Original common probability replay differs")
    masks = {h: {p: heads.adoption(arrays["held"]["q"], predictions["held"][h]["harm"], p)
                 for p in heads.POLICIES} for h in heads.HEADS}
    # Existing frozen .075 decisions must be reproduced by a real common-head refit.
    old_adopted = {t["query_index"] for t in checkpoint["outputs"]["risk"] if t["accepted"]}
    new_adopted = {q for q, use in zip(arrays["held"]["q"], masks["common"]["threshold075"]) if use}
    if old_adopted != new_adopted:
        raise ValueError("Common-head post-gate action replay differs")
    batch = dict(
        family=family, upstream=upstream_arm, fold=fold, roles=roles,
        protocol_sha256=protocol_sha, source_checkpoint_sha256=sha(DEV / f"folds/{key}.pkl"),
        fixed=fixed, inputs=arrays, training=data, models=models,
        predictions=predictions, masks=masks, common_probability_max_error=error,
    )
    save(path, batch)
    event("fold_saved", family=family, upstream=upstream_arm, fold=fold,
          fit_actions=len(data["q"]), held_actions=len(arrays["held"]["q"]),
          file_sha256=sha(path))
    print(f"Fitted {family}/{upstream_arm}/fold{fold}: "
          f"risk actions={len(data['q'])}, held actions={len(arrays['held']['q'])}; "
          f"common probability replay error={error:.3g}", flush=True)
    return batch


def policy_vectors(batches, byid, head, policy):
    records = {}
    for batch in batches:
        inputs = batch["inputs"]["held"]
        mask = batch["masks"][head][policy]
        chosen = {q for q, keep in zip(inputs["q"], mask) if keep}
        for t in batch["fixed"]["held"]:
            q = t["query_index"]
            e = byid[q]
            original = rr(e["original_top100"], q)
            d = rr(t["ranking"], q) - original
            if t["accepted"]:
                if abs(d - float(e["y"][batch["family"]][t["chosen"]])) > 1e-12:
                    raise ValueError("Executed ranking label mismatch")
            elif abs(d) > 1e-12:
                raise ValueError("Rejected action changes RR")
            records[q] = dict(
                q=q, repo=e["repo"], fold=batch["fold"], eligible=bool(t["accepted"]),
                keep=q in chosen, original_rr=original, pre_delta=d,
                delta=d if q in chosen else 0., verified=t["verified"],
                **{k: int(t[k]) for k in heads.COSTS},
            )
    if len(records) != 512:
        raise ValueError("Incomplete outer-fold coverage")
    rows = [records[q] for q in sorted(records)]
    if len({r["repo"] for r in rows}) != len(rows):
        raise ValueError("Repository is not the statistical unit")
    return rows


def summarize(rows):
    d = np.array([r["delta"] for r in rows])
    pre = np.array([r["pre_delta"] for r in rows])
    adopted = np.array([r["keep"] for r in rows], bool)
    n = len(rows)
    positive_mass, negative_mass = np.maximum(d, 0.).sum(), np.maximum(-d, 0.).sum()
    pre_positive = np.maximum(pre, 0.).sum()
    harmful = int(((d < -TOL) & adopted).sum())
    return dict(
        queries=n, eligible=sum(r["eligible"] for r in rows), accepted=int(adopted.sum()),
        coverage=float(adopted.mean()), positive=int(((d > TOL) & adopted).sum()),
        harmful=harmful, zero=int(((np.abs(d) <= TOL) & adopted).sum()),
        harmful_rate=harmful / int(adopted.sum()) if adopted.any() else None,
        baseline_mrr=float(np.mean([r["original_rr"] for r in rows])),
        mrr=float(np.mean([r["original_rr"] for r in rows]) + d.mean()),
        delta_mrr=float(d.mean()), positive_rr_mass=float(positive_mass),
        negative_rr_mass=float(negative_mass),
        positive_retention=float(positive_mass / pre_positive) if pre_positive else None,
        gross_gain_mrr=float(positive_mass / n), loss_mrr=float(negative_mass / n),
        verified=sum(r["verified"] for r in rows),
        **{k: sum(r[k] for r in rows) for k in heads.COSTS},
    )


def probability_metrics(delta, p):
    delta = np.asarray(delta, float)
    y = (delta < -TOL).astype(float)
    labels = np.where(delta < -TOL, 0, np.where(delta > TOL, 2, 1))
    harm = np.clip(p[:, 0], 1e-12, 1 - 1e-12)
    n = len(y)
    if not n:
        return dict(n=0)
    return dict(
        n=n, harmful=int(y.sum()), harm_brier=float(np.mean((harm - y) ** 2)),
        harm_nll=float(np.mean(-y * np.log(harm) - (1 - y) * np.log1p(-harm))),
        harm_auc=float(roc_auc_score(y, harm)) if len(set(y)) > 1 else None,
        predicted_harm=float(harm.mean()), actual_harm=float(y.mean()),
        class_nll=float(-np.log(np.clip(p[np.arange(n), labels], 1e-12, 1.)).mean()),
        class_brier=float(np.sum((p - np.eye(3)[labels]) ** 2, axis=1).mean()),
    )


def contrast(before, after, indices):
    if [r["q"] for r in before] != [r["q"] for r in after]:
        raise ValueError("Unpaired contrast")
    a, b = (np.array([r["delta"] for r in rows]) for rows in (before, after))
    values = b - a
    harm = (b < -TOL).astype(float) - (a < -TOL).astype(float)
    loss = np.maximum(-b, 0.) - np.maximum(-a, 0.)
    ci = lambda v: np.quantile(v[indices].mean(1), [.025, .975]).tolist()
    return dict(
        delta_mrr=float(values.mean()), descriptive_bootstrap95=ci(values),
        paired_sd=float(values.std(ddof=1)),
        harmful_count_change=int((b < -TOL).sum() - (a < -TOL).sum()),
        harm_frequency_change=float(harm.mean()), harm_frequency_bootstrap95=ci(harm),
        loss_mrr_change=float(loss.mean()), loss_mrr_bootstrap95=ci(loss),
        changed_adoptions=sum(x["keep"] != y["keep"] for x, y in zip(before, after)),
    )


def evaluate(batches, byid, protocol):
    seal = read_json(OUT / "DECISION_SEAL.json")
    check_hashes(seal["fold_hashes"])
    event("held_evaluation_started", seal_sha256=sha(OUT / "DECISION_SEAL.json"))
    vectors, groups = {}, {}
    indices = np.random.default_rng(SEED).integers(512, size=(10000, 512), dtype=np.int32)
    for family in FAMILIES:
        groups[family], vectors[family] = {}, {}
        for arm in UPSTREAMS:
            batch = [batches[f"{family}_{arm}_{f}"] for f in range(4)]
            rows = {h: {p: policy_vectors(batch, byid, h, p) for p in heads.POLICIES}
                    for h in heads.HEADS}
            vectors[family][arm] = rows
            curves = {h: {p: summarize(rows[h][p]) for p in heads.POLICIES} for h in heads.HEADS}
            base = curves["common"]["all"]
            for p in ("fixed25", "fixed50", "fixed75", "all"):
                if len({curves[h][p]["accepted"] for h in heads.HEADS}) != 1:
                    raise ValueError("Adoption counts differ at fixed coverage")
            deltas, probability_rows = [], {h: [] for h in heads.HEADS}
            for b in batch:
                traces = {t["query_index"]: t for t in b["fixed"]["held"]}
                for q in b["inputs"]["held"]["q"]:
                    t = traces[q]
                    deltas.append(rr(t["ranking"], q) - rr(byid[q]["original_top100"], q))
                for h in heads.HEADS:
                    probability_rows[h].append(b["predictions"]["held"][h]["classes"])
            groups[family][arm] = dict(
                pre_gate=base, methods=curves,
                probability_quality={h: probability_metrics(deltas, np.concatenate(probability_rows[h]))
                                     for h in heads.HEADS},
                contrasts={
                    p: {f"{after}_minus_{before}": contrast(rows[before][p], rows[after][p], indices)
                        for after, before in (("full", "common"), ("support", "common"), ("full", "support"))}
                    for p in heads.POLICIES
                },
                folds=[dict(
                    fold=b["fold"], risk_actions=len(b["training"]["q"]),
                    risk_class_counts=b["models"]["common"]["counts"],
                    held_actions=len(b["inputs"]["held"]["q"]),
                    adopted={h: {p: int(b["masks"][h][p].sum()) for p in heads.POLICIES}
                             for h in heads.HEADS},
                ) for b in batch],
            )
    primary = groups["fusion"]["no_seed_uq"]["contrasts"]["fixed50"]["full_minus_common"]
    sd = max(.05, primary["paired_sd"] * math.sqrt(511 / stats.chi2.ppf(.2, 511)))
    z = stats.norm.ppf(.975) + stats.norm.ppf(.8)
    required = int(math.ceil(z * z * sd * sd / .005 ** 2))
    planning = dict(
        status="sample planning only; future independent protocol required",
        reference_effect=.005, power=.8, two_sided_alpha=.05,
        sd_upper80_floor05=sd, required_unrounded=required,
        planned_queries=int(math.ceil(max(1024, required) / 128) * 128),
        method="paired-normal approximation with development SD upper80 and floor .05",
        risk_certification_power=False, training_uncertainty_included=False,
    )
    result = dict(
        completed=True, protocol_sha256=sha(OUT / "PROTOCOL.json"),
        decision_seal_sha256=sha(OUT / "DECISION_SEAL.json"),
        primary=protocol["primary"], primary_result=primary, families=groups,
        future_power_planning=planning, confirmation=False, historical_development=True,
        risk_certified=False, new_t5=False, new_retrieval=False, new_ce=False,
        protected_unchanged=protected_hashes() == protocol["protected_hashes"],
    )
    save(OUT / "PER_QUERY.pkl", vectors)
    write_json(OUT / "RESULTS.json", result, immutable=True)
    event("held_evaluation_completed", result_sha256=sha(OUT / "RESULTS.json"))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-only", action="store_true")
    args = ap.parse_args()
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        protocol = freeze()
        if args.freeze_only:
            print("Protocol frozen: " + sha(OUT / "PROTOCOL.json"), flush=True)
            return
        if (OUT / "RESULTS.json").exists():
            print("Completed results already exist; hashes checked; no refit.", flush=True)
            return
        items, scores = load(DEV / "items.pkl"), load(DEV / "scores.pkl")
        byid, scoreby = ({int(e["query_index"]): e for e in rows} for rows in (items, scores))
        roles = read_json(DEV / "DESIGN.json")["folds"]
        batches = {}
        for family in FAMILIES:
            for arm in UPSTREAMS:
                for f in range(4):
                    batches[f"{family}_{arm}_{f}"] = fit_batch(
                        family, arm, f, roles[f], byid, scoreby, sha(OUT / "PROTOCOL.json")
                    )
        seal_path = OUT / "DECISION_SEAL.json"
        if not seal_path.exists():
            write_json(seal_path, dict(
                sealed_at_utc=utc(), protocol_sha256=sha(OUT / "PROTOCOL.json"),
                folds=16, risk_head_fits=48,
                held_outcome_metrics_computed=False,
                local_workflow_seal=True, fresh_confirmation=False,
                fold_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((OUT / "folds").glob("*.pkl"))},
            ), immutable=True)
            event("all_predictions_and_adoptions_sealed", seal_sha256=sha(seal_path))
        result = evaluate(batches, byid, protocol)
        check_hashes(protocol["source_hashes"])
        check_hashes(protocol["input_hashes"])
        check_hashes(protocol["protected_hashes"])
        write_json(OUT / "RUN_STATUS.json", dict(
            completed=True, elapsed_seconds=time.perf_counter() - started,
            fits=48, protected_unchanged=True, neural_model_calls=False,
            audit_pending=True,
        ), immutable=True)
        print("PRIMARY " + str(result["primary_result"]), flush=True)


if __name__ == "__main__":
    main()
