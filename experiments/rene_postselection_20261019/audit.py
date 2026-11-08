"""Independent model replay, exact-RR accounting and development-only acceptance."""
from __future__ import annotations

import copy
import hashlib
from collections import Counter, defaultdict
from fractions import Fraction

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import (ALGORITHMS, BLOCKS, CALIBRATORS, OUT, ROOT, load_data, now, read, readl,
                     record, require, sha, verify, write)


def independently_select(data, query_ids, raw):
    result = {}
    for q in query_ids:
        ids = data.query_rows[q]
        values = {i: 0.0 if data.rows[i]["structural_zero"] else min(1.0, max(-1.0, float(raw[i]))) for i in ids}
        ordered = sorted(ids, key=lambda i: (-values[i], data.rows[i]["entr_rank"], data.rows[i]["position"]))
        i = ordered[0]
        runner = max(0.0, values[ordered[1]])
        result[q] = {"query_id": q, "index": i, "position": data.rows[i]["position"], "fold": data.fold_map[q],
                     "score": values[i], "runner_up": runner, "gap": values[i] - runner,
                     "eligible": values[i] > 0 and not data.rows[i]["structural_zero"],
                     "structural_zero": data.rows[i]["structural_zero"]}
    return result


def independent_matrix(data, chosen, method):
    if method in ("AFFINE", "ISOTONIC"):
        return np.asarray([[r["score"]] for r in chosen], dtype=float).reshape(len(chosen), 1)
    return np.asarray([[r["score"], max(0.0, r["runner_up"]), r["score"] - r["runner_up"],
                        *data.x[r["index"]][19:]] for r in chosen], dtype=float).reshape(len(chosen), 55)


def independent_cal_predict(model, x, method):
    use = x[:, 0] if isinstance(model, IsotonicRegression) else x
    return np.maximum(-1.0, np.minimum(1.0, model.predict(use)))


def audit_retriever(retriever, p):
    data = load_data(retriever)
    cached = readl(OUT / f"candidate_predictions/{retriever}.jsonl")
    require(len(cached) == len(data.rows), "candidate prediction denominator differs")
    bases = [f"{block}_{algorithm}" for block in BLOCKS for algorithm in ALGORITHMS]
    raw = {name: np.asarray([r["raw"][name] for r in cached]) for name in bases}
    for i, row in enumerate(cached):
        require(row["index"] == i and row["query_id"] == data.rows[i]["query_id"]
                and row["position"] == data.rows[i]["position"] and row["fold"] == int(data.folds[i]), "candidate row drift")
        require(set(row["raw"]) == set(bases), "missing model prediction")
    stored = readl(OUT / f"winners/{retriever}.jsonl")
    require(Counter(r["base"] for r in stored) == Counter({b: len(data.qids) for b in bases}), "winner denominator differs")
    stored_winners = {(r["base"], r["query_id"]): r for r in stored}
    require(len(stored_winners) == len(stored), "duplicate winner")
    independently_chosen = {b: independently_select(data, data.qids, raw[b]) for b in bases}
    for base, chosen in independently_chosen.items():
        for q, row in chosen.items():
            require(all(stored_winners[base, q][key] == value for key, value in row.items()), "independent winner mismatch")

    fits = read(OUT / f"TRAINING_LEDGER_{retriever}.json")["fits"]
    require(Counter(r["kind"] for r in fits) == Counter({"outer": 50, "inner": 40, "calibrator": 30}), "fit manifest incomplete")
    max_error, inner_arrays = 0.0, {}
    reloads = Counter()
    for info in fits:
        outer = info["outer"]
        model_path = ROOT / info["model_path"]
        require(sha(model_path) == info["model_sha256"], "model changed")
        model = joblib.load(model_path)
        kind, algorithm = info["kind"], info["algorithm"]
        reloads[kind] += 1
        if kind == "calibrator":
            continue
        if kind == "outer":
            train = np.flatnonzero(data.folds != outer)
            held = np.flatnonzero(data.folds == outer)
            expected = raw[f'{info["block"]}_{algorithm}'][held]
        else:
            inner = info["inner"]
            train = np.flatnonzero((data.folds != outer) & (data.folds != inner))
            held = np.flatnonzero(data.folds == inner)
            require(not np.any(data.folds[train] == outer) and inner != outer, "outer leakage in inner fit")
            key = algorithm, outer
            if key not in inner_arrays:
                items = readl(OUT / f"inner_predictions/{retriever}/{algorithm}-outer{outer}.jsonl")
                ids = [r["index"] for r in items]
                require(len(ids) == len(set(ids)) and set(ids) == set(np.flatnonzero(data.folds != outer)), "bad inner OOF indices")
                arr = np.full(len(data.rows), np.nan)
                for row in items:
                    require(row["query_id"] == data.rows[row["index"]]["query_id"] and
                            row["position"] == data.rows[row["index"]]["position"], "inner row identity differs")
                    arr[row["index"]] = row["raw"]
                inner_arrays[key] = arr
            expected = inner_arrays[key][held]
        require(not set(data.groups[train]) & set(data.groups[held]), "model group leakage")
        require(info["train_groups"] == sorted(set(data.groups[train])) and info["held_groups"] == sorted(set(data.groups[held])), "group manifest differs")
        require(info["train_labels_sha256"] == hashlib.sha256(data.y[train].tobytes()).hexdigest(), "training label binding differs")
        require(info["train_rows"] == len(train) and info["held_rows"] == len(held), "fit denominator differs")
        values = np.clip(model.predict(data.x[np.ix_(held, BLOCKS[info["block"]])]), -1, 1)
        err = float(np.max(np.abs(values - expected)))
        max_error = max(max_error, err)
        require(err < 1e-12, "reloaded base predictions differ")
    require(len(inner_arrays) == 10, "inner predictions missing")
    for info in fits:
        if info["kind"] != "calibrator":
            continue
        algorithm, outer, method = info["algorithm"], info["outer"], info["method"]
        training_ids = [q for q in data.qids if data.fold_map[q] != outer]
        train = independently_select(data, training_ids, inner_arrays[algorithm, outer])
        saved_train = readl(ROOT / info["training_cohort_path"])
        require(len(saved_train) == len(train), "inner winner cohort incomplete")
        for row in saved_train:
            require(all(row[key] == value for key, value in train[row["query_id"]].items()), "inner winner selection differs")
            require(abs(row["actual_gain"] - data.y[row["index"]]) < 1e-14, "inner training gain differs")
        eligible = [r for r in train.values() if r["eligible"]]
        held = [independently_chosen[f"FULL_{algorithm}"][q] for q in data.qids if data.fold_map[q] == outer]
        require(info["calibration_training_queries"] == [r["query_id"] for r in eligible], "calibration training cohort differs")
        require(not {data.group_map[r["query_id"]] for r in eligible} & {data.group_map[r["query_id"]] for r in held}, "calibration group leakage")
        train_indices = np.asarray([r["index"] for r in eligible], dtype=int)
        require(info["train_labels_sha256"] == hashlib.sha256(data.y[train_indices].tobytes()).hexdigest(), "calibration label binding differs")
        require(info["train_rows"] == len(eligible) and info["held_rows"] == sum(r["eligible"] for r in held), "calibration counts differ")
        model = joblib.load(ROOT / info["model_path"])
        matrix = independent_matrix(data, held, method)
        predicted = independent_cal_predict(model, matrix, method)
        expected = np.asarray([stored_winners[f"FULL_{algorithm}", r["query_id"]]["calibrated"][method] for r in held])
        predicted = np.asarray([value if row["eligible"] else 0 for row, value in zip(held, predicted)])
        err = float(np.max(np.abs(predicted - expected)))
        max_error = max(max_error, err)
        require(err < 1e-12, "reloaded calibration prediction differs")
        # Independent reconstruction of the calibration fit from inner winners.
        tx = independent_matrix(data, eligible, method)
        ty = data.y[train_indices]
        if len(ty) >= 30:
            if method == "ISOTONIC":
                refit = IsotonicRegression(increasing=True, y_min=-1, y_max=1, out_of_bounds="clip").fit(tx[:, 0], ty)
            else:
                refit = Pipeline([("scale", StandardScaler()), ("regressor", Ridge(alpha=10.0))]).fit(tx, ty)
            refitted = independent_cal_predict(refit, matrix, method)
        else:
            refitted = np.full(len(held), float(ty.mean()) if len(ty) else 0)
        original = independent_cal_predict(model, matrix, method)
        err = float(np.max(np.abs(refitted - original)))
        require(err < 1e-12, "independent calibration refit differs")
        max_error = max(max_error, err)

    # A real-data poisoned-label test: the current outer held labels are changed,
    # then each four-inner-fold routine is actually refitted, not just inspected.
    from .modeling import calibrated_winners, inner_oof
    poison = copy.copy(data)
    poison.y = data.y.copy()
    poison.y[data.folds == 0] += 100000.0
    poison_error = 0.0
    for algorithm in ALGORITHMS:
        changed = inner_oof(poison, 0, algorithm, p["base_models"]["hgb_parameters"])
        train = data.folds != 0
        err = float(np.max(np.abs(changed[train] - inner_arrays[algorithm, 0][train])))
        poison_error = max(poison_error, err)
        require(err < 1e-10, "outer labels affected inner OOF")
        held, _, _ = calibrated_winners(poison, 0, raw[f"FULL_{algorithm}"], changed)
        for row in held:
            expected = stored_winners[f"FULL_{algorithm}", row["query_id"]]
            for method in CALIBRATORS:
                err = abs(row["calibrated"][method] - expected["calibrated"][method])
                poison_error = max(poison_error, err)
                require(err < 1e-10, "outer labels affected calibrated predictions")

    positives = data.positives
    def fraction_rr(q, position):
        return next((Fraction(1, rank) for rank, doc in enumerate(data.actions[q, position]["ranked_doc_ids"], 1)
                     if doc in positives[q]), Fraction(0))
    exact = {key: fraction_rr(*key) for key in data.actions}
    label_error = max(abs(float(exact[r["query_id"], r["position"]] - exact[r["query_id"], None]) - data.y[i])
                      for i, r in enumerate(data.rows))
    require(label_error < 1e-14, "independent exact gain differs")
    specs = read(OUT / "POLICY_SPECS.json")
    decisions = readl(OUT / f"decisions/{retriever}.jsonl")
    require(Counter(r["policy"] for r in decisions) == Counter({name: len(data.qids) for name in specs}), "decision denominator mismatch")
    require(len({(r["policy"], r["query_id"]) for r in decisions}) == len(decisions), "duplicate decision")
    quota_sets = {}
    for policy, spec in specs.items():
        if spec["kind"] not in ("half_eligible", "fixed20"):
            continue
        pool_by_q = independently_chosen[spec["base"]]
        accepted = set()
        for fold in range(5):
            cohort = [r for r in pool_by_q.values() if r["fold"] == fold]
            pool = [r for r in cohort if r["eligible"]] if spec["kind"] == "half_eligible" else [r for r in cohort if not r["structural_zero"]]
            count = int(np.floor(.5 * len(pool))) if spec["kind"] == "half_eligible" else int(np.floor(.2 * len(cohort)))
            def score(row):
                stored_row = stored_winners[spec["base"], row["query_id"]]
                return stored_row["score"] if spec["method"] == "RAW" else stored_row["calibrated"][spec["method"]]
            require(len(pool) >= count, "quota cannot be met")
            ordered = sorted(pool, key=lambda row: (-score(row), -row["score"], row["query_id"]))
            accepted.update(row["query_id"] for row in ordered[:count])
        quota_sets[policy] = accepted
    by_policy = defaultdict(list)
    for row in decisions:
        q, policy = row["query_id"], row["policy"]
        spec = specs[policy]
        if spec["kind"] in ("original", "entropy"):
            position = None if spec["kind"] == "original" else data.rows[data.query_rows[q][0]]["position"]
            score = None
        else:
            winner = independently_chosen[spec["base"]][q]
            score = winner["score"] if spec["method"] == "RAW" else stored_winners[spec["base"], q]["calibrated"][spec["method"]]
            if spec["kind"] in ("half_eligible", "fixed20"):
                accept = q in quota_sets[policy]
            elif spec["kind"] == "forced":
                accept = True
            else:
                accept = winner["eligible"] and score > 0
            position = winner["position"] if accept else None
            require(row["winner_position"] == winner["position"], "winner changed after calibration")
        adopted = data.actions[q, position]["ranked_doc_ids"] != data.actions[q, None]["ranked_doc_ids"]
        position = position if adopted else None
        require(row["selected_position"] == position and row["adopted"] == adopted, "decision replay mismatch")
        require(row["predicted_gain"] == score, "decision score differs")
        require(row["ranking_reference"] == {"retriever": retriever, "query_id": q, "position": position}, "ranking reference differs")
        require(row["fallback_restores_full_ranking"] == (not adopted), "incomplete fallback")
        require(not row["used_gold_at_prediction"] and not row["candidate_reselected_after_calibration"], "invalid prediction claim")
        by_policy[policy].append((q, float(exact[q, position] - exact[q, None]), adopted))
    means = {}
    for policy, rows in by_policy.items():
        values = [r[1] for r in rows]
        means[policy] = {"delta": float(np.mean(values)), "adopted": sum(r[2] for r in rows),
                         "harmful": sum(r[2] and r[1] < -1e-10 for r in rows),
                         "beneficial": sum(r[2] and r[1] > 1e-10 for r in rows)}
    fixed_counts = {means[f"{block}_{a}_fixed20"]["adopted"] for block in BLOCKS for a in ALGORITHMS}
    require(len(fixed_counts) == 1, "base ablations not evaluated at equal adoption")
    for algorithm in ALGORITHMS:
        names = [f"FULL_{algorithm}_half_eligible"] + [f"FULL_{algorithm}_CAL_{m}_half_eligible" for m in CALIBRATORS]
        require(len({means[name]["adopted"] for name in names}) == 1, "calibrators not evaluated at equal count")
        raw_set = quota_sets[f"FULL_{algorithm}_half_eligible"]
        iso_set = quota_sets[f"FULL_{algorithm}_CAL_ISOTONIC_half_eligible"]
        require(raw_set == iso_set, "isotonic monotone calibration unexpectedly changed ranking")
    return {"queries": len(data.qids), "groups": len(set(data.group_map.values())), "candidate_rows": len(data.rows),
            "policies": len(specs), "models_reloaded": dict(reloads), "prediction_max_error": max_error,
            "real_outer_label_poison_max_error": poison_error, "exact_label_max_error": label_error,
            "RR_values_replayed": len(data.actions) + len(decisions), "decision_count": len(decisions),
            "base_fixed20_count": next(iter(fixed_counts)), "policy_replay": means,
            "confirmation_opened": False, "passed": True}


def audit_predictions():
    p = verify()
    seal = read(OUT / "PREDICTIONS_SEALED.json")
    require(seal["protocol_sha256"] == sha(OUT / "PROTOCOL.json"), "protocol seal differs")
    for row in seal["files"]:
        require(sha(ROOT / row["path"]) == row["sha256"], "prediction seal changed")
    for retriever in p["retrievers"]:
        receipt = read(OUT / f"PREDICTIONS_{retriever}.json")
        require(receipt["protocol_sha256"] == sha(OUT / "PROTOCOL.json"), "prediction protocol differs")
        for row in receipt["files"]:
            require(sha(ROOT / row["path"]) == row["sha256"], "prediction/model file changed")
    checks = {}
    for retriever in p["retrievers"]:
        checks[retriever] = audit_retriever(retriever, p)
        print({"independent_audit": retriever, "passed": True,
               "prediction_error": checks[retriever]["prediction_max_error"],
               "poison_error": checks[retriever]["real_outer_label_poison_max_error"]}, flush=True)
    write(OUT / "PREDICTION_AUDIT.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                                       "passed": True, "confirmation_opened": False, "checks": checks}, immutable=True)


def final_acceptance():
    p = verify()
    audit = read(OUT / "PREDICTION_AUDIT.json")
    files = [OUT / name for name in ("PROTOCOL.json", "PREDICTIONS_SEALED.json", "PREDICTION_AUDIT.json",
                                    "SUMMARY.json", "REPORT.md", "POLICY_SPECS.json", "TESTS_PRE_FREEZE.json")]
    max_error = 0.0
    for retriever in p["retrievers"]:
        m = read(OUT / f"metrics/{retriever}.json")
        require(m["queries"] == p["queries"] and m["groups"] == p["groups"], "summary denominator differs")
        for name, ref in audit["checks"][retriever]["policy_replay"].items():
            row = m["policies"][name]
            max_error = max(max_error, abs(row["delta"]["mean"] - ref["delta"]))
            require(row["adopted"] == ref["adopted"] and row["harmful"] == ref["harmful"]
                    and row["beneficial"] == ref["beneficial"], "summary counts differ")
            require(abs(row["positive_gain_per_query"] + row["negative_gain_per_query"] - ref["delta"]) < 1e-14,
                    "summary gain/loss accounting differs")
        per_query = readl(OUT / f"per_query/{retriever}.jsonl")
        require(Counter(r["policy"] for r in per_query) == Counter({k: p["queries"] for k in m["policies"]}), "per-query denominator differs")
        for algorithm in ALGORITHMS:
            methods = m["selected_action_calibration"][f"FULL_{algorithm}"]
            require(len({r["cohort_queries"] for r in methods.values()}) == 1, "calibration cohort changed")
            require(len({r["mean_actual"] for r in methods.values()}) == 1, "calibration outcome cohort differs")
            for result in methods.values():
                require(sum(r["count"] for r in result["bins"]) == result["cohort_queries"], "calibration bins omit observations")
        files += [OUT / f"metrics/{retriever}.json", OUT / f"per_query/{retriever}.jsonl", OUT / f"PREDICTIONS_{retriever}.json"]
    require(max_error < 1e-14, "RR summary differs from exact audit")
    write(OUT / "FINAL_ACCEPTANCE.json", {"created_utc": now(), "passed": True, "summary_max_error": max_error,
                                       "confirmation_opened": False, "risk_certified": False,
                                       "formal_pointers_unchanged": True, "files": [record(path) for path in files]}, immutable=True)
    print({"final_acceptance": True, "summary_max_error": max_error}, flush=True)


if __name__ == "__main__":
    audit_predictions()
