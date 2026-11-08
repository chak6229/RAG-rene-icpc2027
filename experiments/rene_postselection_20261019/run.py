from __future__ import annotations

import hashlib
from collections import Counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .common import (ALGORITHMS, BLOCKS, CALIBRATORS, OUT, PARENT, ROOT, load_data, now,
                     parent_predictions, read, readl, record, require, sha, verify, write, writel)
from .modeling import (calibrated_winners, fit_base, inner_oof, make_decision, quota, winners)


def policy_specs():
    specs = {"NoRewrite": {"kind": "original"}, "ENTR_top1": {"kind": "entropy"}}
    for block in BLOCKS:
        for algorithm in ALGORITHMS:
            name = f"{block}_{algorithm}"
            for kind in ("forced", "selective", "fixed20"):
                specs[f"{name}_{kind}"] = {"kind": kind, "base": name, "method": "RAW"}
    for algorithm in ALGORITHMS:
        base = f"FULL_{algorithm}"
        specs[f"{base}_half_eligible"] = {"kind": "half_eligible", "base": base, "method": "RAW"}
        for method in CALIBRATORS:
            for kind in ("selective", "half_eligible"):
                specs[f"{base}_CAL_{method}_{kind}"] = {"kind": kind, "base": base, "method": method}
    return specs


def fit_retriever(retriever, p):
    require(not (OUT / f"PREDICTIONS_{retriever}.json").exists(), "predictions already sealed")
    require(not (OUT / f"LABEL_ACCESS_{retriever}.json").exists(), "partial fit exists; inspect before resuming")
    write(OUT / f"LABEL_ACCESS_{retriever}.json", {"created_utc": now(), "confirmation_used": False,
                                               "purpose": "nested cached-feature development refits only"}, immutable=True)
    data = load_data(retriever)
    raw, ledger, all_winners = {}, [], []

    def save_model(model, destination, info, train, held):
        require(not destination.exists(), "model must not be overwritten")
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, destination)
        ledger.append({**info, "model_path": destination.relative_to(ROOT).as_posix(), "model_sha256": sha(destination),
                       "train_rows": len(train), "held_rows": len(held),
                       "train_groups": sorted(set(data.groups[train])), "held_groups": sorted(set(data.groups[held])),
                       "train_labels_sha256": hashlib.sha256(data.y[train].tobytes()).hexdigest()})

    for block in BLOCKS:
        for algorithm in ALGORITHMS:
            name = f"{block}_{algorithm}"
            reused = block == "FULL" or (block == "GEN" and algorithm == "HGB")
            raw[name] = parent_predictions(data, algorithm, block) if reused else np.full(len(data.rows), np.nan)
            for fold in range(5):
                train = np.flatnonzero(data.folds != fold)
                held = np.flatnonzero(data.folds == fold)
                info = {"kind": "outer", "algorithm": algorithm, "block": block, "outer": fold, "source": "parent" if reused else "new"}
                if reused:
                    old_name = "GEN_HGB" if block == "GEN" else f"GAIN_{algorithm}"
                    path = PARENT / f"models/{retriever}/{old_name}-fold{fold}.joblib"
                    ledger.append({**info, "model_path": path.relative_to(ROOT).as_posix(), "model_sha256": sha(path),
                                   "train_rows": len(train), "held_rows": len(held),
                                   "train_groups": sorted(set(data.groups[train])), "held_groups": sorted(set(data.groups[held])),
                                   "train_labels_sha256": hashlib.sha256(data.y[train].tobytes()).hexdigest()})
                else:
                    model, prediction = fit_base(data, train, held, algorithm, block, p["base_models"]["hgb_parameters"])
                    raw[name][held] = prediction
                    save_model(model, OUT / f"models/{retriever}/outer-{name}-{fold}.joblib", info, train, held)
            require(np.isfinite(raw[name]).all(), "base OOF incomplete")
            current = winners(data, range(len(data.rows)), raw[name])
            all_winners += [{**r, "base": name} for r in current]
            print({"retriever": retriever, "base_complete": name, "reused": reused}, flush=True)

    full_selected = {}
    for algorithm in ALGORITHMS:
        base = f"FULL_{algorithm}"
        full_selected[base] = []
        for outer in range(5):
            def emit(inner, train, held, model):
                save_model(model, OUT / f"models/{retriever}/inner-{algorithm}-outer{outer}-held{inner}.joblib",
                           {"kind": "inner", "algorithm": algorithm, "block": "FULL", "outer": outer, "inner": inner, "source": "new"},
                           train, held)
            inner_prediction = inner_oof(data, outer, algorithm, p["base_models"]["hgb_parameters"], emit=emit)
            inner_path = OUT / f"inner_predictions/{retriever}/{algorithm}-outer{outer}.jsonl"
            writel(inner_path, [{"index": int(i), "query_id": data.rows[i]["query_id"], "position": data.rows[i]["position"],
                                "raw": float(inner_prediction[i])} for i in np.flatnonzero(data.folds != outer)])
            selected, training, calibrators = calibrated_winners(data, outer, raw[base], inner_prediction)
            full_selected[base] += selected
            cohort_path = OUT / f"inner_winners/{retriever}/{algorithm}-outer{outer}.jsonl"
            writel(cohort_path, [{**r, "actual_gain": float(data.y[r["index"]])} for r in training])
            for method, (model, eligible) in calibrators.items():
                train = np.asarray([r["index"] for r in eligible], dtype=int)
                held = np.asarray([r["index"] for r in selected if r["eligible"]], dtype=int)
                save_model(model, OUT / f"models/{retriever}/cal-{algorithm}-outer{outer}-{method}.joblib",
                           {"kind": "calibrator", "algorithm": algorithm, "block": "FULL", "outer": outer, "method": method,
                            "source": "new", "training_cohort_path": cohort_path.relative_to(ROOT).as_posix(),
                            "inner_prediction_path": inner_path.relative_to(ROOT).as_posix(),
                            "calibration_training_queries": [r["query_id"] for r in eligible]}, train, held)
            print({"retriever": retriever, "nested_complete": base, "outer": outer,
                   "inner_positive_winners": len(next(iter(calibrators.values()))[1]),
                   "held_queries": len(selected)}, flush=True)
    for row in all_winners:
        if row["base"] in full_selected:
            found = next(r for r in full_selected[row["base"]] if r["query_id"] == row["query_id"])
            require(all(row[key] == found[key] for key in ("position", "index", "score", "eligible")), "calibration changed raw winner")
            row["calibrated"] = found["calibrated"]

    by_base = {name: [r for r in all_winners if r["base"] == name] for name in raw}
    decisions = []
    for policy, spec in policy_specs().items():
        kind = spec["kind"]
        if kind in ("original", "entropy"):
            for q in data.qids:
                row = data.rows[data.query_rows[q][0]]
                decisions.append(make_decision(data, {"query_id": q, "position": row["position"]}, policy, None, kind == "entropy"))
            continue
        method = spec["method"]
        current = [{**r, "policy_score": r["score"] if method == "RAW" else r["calibrated"][method]} for r in by_base[spec["base"]]]
        if kind in ("fixed20", "half_eligible"):
            accepted = quota(current, "policy_score", .2 if kind == "fixed20" else .5, eligible_only=kind == "half_eligible")
        else:
            accepted = {r["query_id"] for r in current if kind == "forced" or (r["eligible"] and r["policy_score"] > 0)}
        decisions += [make_decision(data, r, policy, r["policy_score"], r["query_id"] in accepted) for r in current]
    counts = Counter(r["policy"] for r in decisions)
    require(counts == Counter({name: len(data.qids) for name in policy_specs()}), "policy denominator differs")
    writel(OUT / f"candidate_predictions/{retriever}.jsonl", [
        {"index": i, "query_id": r["query_id"], "position": r["position"], "fold": int(data.folds[i]),
         "structural_zero": r["structural_zero"], "raw": {name: float(values[i]) for name, values in raw.items()}}
        for i, r in enumerate(data.rows)])
    writel(OUT / f"winners/{retriever}.jsonl", all_winners)
    writel(OUT / f"decisions/{retriever}.jsonl", decisions)
    write(OUT / f"TRAINING_LEDGER_{retriever}.json", {"created_utc": now(), "fits": ledger,
                                                   "confirmation_used": False, "outer_folds": 5, "inner_folds": 4}, immutable=True)
    files = [OUT / f"candidate_predictions/{retriever}.jsonl", OUT / f"winners/{retriever}.jsonl",
             OUT / f"decisions/{retriever}.jsonl", OUT / f"TRAINING_LEDGER_{retriever}.json"]
    for subdir, pattern in (("models", "*.joblib"), ("inner_predictions", "*.jsonl"), ("inner_winners", "*.jsonl")):
        files += list((OUT / f"{subdir}/{retriever}").glob(pattern))
    write(OUT / f"PREDICTIONS_{retriever}.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                                               "files": [record(path) for path in sorted(files)], "confirmation_used": False}, immutable=True)


def main():
    p = verify()
    require(not (OUT / "FINAL_ACCEPTANCE.json").exists(), "accepted run is immutable")
    write(OUT / "RUN_STATUS.json", {"stage": "fitting", "updated_utc": now(), "confirmation_opened": False})
    write(OUT / "POLICY_SPECS.json", policy_specs(), immutable=True)
    with threadpool_limits(limits=4):
        for retriever in p["retrievers"]:
            if not (OUT / f"PREDICTIONS_{retriever}.json").exists():
                fit_retriever(retriever, p)
        verify()
        write(OUT / "PREDICTIONS_SEALED.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                                             "before_new_aggregate_outcomes": True, "prior_D_effects_known": True,
                                             "files": [record(OUT / f"PREDICTIONS_{r}.json") for r in p["retrievers"]]
                                             + [record(OUT / "POLICY_SPECS.json")]}, immutable=True)
        write(OUT / "RUN_STATUS.json", {"stage": "independent_prediction_audit", "updated_utc": now(), "confirmation_opened": False})
        from .audit import audit_predictions, final_acceptance
        audit_predictions()
        write(OUT / "RUN_STATUS.json", {"stage": "development_summary", "updated_utc": now(), "confirmation_opened": False})
        from .analyze import analyze
        analyze()
        final_acceptance()
    write(OUT / "RUN_STATUS.json", {"stage": "complete", "updated_utc": now(), "confirmation_opened": False,
                                  "acceptance_passed": True})


if __name__ == "__main__":
    main()
