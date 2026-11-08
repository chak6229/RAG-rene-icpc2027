"""Fixed-hyperparameter direct gain models with whole-group OOF predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.rene_cosqa_pilot_20261019.evaluation import rr
from .common import (OUT, now, read, readl, require, sha, verify_decisions, verify_frozen, write, writel)
from .features import GEN_NAMES


def fit_model(name, spec, hgb_parameters, x, y, weights):
    weights = np.asarray(weights, dtype=float) / float(np.mean(weights))
    columns = len(GEN_NAMES) if spec["features"] == "generation" else x.shape[1]
    if spec["algorithm"] == "StandardScaler+Ridge":
        model = Pipeline([("scale", StandardScaler()), ("regressor", Ridge(alpha=spec["alpha"]))])
        model.fit(x[:, :columns], y, scale__sample_weight=weights, regressor__sample_weight=weights)
    else:
        model = HistGradientBoostingRegressor(**hgb_parameters)
        model.fit(x[:, :columns], y, sample_weight=weights)
    return model, columns


def train_partition(name, spec, hgb_parameters, x, y, weights, groups, train_indices, test_indices):
    train_groups = {groups[i] for i in train_indices}
    test_groups = {groups[i] for i in test_indices}
    require(train_groups and test_groups and not train_groups & test_groups, "train/test connected-group leakage")
    model, columns = fit_model(name, spec, hgb_parameters, x[train_indices], y[train_indices], weights[train_indices])
    return model, model.predict(x[test_indices, :columns])


def choose(rows, model, *, selective):
    ordered = sorted(rows, key=lambda r: (-r["predictions"][model]["effective"], r["entr_rank"], r["position"]))
    best = ordered[0]
    if selective and best["predictions"][model]["effective"] <= 0:
        return None, 0.0
    return best["position"], best["predictions"][model]["effective"]


def decision(query_id, policy, position, score, actions, fold):
    original = actions[query_id, None]
    row = actions[query_id, position]
    adopted = row["ranked_doc_ids"] != original["ranked_doc_ids"]
    return {"query_id": query_id, "policy": policy, "selected_position": position, "predicted_gain": score,
            "fold": fold, "adopted": adopted, "action": "adopt" if adopted else "fallback",
            "ranked_doc_ids": row["ranked_doc_ids"] if adopted else original["ranked_doc_ids"],
            "fallback_restores_full_ranking": not adopted, "used_gold_at_prediction": False}


def train(retriever):
    protocol = verify_frozen(private=True)
    verify_decisions()
    require(read(OUT / "PREOUTCOME_AUDIT.json")["passed"], "independent label-free audit required")
    require(not (OUT / f"OOF_SEALED_{retriever}.json").exists(), "OOF already completed; reuse saved predictions")
    access = OUT / f"GAIN_LABEL_ACCESS_{retriever}.json"
    require(not access.exists(), "training started before; inspect partial artifacts before retry")
    write(access, {"created_utc": now(), "purpose": "fit on D only; no H effects", "confirmation_opened": False,
                   "protocol_sha256": sha(OUT / "PROTOCOL.json"), "preoutcome_audit_passed": True}, immutable=True)
    feature_rows = readl(OUT / f"features/{retriever}.jsonl")
    groups_map = read(OUT / "data/groups.json")
    fold_map = read(OUT / "FOLDS.json")["query_to_fold"]
    actions = {(r["query_id"], r["position"]): r for r in readl(OUT / f"rankings/{retriever}.jsonl")}
    qrels = readl(OUT / "data/private/development_qrels.jsonl")
    positives = defaultdict(set)
    for row in qrels:
        if row.get("relevance", 1) > 0:
            positives[row["query_id"]].add(row["doc_id"])
    query_counts = Counter(row["query_id"] for row in feature_rows)
    x = np.asarray([row["values"] for row in feature_rows], dtype=np.float64)
    y = np.asarray([rr(actions[row["query_id"], row["position"]]["ranked_doc_ids"], positives[row["query_id"]], 100)
                    - rr(actions[row["query_id"], None]["ranked_doc_ids"], positives[row["query_id"]], 100)
                    for row in feature_rows])
    groups = [groups_map[row["query_id"]] for row in feature_rows]
    folds = np.asarray([fold_map[row["query_id"]] for row in feature_rows])
    weights = np.asarray([1.0 / query_counts[row["query_id"]] for row in feature_rows])
    weights /= weights.mean()
    models = protocol["direct_gain"]["models"]
    predictions = {name: np.full(len(x), np.nan) for name in models}
    fits = []
    models_dir = OUT / f"models/{retriever}"
    models_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        for fold in range(protocol["direct_gain"]["folds"]):
            train_ids, test_ids = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            train_groups, test_groups = sorted({groups[i] for i in train_ids}), sorted({groups[i] for i in test_ids})
            for name, spec in models.items():
                model, pred = train_partition(name, spec, protocol["direct_gain"]["hgb_parameters"], x, y, weights,
                                              groups, train_ids, test_ids)
                path = models_dir / f"{name}-fold{fold}.joblib"
                joblib.dump(model, path)
                predictions[name][test_ids] = np.clip(pred, -1, 1)
                fits.append({"model": name, "fold": fold, "train_queries": len({feature_rows[i]["query_id"] for i in train_ids}),
                             "test_queries": len({feature_rows[i]["query_id"] for i in test_ids}),
                             "train_groups": train_groups, "test_groups": test_groups,
                             "train_rows": len(train_ids), "test_rows": len(test_ids),
                             "model_path": path.relative_to(OUT).as_posix(), "model_sha256": sha(path),
                             "feature_columns": len(GEN_NAMES) if spec["features"] == "generation" else x.shape[1]})
            print({"gain_fit": retriever, "fold_complete": fold, "train_queries": fits[-1]["train_queries"],
                   "held_queries": fits[-1]["test_queries"]}, flush=True)
        for name, spec in models.items():
            model, columns = fit_model(name, spec, protocol["direct_gain"]["hgb_parameters"], x, y, weights)
            path = models_dir / f"{name}-fullD.joblib"
            joblib.dump(model, path)
            fits.append({"model": name, "fold": "fullD-research-only", "train_groups": sorted(set(groups)),
                         "train_queries": len(query_counts), "train_rows": len(x), "feature_columns": columns,
                         "test_queries": 0, "used_for_development_metrics": False,
                         "model_path": path.relative_to(OUT).as_posix(), "model_sha256": sha(path)})
    require(all(np.isfinite(p).all() for p in predictions.values()), "OOF predictions incomplete/nonfinite")
    out_rows = []
    for i, row in enumerate(feature_rows):
        out_rows.append({"query_id": row["query_id"], "position": row["position"], "entr_rank": row["entr_rank"],
                         "fold": int(folds[i]), "group_id": groups[i], "structural_zero": row["structural_zero"],
                         "predictions": {name: {"raw_clipped": float(pred[i]), "effective": 0.0 if row["structural_zero"] else float(pred[i])}
                                         for name, pred in predictions.items()}})
    writel(OUT / f"oof_predictions/{retriever}.jsonl", out_rows)
    write(OUT / f"TRAINING_LEDGER_{retriever}.json", {"created_utc": now(), "fits": fits, "hyperparameters_fixed": True,
                                                   "confirmation_used": False, "label": "delta_RR@100", "folds": 5}, immutable=True)
    pooled = defaultdict(list)
    for row in out_rows:
        pooled[row["query_id"]].append(row)
    candidates = {(r["query_id"], r["position"]): r for r in readl(OUT / "candidates.jsonl")}
    decisions = []
    for qid in protocol["query_ids"]:
        rows = sorted(pooled[qid], key=lambda r: r["entr_rank"])
        fold = fold_map[qid]
        native_position = rows[0]["position"]
        probability_position = min(rows, key=lambda r: (-candidates[qid, r["position"]]["label_mean_log_probability"], r["position"]))["position"]
        decisions += [decision(qid, "NoRewrite", None, 0.0, actions, fold),
                      decision(qid, "ENTR_top1", native_position, None, actions, fold),
                      decision(qid, "PROB_in_ENTR3", probability_position, None, actions, fold)]
        for name in models:
            for selective in (False, True):
                position, score = choose(rows, name, selective=selective)
                decisions.append(decision(qid, name + ("_selective" if selective else "_forced"), position, score, actions, fold))
    writel(OUT / f"decisions/{retriever}.jsonl", decisions)
    paths = [OUT / f"oof_predictions/{retriever}.jsonl", OUT / f"decisions/{retriever}.jsonl",
             OUT / f"TRAINING_LEDGER_{retriever}.json"] + list(models_dir.glob("*.joblib"))
    write(OUT / f"OOF_SEALED_{retriever}.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                                              "out_of_group_predictions": True, "decisions_before_rq2_summary": True,
                                              "confirmation_opened": False,
                                              "files": [{"path": path.relative_to(OUT).as_posix(), "sha256": sha(path)} for path in paths]}, immutable=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("retriever", choices=("CodeBERT", "BM25"))
    train(parser.parse_args().retriever)
