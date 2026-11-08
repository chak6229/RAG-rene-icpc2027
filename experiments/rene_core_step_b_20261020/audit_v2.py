"""Independent raw-feature, exact-RR, linear algebra and scalar interval replay.

No production data-loading, decision, feature or metric functions are imported.
The production fit entry is invoked only for a separate held-label poison check.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rene_core_step_b_20261020"
PARENT = ROOT / "results/rene_cosqa_development_20261019"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def words(text):
    return set(re.findall(r"[a-z][a-z0-9]*", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower()))


def score_vector(values):
    a = np.asarray(values, dtype=float)
    sd = np.std(a)
    weight = np.exp((a - max(a)) / max(sd, 1e-8))
    weight /= sum(weight)
    entropy = -sum(float(p) * math.log(max(float(p), 1e-300)) for p in weight)
    return [a[0], a[1], a[2], a[9], a[99], a[0]-a[1], a[0]-a[9],
            np.mean(a[:10]), np.std(a[:10]), sd, (a[0]-a[1])/max(sd, 1e-8), entropy]


def raw_features(query, candidate, original, rewritten, pool, codes):
    """Independently reconstruct all 71 published feature columns from raw data."""
    qe, se, rw = words(query), words(candidate["span"]), words(candidate["rewrite"])
    nq = len(query.split())
    ent = sorted(pool, key=lambda r: (r["full_vocab_entropy"], r["position"]))
    prob = sorted(pool, key=lambda r: (-r["label_mean_log_probability"], r["position"]))
    first = [candidate[k] for k in ("full_vocab_entropy", "label_mean_log_probability", "label_sum_log_probability",
                                    "mean_max_vocab_probability", "label_token_count", "span_token_count")]
    first += [candidate["span_full_vocab_entropy"] or 0, candidate["span_mean_log_probability"] or 0,
              float(not candidate["span"]), candidate["position"] / max(nq, 1), math.log1p(nq),
              math.log1p(len(candidate["span"].split())), len(se-qe)/max(len(se), 1),
              len(candidate["rewrite"].split())/max(nq, 1), float(candidate["is_identity"]),
              [r["position"] for r in ent].index(candidate["position"])/max(len(pool)-1, 1),
              [r["position"] for r in prob].index(candidate["position"])/max(len(pool)-1, 1),
              candidate["full_vocab_entropy"]-ent[0]["full_vocab_entropy"],
              prob[0]["label_mean_log_probability"]-candidate["label_mean_log_probability"]]
    before, after = score_vector(original["scores"]), score_vector(rewritten["scores"])
    old, new = original["ranked_doc_ids"], rewritten["ranked_doc_ids"]
    def inverse(order, document):
        return 1/(order.index(document)+1) if document in order else 0.0
    def cov(a, b):
        return len(a & b)/max(len(a), 1)
    rel = [float(old == new), float(old[0] == new[0])]
    rel += [len(set(old[:k]) & set(new[:k]))/len(set(old[:k]) | set(new[:k])) for k in (3, 10, 100)]
    rel += [inverse(new, old[0]), inverse(old, new[0]), sum(inverse(new, doc) for doc in old[:3])/3,
            sum(.1*.9**(k-1)*len(set(old[:k]) & set(new[:k]))/k for k in range(1, 21)),
            cov(qe, codes[old[0]]), cov(qe, codes[new[0]]), sum(cov(qe, codes[d]) for d in new[:3])/3,
            cov(rw, codes[new[0]]), cov(se, codes[new[0]]), max(cov(se, codes[d]) for d in new[:3]),
            math.log1p(len(codes[new[0]]))]
    return np.asarray(first + before + after + [b-a for a, b in zip(before, after)] + rel)


def rr_exact(order, relevant):
    return next((Fraction(1, i) for i, doc in enumerate(order, 1) if doc in relevant), Fraction(0))


def independent_model(x, y, train, held, columns):
    a = x[train][:, columns]
    mean, var = a.mean(axis=0), a.var(axis=0)
    # StandardScaler's machine precision constant-feature criterion, including tiny constants.
    epsilon = np.finfo(np.float64).eps
    constant = var <= len(a)*epsilon*var + (len(a)*mean*epsilon)**2
    scale = np.sqrt(var)
    scale[constant] = 1
    z = (a-mean)/scale
    zmean = z.mean(axis=0)
    centered = z-zmean
    target = y[train]
    coef = np.linalg.solve(centered.T @ centered + np.eye(len(columns)), centered.T @ (target-target.mean()))
    intercept = target.mean() - zmean @ coef
    pred = ((x[held][:, columns]-mean)/scale) @ coef + intercept
    return mean, scale, var, coef, intercept, pred


def independent_interval(vector, group_list, seed=20261020, reps=10000, family=False):
    # Each scalar contrast is replayed separately, with the identical shared group draws.
    order = sorted(set(group_list))
    lookup = {g: i for i, g in enumerate(order)}
    indices = np.asarray([lookup[g] for g in group_list])
    counts = np.bincount(indices, minlength=len(order))
    totals = np.bincount(indices, weights=vector, minlength=len(order))
    estimate = math.fsum(float(v) for v in vector)/len(vector)
    se = math.sqrt(len(order)/(len(order)-1)*sum((totals-counts*estimate)**2)/len(vector)**2)
    rng = np.random.default_rng(seed)
    distribution = np.empty(reps)
    for start in range(0, reps, 100):
        draw = rng.integers(0, len(order), size=(min(100, reps-start), len(order)))
        distribution[start:start+len(draw)] = np.sum(totals[draw], axis=1)/np.sum(counts[draw], axis=1)
    ret = {"mean": estimate, "standard_error_cluster": se,
           "ci95_cluster_t": [estimate-t.ppf(.975, len(order)-1)*se, estimate+t.ppf(.975, len(order)-1)*se],
           "ci95_group_bootstrap": np.quantile(distribution, [.025, .975]).tolist(),
           "nonzero_queries": sum(abs(float(v)) > 1e-12 for v in vector),
           "groups_with_nonzero_queries": len({g for g, v in zip(group_list, vector) if abs(v) > 1e-12})}
    if family:
        critical = t.ppf(1-.05/16, len(order)-1)
        ret["family95_bonferroni_cluster_t"] = [estimate-critical*se, estimate+critical*se]
        ret["family95_bonferroni_group_bootstrap"] = np.quantile(distribution, [.05/16, 1-.05/16]).tolist()
    return ret


def main():
    protocol = read(OUT / "PROTOCOL.json")
    summary = read(OUT / "RESULTS.json")
    check(protocol["role"] == summary["role"] == "consumed_development_OOF_diagnostic", "Invalid role")
    check(not summary["new_confirmation"] and not summary["risk_certified"], "Unsupported status")
    protected = protocol["bound_files"] + protocol["protected_files"]
    seal = read(OUT / "PREDICTIONS_SEALED.json")
    check(seal["protocol_sha256"] == sha(OUT / "PROTOCOL.json"), "Protocol seal differs")
    for item in protected + seal["files"]:
        check(sha(ROOT / item["path"]) == item["sha256"], f'Hash drift: {item["path"]}')
    qids = read(PARENT / "PROTOCOL.json")["query_ids"]
    groups = read(PARENT / "data/groups.json")
    fold_map = read(PARENT / "FOLDS.json")
    folds = np.asarray([fold_map["query_to_fold"][q] for q in qids])
    check(all(fold_map["group_to_fold"][groups[q]] == folds[i] for i, q in enumerate(qids)), "Group/fold contract differs")
    roles = read(PARENT / "ROLE_MANIFEST.json")
    check(not set(qids) & set(roles["confirmation_pool_query_ids"]), "H queries included")
    check(not set(groups.values()) & set(roles["confirmation_pool_group_ids"]), "H groups included")
    repair = read(OUT / "AUDIT_REPAIR.json")
    for item in repair["bound_files"]:
        check(sha(ROOT / item["path"]) == item["sha256"], "Audit repair source drift")
    rational_spearman = {}
    expected_blocks = {
        "M0": sorted([4, 5] + list(range(8, 15)) + list(range(31, 43)) + list(range(65, 71))),
        "T": [0, 1, 2, 3, 6, 7, 15, 16, 17, 18],
        "R": list(range(19, 31)) + list(range(43, 65)),
    }
    expected_blocks.update({"MT": sorted(expected_blocks["M0"] + expected_blocks["T"]),
                            "MR": sorted(expected_blocks["M0"] + expected_blocks["R"]),
                            "MRT": list(range(71))})
    for h in ("M0", "MT", "MR", "MRT"):
        check(protocol["feature_blocks"][h]["columns"] == expected_blocks[h], "Feature partition differs")
    queries = {r["query_id"]: r["query"] for r in lines(PARENT / "data/queries.jsonl")}
    candidates = {(r["query_id"], r["position"]): r for r in lines(PARENT / "candidates.jsonl")}
    pools = defaultdict(list)
    for (q, _), c in candidates.items():
        pools[q].append(c)
    selections = {r["query_id"]: [v["position"] for v in sorted(r["selected"], key=lambda v: v["selected_rank"])]
                  for r in lines(PARENT / "selections.jsonl") if r["policy"] == "ENTR"}
    codewords = {r["doc_id"]: words(r["code_text"]) for r in lines(ROOT / "results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl")}
    relevant = defaultdict(set)
    for r in lines(PARENT / "data/private/development_qrels.jsonl"):
        check(r["query_id"] in groups, "Qrels outside D")
        if r.get("relevance", 1) > 0:
            relevant[r["query_id"]].add(r["doc_id"])
    evaluated = {(r["retriever"], r["query_id"]): r for r in lines(OUT / "per_query.jsonl")}
    check(len(evaluated) == 2980 and len(qids) == 1490 and len(set(groups.values())) == 512, "Denominator differs")
    errors = defaultdict(float)
    counts = defaultdict(int)

    def near(a, b, category, tolerance=2e-10):
        error = float(np.max(np.abs(np.asarray(a, dtype=float)-np.asarray(b, dtype=float))))
        errors[category] = max(errors[category], error)
        counts[category] += 1
        check(error < tolerance, f"{category} mismatch {error}")

    with threadpool_limits(limits=1):
        for retriever in ("CodeBERT", "BM25"):
            ranks = {(r["query_id"], r["position"]): r for r in lines(PARENT / f"rankings/{retriever}.jsonl")}
            cache = {(r["query_id"], r["position"]): r for r in lines(PARENT / f"features/{retriever}.jsonl")}
            predrows = lines(OUT / f"predictions/{retriever}.jsonl")
            check([r["query_id"] for r in predrows] == qids, "Prediction IDs/order differ")
            x, base, y, floating_y, eligibility = [], [], [], [], []
            for i, q in enumerate(qids):
                entr = sorted(pools[q], key=lambda r: (r["full_vocab_entropy"], r["position"]))[:3]
                check([r["position"] for r in entr] == selections[q], "ENTR action selection differs")
                chosen = entr[0]
                p = chosen["position"]
                before, after = ranks[q, None], ranks[q, p]
                check(before["text"] == queries[q] and after["text"] == chosen["rewrite"], "Ranking text differs")
                raw = raw_features(queries[q], chosen, before, after, entr, codewords)
                near(raw, cache[q, p]["values"], "independent_feature_vectors")
                x.append(raw)
                br, cr = rr_exact(before["ranked_doc_ids"], relevant[q]), rr_exact(after["ranked_doc_ids"], relevant[q])
                base.append(float(br))
                # Correct policy sign comes from exact rationals, not rounded MRR.
                y.append(float(cr-br))
                floating_y.append(float(cr)-float(br))
                row = evaluated[retriever, q]
                near([float(br), float(cr), float(cr-br)], [row["original_rr"], row["candidate_rr"], row["delta_rr"]], "fraction_rr")
                identity = chosen["rewrite"] == queries[q]
                same = before["ranked_doc_ids"] == after["ranked_doc_ids"]
                eligibility.append(not identity and not same)
                check(row["position"] == predrows[i]["position"] == p, "Fixed action changed")
                check(predrows[i]["eligible"] == eligibility[-1] and predrows[i]["identity_text"] == identity and predrows[i]["same_ranking"] == same, "Eligibility mismatch")
                check(predrows[i]["held_fold"] == folds[i] and predrows[i]["group"] == groups[q], "Fold/group identity differs")
                counts["fixed_actions"] += 1
            x, y, base, eligibility = np.asarray(x), np.asarray(y), np.asarray(base), np.asarray(eligibility)
            # Use the original cache for numerically exact production-entry poison checking.
            cached_x = np.asarray([cache[q, selections[q][0]]["values"] for q in qids])
            production_y = np.asarray(floating_y)
            near(production_y, [evaluated[retriever, q]["delta_rr"] for q in qids], "independent_floating_targets", 1e-14)
            predictions, adopted = {}, {"positive": {}, "fixed20": {}}
            for h in ("M0", "MT", "MR", "MRT"):
                predictions[h] = np.asarray([r["predictions"][h] for r in predrows])
                for fold in range(5):
                    train, held = folds != fold, folds == fold
                    train_groups = {groups[q] for i, q in enumerate(qids) if train[i]}
                    held_groups = {groups[q] for i, q in enumerate(qids) if held[i]}
                    check(not train_groups & held_groups, "Fold group leakage")
                    model = read(OUT / f"models/{retriever}/{h}/fold{fold}.json")
                    check(set(model["train_query_ids"]) == set(np.asarray(qids)[train]) and set(model["held_query_ids"]) == set(np.asarray(qids)[held]), "Model training/held population differs")
                    check(set(model["train_group_ids"]) == train_groups and set(model["held_group_ids"]) == held_groups, "Model group IDs differ")
                    mean, scale, var, coef, intercept, predicted = independent_model(x, y, train, held, expected_blocks[h])
                    near(mean, model["scaler_mean"], "training_scaler_mean")
                    near(scale, model["scaler_scale"], "training_scaler_scale")
                    near(var, model["scaler_variance"], "training_scaler_variance")
                    near(coef, model["coefficient"], "independent_ridge_coefficients", 2e-8)
                    near(intercept, model["intercept"], "independent_ridge_intercept", 2e-8)
                    near(predicted, predictions[h][held], "independent_ridge_predictions", 2e-8)
                    savedpred = ((cached_x[held][:, model["columns"]]-model["scaler_mean"])/model["scaler_scale"]) @ np.asarray(model["coefficient"]) + model["intercept"]
                    near(savedpred, predictions[h][held], "saved_model_reloaded")
                    # Deliberately call the production fit entry on corrupted HELD labels.
                    from .modeling import fit_fold
                    poisoned = production_y.copy()
                    poisoned[held] = np.linspace(-1e6, 1e6, held.sum())
                    poisoned_model, poisoned_pred = fit_fold(cached_x, poisoned, folds, h, fold)
                    near(poisoned_pred, predictions[h][held], "held_label_poison_predictions", 1e-12)
                    near(poisoned_model["coefficient"], model["coefficient"], "held_label_poison_coefficients", 1e-12)
                    counts["independently_refitted_models"] += 1
                adopted["positive"][h] = eligibility & (predictions[h] > 0)
                quota = np.zeros(len(qids), dtype=bool)
                for fold in range(5):
                    pool = [i for i in range(len(qids)) if folds[i] == fold and eligibility[i]]
                    order = sorted(pool, key=lambda i: (-predictions[h][i], qids[i]))
                    quota[order[:len(pool)//5]] = True
                adopted["fixed20"][h] = quota
                for policy in ("positive", "fixed20"):
                    check(np.array_equal(adopted[policy][h], np.asarray([r["adopted"][policy][h] for r in predrows])), "Independent adoption differs")
                    for i, q in enumerate(qids):
                        near(y[i]*adopted[policy][h][i], evaluated[retriever, q]["policy_gain"][policy][h], "policy_query_gain")
                    u = summary["views"][retriever]["policies"][policy][h]
                    gain = y*adopted[policy][h]
                    harm, positive = (gain < 0), (gain > 0)
                    near(np.mean(gain), u["delta_mrr"], "utility_mean")
                    near(np.mean(base+gain), u["mrr"], "absolute_mrr")
                    near(np.mean(adopted[policy][h]), u["coverage"], "coverage")
                    near(-sum(gain[harm])/len(y), u["harm_mass_per_query"], "harm_mass")
                    near(sum(gain[positive])/len(y), u["benefit_mass_per_query"], "benefit_mass")
                    near(sum(gain[positive])/sum(np.maximum(y, 0)), u["retained_fixed_action_positive_mass"], "retained_positive_mass")
                    check(int(sum(harm)) == u["harmful"] and int(sum(positive)) == u["beneficial"] and int(sum(adopted[policy][h])) == u["adopted"], "Adoption/harm count differs")
                    for foldrow in u["per_fold"]:
                        subset = folds == foldrow["fold"]
                        near(gain[subset].mean(), foldrow["delta_mrr"], "fold_utility")
                        check(int(sum(adopted[policy][h] & subset)) == foldrow["adopted"], "Fold quota differs")
                quality = summary["views"][retriever]["prediction_quality"][h]
                near(np.mean((predictions[h]-y)**2), quality["mse"], "prediction_mse")
                near(np.mean(np.abs(predictions[h]-y)), quality["mae"], "prediction_mae")
                near(spearmanr(predictions[h], production_y).statistic, quality["prediction_spearman"], "prediction_spearman")
                rational_spearman[f"{retriever}:{h}"] = {"reported_float_target": quality["prediction_spearman"], "exact_rational_target": float(spearmanr(predictions[h], y).statistic), "reason": "exact Fraction differences restore mathematical ties; reported target uses IEEE754 float RR subtraction"}
                indices = sorted(range(len(y)), key=lambda i: (predictions[h][i], qids[i]))
                for b, subset in enumerate(np.array_split(indices, 5)):
                    near(predictions[h][subset].mean(), quality["calibration_quintiles"][b]["mean_prediction"], "calibration_mean_prediction")
                    near(y[subset].mean(), quality["calibration_quintiles"][b]["observed_mean_delta"], "calibration_observed")
            check(len({int(sum(adopted["fixed20"][h])) for h in predictions}) == 1, "Heads have unequal fixed-budget counts")
            for name, estimate in summary["views"][retriever]["estimates"].items():
                if name == "ForcedENTRTop1-NoRewrite":
                    vector = y
                    primary = False
                else:
                    policy, comparison = name.split(":")
                    left, right = comparison.split("-")
                    vector = y*adopted[policy][left]
                    if right != "NoRewrite":
                        vector = vector-y*adopted[policy][right]
                    primary = comparison in ("MRT-MT", "MRT-MR")
                independent = independent_interval(vector, [groups[q] for q in qids], family=primary)
                for key, value in independent.items():
                    near(value, estimate[key], "independent_interval_fields")
                counts["scalar_contrasts"] += 1
            print({"independent_audit": retriever, "status": "PASS"}, flush=True)
    for item in protected:
        check(sha(ROOT / item["path"]) == item["sha256"], "Protected/input/source changed during audit")
    result = {"passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "checks": dict(counts), "max_errors": dict(errors),
              "audit_repair_sha256": sha(OUT / "AUDIT_REPAIR.json"),
              "exact_rational_spearman_sensitivity": rational_spearman,
              "input_source_protected_hashes_verified": len(protected),
              "sealed_prediction_model_hashes_verified": len(seal["files"]),
              "independent_raw_feature_and_exact_RR_replay": True,
              "production_fit_held_labels_poisoned": True,
              "H_effects_read": False, "official_files_unchanged": True,
              "intervals_conditional_on_fixed_OOF_not_algorithm_generalization": True}
    with (OUT / "AUDIT.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    acceptance = {"passed": True, "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                  "results_sha256": sha(OUT / "RESULTS.json"), "audit_sha256": sha(OUT / "AUDIT.json"),
                  "tests": read(OUT / "TEST_RESULTS.json"), "scope": "step B development diagnostic only",
                  "risk_certified": False, "official_configuration_changed": False}
    check(acceptance["tests"]["passed"], "Contract tests not passed")
    with (OUT / "FINAL_ACCEPTANCE.json").open("x", encoding="utf-8") as handle:
        json.dump(acceptance, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print({"step_B_audit_passed": True, "models": counts["independently_refitted_models"],
           "fixed_actions": counts["fixed_actions"], "scalar_contrasts": counts["scalar_contrasts"]}, flush=True)


if __name__ == "__main__":
    main()
