"""Descriptive development summaries, only after every new decision is sealed."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

from .common import (ALGORITHMS, BLOCKS, CALIBRATORS, OUT, load_data, now, read, readl,
                     record, require, sha, verify, write, writel)


class GroupStats:
    def __init__(self, data, p):
        groups = sorted(set(data.group_map.values()))
        mapping = {g: i for i, g in enumerate(groups)}
        self.codes = np.asarray([mapping[data.group_map[q]] for q in data.qids])
        self.g = len(groups)
        rng = np.random.default_rng(p["statistics"]["seed"])
        self.samples = rng.integers(0, self.g, size=(p["statistics"]["bootstrap"], self.g))

    def mean(self, values, mask=None):
        values = np.asarray(values, dtype=float)
        mask = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        require(values.shape == self.codes.shape and mask.shape == values.shape, "metric denominator differs")
        require(np.isfinite(values[mask]).all(), "nonfinite metric")
        counts = np.bincount(self.codes, weights=mask, minlength=self.g)
        sums = np.bincount(self.codes, weights=np.where(mask, values, 0), minlength=self.g)
        if not mask.any():
            return {"mean": None, "ci95": None, "queries": 0, "active_groups": 0, "groups_resampled": self.g}
        den = counts[self.samples].sum(1)
        replicates = sums[self.samples].sum(1)[den > 0] / den[den > 0]
        return {"mean": float(values[mask].mean()), "ci95": np.quantile(replicates, [.025, .975]).tolist(),
                "queries": int(mask.sum()), "active_groups": int((counts > 0).sum()), "groups_resampled": self.g}


def calibration_metrics(scores, actual, mask, stats, bins):
    residual = scores - actual
    details = []
    for left, right in zip(bins[:-1], bins[1:]):
        selected = mask & (scores >= left) & (scores < right)
        details.append({"left_inclusive": left, "right_exclusive": right, "count": int(selected.sum()),
                        "mean_prediction": float(scores[selected].mean()) if selected.any() else None,
                        "mean_actual": float(actual[selected].mean()) if selected.any() else None})
    ece = sum(r["count"] * abs(r["mean_prediction"] - r["mean_actual"]) for r in details if r["count"]) / max(int(mask.sum()), 1)
    rho = None
    if mask.sum() > 2 and np.ptp(scores[mask]) > 0 and np.ptp(actual[mask]) > 0:
        rho = float(spearmanr(scores[mask], actual[mask]).statistic)
    return {"cohort_queries": int(mask.sum()), "mean_prediction": float(scores[mask].mean()) if mask.any() else None,
            "mean_actual": float(actual[mask].mean()) if mask.any() else None,
            "bias": stats.mean(residual, mask), "mse": stats.mean(residual ** 2, mask),
            "mae": stats.mean(np.abs(residual), mask), "spearman_descriptive": rho,
            "fixed_bin_absolute_calibration_error": ece, "bins": details}


def evidence_audit(data, chosen, stats):
    by_q = {r["query_id"]: r for r in chosen}
    actual = np.asarray([data.y[by_q[q]["index"]] for q in data.qids])
    raw_eligible = np.asarray([by_q[q]["eligible"] for q in data.qids])
    original_top1_gold = np.asarray([data.actions[q, None]["ranked_doc_ids"][0] in data.positives[q] for q in data.qids])
    candidate_top1_gold = np.asarray([data.actions[q, by_q[q]["position"]]["ranked_doc_ids"][0] in data.positives[q] for q in data.qids])
    f = np.asarray([data.x[by_q[q]["index"]] for q in data.qids])
    strata = {
        "all_raw_adopters": np.ones(len(data.qids), dtype=bool),
        "original_top1_gold_AUDIT_ONLY": original_top1_gold,
        "original_top1_not_gold_AUDIT_ONLY": ~original_top1_gold,
        "candidate_top1_gold_AUDIT_ONLY": candidate_top1_gold,
        "candidate_top1_not_gold_AUDIT_ONLY": ~candidate_top1_gold,
        "same_retrieved_top1": f[:, 56] == 1,
        "changed_retrieved_top1": f[:, 56] == 0,
        "original_top1_retained_at_candidate_rank1": f[:, 60] == 1,
        "original_top1_displaced": f[:, 60] < 1,
        "jaccard10_ge_half": f[:, 58] >= .5,
        "jaccard10_lt_half": f[:, 58] < .5,
        "original_query_code_coverage_increased": f[:, 65] > f[:, 64],
        "original_query_code_coverage_not_increased": f[:, 65] <= f[:, 64],
        "retrieval_margin12_increased": f[:, 48] > 0,
        "retrieval_margin12_not_increased": f[:, 48] <= 0,
    }
    result = {}
    for name, mask in strata.items():
        selected = mask & raw_eligible
        result[name] = {"queries": int(selected.sum()), "positive": int(((actual > 1e-10) & selected).sum()),
                        "harmful": int(((actual < -1e-10) & selected).sum()),
                        "neutral": int(((np.abs(actual) <= 1e-10) & selected).sum()),
                        "mean_gain_within_stratum": stats.mean(actual, selected),
                        "contribution_per_all_queries": float(np.where(selected, actual, 0).mean())}
    return result


def retriever_summary(retriever, p):
    data = load_data(retriever)
    stats = GroupStats(data, p)
    qindex = {q: i for i, q in enumerate(data.qids)}
    actions = {q: {pos: row["ranked_doc_ids"] for (qid, pos), row in data.actions.items() if qid == q} for q in data.qids}
    # Kept independent from the original outcome file, using cached rankings/qrels.
    def rr(q, position):
        return next((1 / rank for rank, doc in enumerate(actions[q][position], 1) if doc in data.positives[q]), 0.0)
    baseline = np.asarray([rr(q, None) for q in data.qids])
    decisions = readl(OUT / f"decisions/{retriever}.jsonl")
    by_policy = defaultdict(dict)
    for row in decisions:
        require(row["query_id"] not in by_policy[row["policy"]], "duplicate decision")
        by_policy[row["policy"]][row["query_id"]] = row
    metrics, utility, per_query = {}, {}, []
    for policy, rows in by_policy.items():
        require(set(rows) == set(data.qids), "missing policy query")
        selected = np.asarray([rows[q]["adopted"] for q in data.qids])
        delta = np.asarray([rr(q, rows[q]["selected_position"]) for q in data.qids]) - baseline
        require(np.all(delta[~selected] == 0), "fallback changed RR")
        harmful = selected & (delta < -1e-10)
        positive = selected & (delta > 1e-10)
        scores = np.asarray([float(rows[q]["predicted_gain"] or 0) for q in data.qids])
        utility[policy] = delta
        metrics[policy] = {"delta": stats.mean(delta), "mrr": float(baseline.mean() + delta.mean()),
                           "adopted": int(selected.sum()), "coverage": float(selected.mean()),
                           "harmful": int(harmful.sum()), "beneficial": int(positive.sum()),
                           "neutral_adopted": int((selected & ~harmful & ~positive).sum()),
                           "harmful_rate_adopted": stats.mean(harmful.astype(float), selected),
                           "positive_gain_per_query": float(np.maximum(delta, 0).mean()),
                           "negative_gain_per_query": float(np.minimum(delta, 0).mean()),
                           "adopted_mean_actual": float(delta[selected].mean()) if selected.any() else None,
                           "adopted_mean_predicted": float(scores[selected].mean()) if selected.any() else None}
        per_query += [{"query_id": q, "group_id": data.group_map[q], "policy": policy,
                       "baseline_rr": float(baseline[qindex[q]]), "delta_rr": float(delta[qindex[q]]),
                       "selected_position": rows[q]["selected_position"], "adopted": bool(selected[qindex[q]])} for q in data.qids]
    chosen = readl(OUT / f"winners/{retriever}.jsonl")
    by_base = {f"{block}_{a}": [r for r in chosen if r["base"] == f"{block}_{a}"] for block in BLOCKS for a in ALGORITHMS}
    calibration, contrasts, evidence = {}, {}, {}
    for algorithm in ALGORITHMS:
        base = f"FULL_{algorithm}"
        winners_by_q = {r["query_id"]: r for r in by_base[base]}
        actual = np.asarray([data.y[winners_by_q[q]["index"]] for q in data.qids])
        mask = np.asarray([winners_by_q[q]["eligible"] for q in data.qids])
        raw = np.asarray([winners_by_q[q]["score"] for q in data.qids])
        calibration[base] = {}
        for method in ("RAW",) + CALIBRATORS:
            values = raw if method == "RAW" else np.asarray([winners_by_q[q]["calibrated"][method] for q in data.qids])
            result = calibration_metrics(values, actual, mask, stats, p["calibration_bins"])
            result["paired_mse_change_vs_RAW"] = stats.mean((values - actual) ** 2 - (raw - actual) ** 2, mask)
            result["outer_fold_biases"] = {str(f): float((values - actual)[mask & np.asarray([data.fold_map[q] == f for q in data.qids])].mean())
                                           if (mask & np.asarray([data.fold_map[q] == f for q in data.qids])).any() else None for f in range(5)}
            calibration[base][method] = result
            if method != "RAW":
                for kind in ("selective", "half_eligible"):
                    current = f"{base}_CAL_{method}_{kind}"
                    reference = f"{base}_{kind}"
                    contrasts[f"{current}_minus_{reference}"] = stats.mean(utility[current] - utility[reference])
        evidence[base] = evidence_audit(data, by_base[base], stats)
    for algorithm in ALGORITHMS:
        for left, right in (("GS", "GEN"), ("GSA", "GS"), ("GSL", "GS"), ("FULL", "GS"), ("FULL", "GSA"), ("FULL", "GSL")):
            for kind in ("forced", "selective", "fixed20"):
                a, b = f"{left}_{algorithm}_{kind}", f"{right}_{algorithm}_{kind}"
                contrasts[f"{a}_minus_{b}"] = stats.mean(utility[a] - utility[b])
    pred = readl(OUT / f"candidate_predictions/{retriever}.jsonl")
    candidate_quality = {}
    for name in by_base:
        values = np.asarray([0 if r["structural_zero"] else r["raw"][name] for r in pred])
        candidate_quality[name] = {"mse": float(np.mean((values - data.y) ** 2)), "mae": float(np.mean(np.abs(values - data.y))),
                                   "mean_prediction": float(values.mean()), "mean_actual": float(data.y.mean()),
                                   "spearman_descriptive": float(spearmanr(values, data.y).statistic)}
    safe_oracle = np.asarray([max(0.0, float(np.max(data.y[data.query_rows[q]]))) for q in data.qids])
    result = {"retriever": retriever, "queries": len(data.qids), "groups": len(set(data.group_map.values())),
              "NoRewrite_mrr": float(baseline.mean()), "safe_oracle_label_only_delta": float(safe_oracle.mean()),
              "policies": metrics, "selected_action_calibration": calibration, "candidate_prediction_quality": candidate_quality,
              "paired_contrasts": contrasts, "relevance_evidence_audit": evidence,
              "interval_status": "descriptive development; same fold fits held fixed; no multiplicity-adjusted claim"}
    writel(OUT / f"per_query/{retriever}.jsonl", per_query)
    write(OUT / f"metrics/{retriever}.json", result, immutable=True)
    return result


def signed(value):
    return f"{value:+.8f}"


def interval(row):
    return f'[{row["ci95"][0]:+.6f}, {row["ci95"][1]:+.6f}]' if row["ci95"] is not None else "—"


def render(results, p):
    lines = ["# 选择后收益校准与检索证据：缓存开发验证", "",
             "1,490 查询 / 512 连通组；固定 ENTR 三候选；双检索器；外五折、内四折组隔离。",
             "本轮只用已消费开发缓存。确认池未评估，无新生成、检索或 seed/mask；正式指针保持，risk_certified=false，NoRewrite 部署。", "",
             "收益均为全 1,490 查询上的 ΔMRR@100。区间为固定拟合模型后的整组 bootstrap 95% 描述区间；不作多重比较后的确认结论。", "",
             "## 相同赢家的选择后校准", "",
             "校准训练数据是外层训练区内新生成的内折 OOF 正预测赢家；外层标签不进入基模型、标准化或校准拟合。校准不改候选赢家，只能接受或回退。", "",
             "预测误差在每个基模型固定的原始采用队列上计算，避免通过删除难例改善表面校准。", "",
             "| 检索器/基模型 | 校准 | 队列数 | 预测均值 | 真实均值 | 偏差 | MSE | MSE差区间 vs RAW |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for r, result in results.items():
        for base, methods in result["selected_action_calibration"].items():
            for method, m in methods.items():
                lines.append(f'| {r}/{base} | {method} | {m["cohort_queries"]} | {signed(m["mean_prediction"])} | {signed(m["mean_actual"])} | {signed(m["bias"]["mean"])} | {m["mse"]["mean"]:.8f} | {interval(m["paired_mse_change_vs_RAW"])} |')
    lines += ["", "## 零收益阈值下的实际效用", "",
              "| 检索器/基模型 | 方法 | ΔMRR | 95%区间 | 采用数 | 采用内有害率 |", "|---|---|---:|---|---:|---:|"]
    for r, result in results.items():
        for algorithm in ALGORITHMS:
            base = f"FULL_{algorithm}"
            for method in ("RAW",) + CALIBRATORS:
                name = f"{base}_selective" if method == "RAW" else f"{base}_CAL_{method}_selective"
                m = result["policies"][name]
                rate = m["harmful_rate_adopted"]["mean"]
                lines.append(f'| {r}/{algorithm} | {method} | {signed(m["delta"]["mean"])} | {interval(m["delta"])} | {m["adopted"]} | {rate:.2%} |' if rate is not None else f'| {r}/{algorithm} | {method} | {signed(m["delta"]["mean"])} | {interval(m["delta"])} | 0 | — |')
    lines += ["", "## 固定数量：保留原始采用队列的一半", "",
              "每外折按预测分数保留 floor(0.5×原始正预测赢家数)。同一基模型下 RAW 与三种校准的采用数完全一致；这不是全部查询 50% 覆盖。配额诊断可采用校准分数非正动作，用于检查排序，不能与零阈值策略混同。", "",
              "| 检索器/基模型 | 方法 | ΔMRR | 采用数 | 有害率 | 相对RAW差 | 差95%区间 |", "|---|---|---:|---:|---:|---:|---|"]
    for r, result in results.items():
        for algorithm in ALGORITHMS:
            base = f"FULL_{algorithm}"
            raw = result["policies"][f"{base}_half_eligible"]
            for method in ("RAW",) + CALIBRATORS:
                name = f"{base}_half_eligible" if method == "RAW" else f"{base}_CAL_{method}_half_eligible"
                m = result["policies"][name]
                contrast = None if method == "RAW" else result["paired_contrasts"][f"{name}_minus_{base}_half_eligible"]
                delta = m["delta"]["mean"] - raw["delta"]["mean"]
                lines.append(f'| {r}/{algorithm} | {method} | {signed(m["delta"]["mean"])} | {m["adopted"]} | {m["harmful_rate_adopted"]["mean"]:.2%} | {signed(delta)} | {interval(contrast) if contrast else "—"} |')
    lines += ["", "## 同模型族的证据消融", "",
              "GEN=19生成/文本；GS=GEN+36检索分数；GSA=GS+9排名锚点；GSL=GS+7词汇代码覆盖；FULL=71全部特征。锚点RR是已检索文档在另一排名中的位置，不是真实相关文档的RR。", "",
              "| 检索器/模型 | 特征 | 强制选择Δ | 零阈值Δ | 固定20%Δ | 固定20%采用数 |", "|---|---|---:|---:|---:|---:|"]
    for r, result in results.items():
        for a in ALGORITHMS:
            for block in BLOCKS:
                m = [result["policies"][f"{block}_{a}_{kind}"] for kind in ("forced", "selective", "fixed20")]
                lines.append(f'| {r}/{a} | {block} | {signed(m[0]["delta"]["mean"])} | {signed(m[1]["delta"]["mean"])} | {signed(m[2]["delta"]["mean"])} | {m[2]["adopted"]} |')
    lines += ["", "## 独立相关标注诊断", "",
              "以下使用开发标注追溯原始赢家，不能作为线上特征或阈值。每行收益贡献仍除以全体查询数。", "",
              "| 检索器/模型 | 诊断分层 | 数量 | 有益 | 有害 | ΔMRR贡献 |", "|---|---|---:|---:|---:|---:|"]
    wanted = ("original_top1_gold_AUDIT_ONLY", "original_top1_not_gold_AUDIT_ONLY", "same_retrieved_top1",
              "changed_retrieved_top1", "original_query_code_coverage_increased", "original_query_code_coverage_not_increased")
    for r, result in results.items():
        for base, audit in result["relevance_evidence_audit"].items():
            for name in wanted:
                m = audit[name]
                lines.append(f'| {r}/{base} | {name} | {m["queries"]} | {m["positive"]} | {m["harmful"]} | {signed(m["contribution_per_all_queries"])} |')
    lines += ["", "## 解释范围与验收", "",
              "校准均值、收益选择和损害控制是不同问题。校准偏差减小不自动证明收益排序增强；零采用也不构成风险认证。相同数量对照用于检验这一点。",
              "特征消融检验当前模型和缓存中的预测价值；不能将相关特征的差异解释为单一机制的因果贡献，也不据本轮多分支开发结果自动升级正式版本。",
              "完整模型重载、内外组隔离、实际开发标签毒化检查、独立有理数RR复算、固定数量与回退语义见 PREDICTION_AUDIT.json；最终哈希与摘要验收见 FINAL_ACCEPTANCE.json。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def analyze():
    p = verify()
    require(read(OUT / "PREDICTION_AUDIT.json")["passed"], "prediction audit required")
    require((OUT / "PREDICTIONS_SEALED.json").exists(), "predictions must precede summaries")
    results = {r: retriever_summary(r, p) for r in p["retrievers"]}
    render(results, p)
    write(OUT / "SUMMARY.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                               "queries": p["queries"], "groups": p["groups"], "confirmation_opened": False,
                               "risk_certified": False, "formal_pointers_unchanged": True,
                               "results": {r: f"metrics/{r}.json" for r in results}}, immutable=True)
    print({"development_summaries_complete": True, "retrievers": list(results), "confirmation_opened": False}, flush=True)


if __name__ == "__main__":
    analyze()
