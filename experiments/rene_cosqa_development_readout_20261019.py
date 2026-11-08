"""Post-result arithmetic readout; does not fit, tune, or change any policy."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.rene_cosqa_development_20261019.common import (
    OUT, ROOT, now, read, readl, require, sha, verify_frozen, write,
)


def main():
    protocol = verify_frozen()
    acceptance = read(OUT / "FINAL_ACCEPTANCE.json")
    require(acceptance["passed"] and read(OUT / "RUN_STATUS.json")["status"] == "complete", "run not complete")
    for row in acceptance["files"]:
        require(sha(OUT / row["path"]) == row["sha256"], "accepted result changed")
    result = {"created_utc": now(), "role": "post-result descriptive arithmetic", "new_model_fits": 0,
              "new_generation": 0, "policies_changed": False, "confirmation_opened": False, "retrievers": {}}
    sources = []
    chosen_policies = ("ENTR_top1", "GEN_HGB_selective", "GAIN_RIDGE_selective", "GAIN_HGB_forced", "GAIN_HGB_selective")
    for retriever in protocol["retrievers"]:
        query_file = OUT / f"metrics/{retriever}_RQ2_per_query.jsonl"
        decision_file = OUT / f"decisions/{retriever}.jsonl"
        summary_file = OUT / f"metrics/{retriever}_RQ2.json"
        rows, decisions = readl(query_file), readl(decision_file)
        summary = read(summary_file)["subsets"]["all_development"]
        sources += [query_file, decision_file, summary_file]
        n = len(rows)
        require(n == 1490 and len({r["query_id"] for r in rows}) == n, "wrong readout denominator")
        oracle = sum(r["oracle_with_NoRewrite_rr"] - r["NoRewrite"] for r in rows)
        metrics = {"queries": n, "candidate_opportunity_queries": sum(r["oracle_with_NoRewrite_rr"] > r["NoRewrite"] + 1e-10 for r in rows),
                   "all_three_candidates_worse_than_original": sum(r["oracle_ENTR3_rr"] < r["NoRewrite"] - 1e-10 for r in rows),
                   "oracle_with_NoRewrite_gain_sum": oracle, "oracle_with_NoRewrite_delta": oracle / n, "policies": {}}
        for policy in chosen_policies:
            gains = [r["policies"][policy]["delta"] for r in rows]
            positive = sum(max(g, 0) for g in gains)
            damage = sum(max(-g, 0) for g in gains)
            point = (positive - damage) / n
            require(abs(point - summary["policies"][policy]["delta"]["mean"]) < 1e-12, "readout arithmetic differs")
            selected = [r for r in decisions if r["policy"] == policy]
            predicted = [r["predicted_gain"] for r in selected]
            predicted_mean = sum(predicted) / n if all(v is not None for v in predicted) else None
            metrics["policies"][policy] = {
                "delta": point, "positive_gain_per_query": positive / n, "damage_per_query": damage / n,
                "available_positive_gain_captured": positive / oracle if oracle else None,
                "available_positive_gain_missed": 1 - positive / oracle if oracle else None,
                "selected_score_mean": predicted_mean,
                "selected_score_optimism": predicted_mean - point if predicted_mean is not None else None,
                "adopted": summary["policies"][policy]["adopted"], "harmful": summary["policies"][policy]["harmful"],
            }
        result["retrievers"][retriever] = metrics
    write(OUT / "READOUT.json", result, immutable=True)
    text = ["# 开发扩展结果解读", "", "本页在结果打开后作损益算术分解，无新增模型、阈值或方法选择。正式四项 RQ1 比较和预设主收益模型保持不变。", "",
            "1,490 条查询 / 512 个关联组的扩展、五折收益选择与独立复算已完成。确认池 8,530 条 / 1,963 组尚未运行。", "",
            "## 当前结果支持什么", "",
            "- 同一 ENTR 三候选池内，主收益模型相对原熵 Top1 改善 CodeBERT +.01242189、BM25 +.01040205（开发 OOF）；选择任务相关监督能减少部分原排序损失。",
            "- 对原查询的正净收益尚未建立：主模型加零收益 NoRewrite 动作后为 +.00020512 / −.00499483。线性辅助对照为 +.00409250 / −.00248255；CodeBERT 正点估计的区间跨零。",
            "- ENTR 对 RAND、PROB 的四项候选质量比较区间均跨零；BM25 的 ENTR−RAND 点估计为负。保留复现与检索器边界问题，不据此宣称熵普遍无效或等效。", "",
            "## 损失具体发生在哪", "",
            "本轮没有运行后置风险门。直接按预测收益选择并保留原查询，已经出现误选和收益抵消。", "",
            "| 检索器 | 池中存在更好候选的查询 | 三候选全部更差 | 主模型获得正收益 / 查询 | 主模型损失 / 查询 | 主模型净收益 |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, data in result["retrievers"].items():
        m = data["policies"]["GAIN_HGB_selective"]
        text.append(f'| {name} | {data["candidate_opportunity_queries"]}/1490 | {data["all_three_candidates_worse_than_original"]}/1490 | +{m["positive_gain_per_query"]:.8f} | −{m["damage_per_query"]:.8f} | {m["delta"]:+.8f} |')
    text += ["", "CodeBERT 主模型捕获约 29.6% 的候选可用正收益，但产生的损失几乎全部抵消收益；BM25 捕获约 7.8%，说明该检索器上既有候选正机会较少，也有选择识别问题。这里的可用收益使用标签，只是诊断上限。", "",
             "## 预测为正为何还会损失", "",
             "| 检索器 | 主模型选择后平均预测 ΔRR | 实际 ΔRR | 预测偏高 |", "|---|---:|---:|---:|"]
    for name, data in result["retrievers"].items():
        m = data["policies"]["GAIN_HGB_selective"]
        text.append(f'| {name} | {m["selected_score_mean"]:+.8f} | {m["delta"]:+.8f} | {m["selected_score_optimism"]:+.8f} |')
    text += ["", "候选总体均值接近真实均值，不意味着取最大值、再筛选预测正收益之后仍校准。上表是观察到的选择后预测偏差，不能单独归因为某一种统计机制。", "",
             "主模型在原试跑子集与新增开发子集（同一次五折 OOF）CodeBERT 净收益分别为 +.01555543 / −.00297939。线性对照在新增子集为 +.00371982，区间仍跨零；因此不把全开发主模型 +.00020512 当作稳定正收益。", "",
             "## 下一步的决策依据", "",
             "保留预定主模型和线性对照；先在现有开发缓存检验选择后的任务收益证据与校准，再冻结单一确认政策。没有依据仅靠放宽风险阈值解决本轮误选。", "",
             "功效缺口仍在：CodeBERT ENTR−RAND 的 .005 / 80% / 四比较保守规划需要 2,475 组，确认池仅 1,963 组；已知组大小差异的工作假设敏感性为 3,662 组。其余三项达到原变化事件数条件。确认前需明确精度目标与联合比较家族，保持原 .005 目标，不按当前结果改成更宽门槛。", "",
             "完整数值、区间、审计和冻结证据见 REPORT.md、RQ1_DEVELOPMENT.json、RQ2_DEVELOPMENT.json、POWER_PLAN.json。旧模型与正式指针保留，risk_certified=false，部署仍为 NoRewrite。", ""]
    with (OUT / "READOUT.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(text))
    write(OUT / "EFFECT_ACCESS_LEDGER.json", {"created_utc": now(), "all_D_permanently_consumed": True,
                                             "development_queries": 1490, "development_groups": 512,
                                             "prior_pilot": {"queries": 256, "groups": 86}, "new_D": {"queries": 1234, "groups": 426},
                                             "confirmation_method_effects_opened": False, "confirmation_queries": 8530, "confirmation_groups": 1963,
                                             "annotation_provenance_previously_accessed": True,
                                             "opening_markers": [f"metrics/{r}_OPENED.json" for r in protocol["retrievers"]],
                                             "training_markers": [f"GAIN_LABEL_ACCESS_{r}.json" for r in protocol["retrievers"]]}, immutable=True)
    output_paths = [OUT / n for n in ("READOUT.json", "READOUT.md", "EFFECT_ACCESS_LEDGER.json")]
    write(OUT / "READOUT_RECEIPT.json", {"passed": True, "created_utc": now(),
                                        "primary_acceptance_sha256": sha(OUT / "FINAL_ACCEPTANCE.json"),
                                        "protocol_sha256": sha(OUT / "PROTOCOL.json"), "primary_results_unmodified": True,
                                        "script": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha(__file__)},
                                        "inputs": [{"path": p.relative_to(OUT).as_posix(), "sha256": sha(p)} for p in sources],
                                        "outputs": [{"path": p.relative_to(OUT).as_posix(), "sha256": sha(p)} for p in output_paths]}, immutable=True)
    print(json.dumps({"readout_complete": True, "confirmation_opened": False, "models_changed": False}))


if __name__ == "__main__":
    main()
