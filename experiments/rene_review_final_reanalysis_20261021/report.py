"""Write a readable assessment of the reviewed labels and fixed sensitivity."""

from experiments.rene_review_final_reanalysis_20261021.analyze import OUT, put, put_rows
from experiments.rene_relevance_sensitivity_20261021.common import read, rows, sha


def interval(values):
    return "[" + ", ".join(f"{value:+.6f}" for value in values) + "]"


def selected(result, reviewer, policy, threshold=2, cutoff=3):
    return next(row for row in result["summary"] if
                (row["reviewer"], row["policy"], row["threshold"], row["cutoff"]) ==
                (reviewer, policy, threshold, cutoff))


def main():
    result = read(OUT / "RESULTS.json")
    validation = read(OUT / "NUMERICAL_VALIDATION.json")
    assert validation["passed"] and validation["results_sha256"] == sha(OUT / "RESULTS.json")
    assert result["supplement_status"]["A"]["independence_confirmed_in_return"] is False
    counts = result["supplement_agreement"]["label_counts"]
    original = result["original_review_agreement"]
    supplement = result["supplement_agreement"]
    pooled = result["pooled_top3_agreement"]
    lines = [
        "# 重新提交的评审 A/B 与补审 A/B：核查和结果解读",
        "",
        "四组材料的条目与固定审阅包匹配，网页、文字版与 JSON 的 2,108 项判断及理由已核对。"
        "原评审 A/B 的所有数据字段与上一版相同；本次新增的是两人各 344 个补审判断。"
        "补审后，预定主要诊断端点的两位审阅者加权相关性扩充收益界均在正侧，"
        "从而加强评价口径和自动采纳的诊断证据。",
        "",
        "状态：补审 B 通过原回收规则；补审 A 内容完整，但返回文件的独立完成确认值为 false，"
        "原回收程序拒绝正式接收。已向用户询问是否漏勾选，尚待答复。"
        "下文保留 A 的返回状态，按两份已返回标签作描述分析，不将其写成双份独立补审已正式验收。",
        "",
        "## 材料和原评审比较",
        "",
        "|材料|查询数|改写判断|代码判断|结果|",
        "|---|---:|---:|---:|---|",
        "|评审 A|128|383|327|与前版数据字段逐项相同|",
        "|评审 B|128|383|327|与前版数据字段逐项相同|",
        "|补审 A|126|—|344|项目完整；独立完成确认待补|",
        "|补审 B|126|—|344|项目完整；通过原回收规则|",
        "",
        "原评审文件仅存在序列化格式差异，不是新增样本或新增两位审阅者。"
        "语义一致率仍为 76.50%、κ=0.525；代码含 unknown 的一致率仍为 53.82%、κ=0.371；"
        "共同可判代码 260/327，线性加权 κ=0.569。原有加权错误比例和分歧统计保持。",
        "",
        "## 新增补审的一致性",
        "",
        "|补审判断|A|B|",
        "|---|---:|---:|",
    ]
    for label, name in (("2", "满足"), ("1", "部分满足"), ("0", "不满足"), ("u", "无法判断")):
        lines.append(f"|{name}|{counts['A'][label]}|{counts['B'][label]}|")
    lines += [
        "",
        f"344 个新面板的精确一致率 {supplement['including_unknown']['agreement']:.2%}，"
        f"名义 κ={supplement['including_unknown']['kappa']:.3f}；"
        f"共同可判 {supplement['jointly_judged']['n']}/344"
        f"（{supplement['jointly_judged_coverage']:.2%}），"
        f"线性加权 κ={supplement['jointly_judged']['kappa']:.3f}。",
        "",
        "154 项分歧中，48 项是 A 的 unknown 与 B 的确定判断不同，另外 106 项是双方均可判但等级不同。"
        "后一组中 B 高于 A 为 83 项，低于 A 为 23 项。新旧两批面板不同，κ 的差不能直接解释为评审质量升降，"
        "也不能用某人一贯宽松或严格概括所有分歧。",
        "",
        f"合并 Top-3 池共有 610 个不同代码面板，其中 266 个继承原评审、344 个来自补审。"
        f"池内 A/B 各有 49 个 unknown；双方共同可判 {pooled['jointly_judged']['n']}/610，"
        f"线性加权 κ={pooled['jointly_judged']['kappa']:.3f}。"
        "补审条目齐全与全部相关性已知分别报告，未将 unknown 填为不相关或自动裁决。",
        "",
        "## 预定主要诊断端点",
        "",
        "沿用原 128 查询/120 组、八层权重和固定动作，主要端点为 RIDGE_verified、RR@3、仅评分 2 纳入新增相关集合。"
        "扩充集合保留原 qrels 正例及池内相同代码文本；两个审阅者分别计算，未知面板按共享文本变量穷举。"
        "以下数值均为政策 RR 减原查询 RR 的加权样本识别界，"
        "不是置信区间，也不替换全 H 的 MRR@100。",
        "",
        "|审阅者|补审前范围|补审后范围|该配对差可确定的查询|",
        "|---|---|---|---:|",
    ]
    for row in result["primary_before_after"]:
        before, after = row["preview"], row["with_all_returned_supplement_labels"]
        lines.append(
            f"|{row['reviewer']}|{interval(before['weighted_expanded_delta_identification_bounds'])}|"
            f"{interval(after['weighted_expanded_delta_identification_bounds'])}|"
            f"{before['point_identified_queries']} → {after['point_identified_queries']}/128|"
        )
    lines += [
        "",
        "原 qrels 下同一 RR@3 端点本身为 +0.001594，不能将本表描述成该端点从负收益翻转为正收益。"
        "相对于原 qrels 差值的改变量，A 为 "
        + interval(selected(result, "A", "RIDGE_verified")["weighted_change_from_original_delta_bounds"])
        + "，B 为 "
        + interval(selected(result, "B", "RIDGE_verified")["weighted_change_from_original_delta_bounds"])
        + "；B 的改变量仍含 0。",
        "",
        "|敏感性口径|A：RIDGE_verified 差范围|B：RIDGE_verified 差范围|",
        "|---|---|---|",
        "|评分 1 或 2 均纳入，原分层加权 RR@3|"
        + interval(selected(result, "A", "RIDGE_verified", threshold=1)["weighted_expanded_delta_identification_bounds"])
        + "|"
        + interval(selected(result, "B", "RIDGE_verified", threshold=1)["weighted_expanded_delta_identification_bounds"])
        + "|",
        "|仅评分 2 纳入，不加权 RR@3|"
        + interval(selected(result, "A", "RIDGE_verified")["unweighted_expanded_delta_bounds"])
        + "|"
        + interval(selected(result, "B", "RIDGE_verified")["unweighted_expanded_delta_bounds"])
        + "|",
        "",
        "按预定分层加权，两种相关阈值下的两位审阅者收益界均为正；严格相关标准下 B 的未加权范围仍跨 0。"
        "因此结论限定于固定样本及其原加权分析，不能扩大为所有样本口径下都获益。",
        "",
        "## 对主线和贡献的影响",
        "",
        "|固定政策，评分 2、加权 RR@3|A 差范围|B 差范围|",
        "|---|---|---|",
    ]
    for policy in ("ENTR_top1", "RIDGE_verified", "RIDGE_risk", "SAFE_ENTR_best3_opportunity"):
        lines.append("|" + policy + "|"
                     + interval(selected(result, "A", policy)["weighted_expanded_delta_identification_bounds"])
                     + "|" + interval(selected(result, "B", policy)["weighted_expanded_delta_identification_bounds"]) + "|")
    lines += [
        "",
        "1. 直接选择 ENTR 第一条改写的加权差仍在负侧，两个相关阈值和两位审阅者均保持该方向。"
        "标签辅助的安全 best3 机会仍在正侧，继续支持候选机会与指定自动选择效果分开衡量。",
        "2. 核验阶段的可执行效用读数对相关性集合有实质敏感性。"
        "原先按单一目标排名计入的损失，不能全部等同于查询含义损坏或用户效用下降。",
        "3. 风险回退后尚无跨两位审阅者、两个相关阈值都可确定为正的结果，"
        "不能据本次补审宣布完整自动政策稳定获益。",
        "4. 本次主要补强第三贡献中的评价口径与固定政策诊断。"
        "B/G 受控比较、StaQC 确认、检索器和权重对照没有重新运行，其原结论保持。"
        "当前仍不识别语义、相关性覆盖与收益学习误差分别解释多少损失。",
        "",
        "## 代表性分歧",
        "",
        "下列案例分别来自 0/2、u/2 和 1/2 三类分歧，每类取 query_id/panel_id 字典序第一项；"
        "用于说明判断边界，保留双方标签，不作第三方裁决。",
        "",
    ]
    disagreements = list(rows(OUT / "supplement_disagreements.jsonl"))
    examples = []
    for pair, description in (
        (("0", "2"), "请求与响应的功能对象"),
        (("u", "2"), "缺少辅助函数与可见功能"),
        (("1", "2"), "字符串转换与完成输出"),
    ):
        options = sorted((r for r in disagreements if (r["A"]["relevance"], r["B"]["relevance"]) == pair),
                         key=lambda r: (r["query_id"], r["panel_id"]))
        example = options[0]
        examples.append({"category": description, **example})
        lines += [
            "### " + description,
            "",
            "查询：" + example["original"] + "（" + example["query_id"] + "）。",
            "",
            "A：" + example["A"]["relevance"] + "；" + example["A"]["note"],
            "",
            "B：" + example["B"]["relevance"] + "；" + example["B"]["note"],
            "",
        ]
    put(OUT / "REPRESENTATIVE_DISAGREEMENTS.json", {"selection": "lexicographic first in fixed label-pair categories",
                                                   "examples": examples, "labels_adjudicated": False})
    lines += [
        "## 完成状态与交接",
        "",
        "已完成四份内容核查、344×2 新判断映射、610 面板合并池分析、48 组合及 6,144 条精确配对界的独立复算。"
        "收到 688 个补审标签；原回收规则当前可接收 B 的 344 个，A 的 344 个独立完成确认待回复。"
        "原文件、原预览、原统计、论文源稿均保留。",
        "",
        "待办为确认 A 的实际独立完成状态，之后统一更新主稿中的补判状态、相关性敏感性表和评审分歧说明。"
        "不以提高 κ 为目的重写原判断。数值、逐查询界和分歧明细分别在 RESULTS.json、"
        "query_bounds.jsonl、supplement_disagreements.jsonl。",
        "",
    ]
    (OUT / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    put(OUT / "DELIVERY_MANIFEST.json", {
        "scope": "new_review_and_supplement_analysis_without_replacing_prior_evidence",
        "files": [{"path": str(p.relative_to(OUT)), "sha256": sha(p)}
                  for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in ("DELIVERY_MANIFEST.json", "ACCEPTANCE.json")],
    })
    acceptance = {
        "numeric_and_content_checks_passed": True,
        "independent_query_bound_replays": 6144,
        "display_judgments_and_reasons_checked": 2108,
        "returned_supplement_labels": 688,
        "supplement_content_complete": {"A": True, "B": True},
        "supplement_independence_confirmation_in_return": {"A": False, "B": True},
        "formal_dual_independent_supplement_completion": False,
        "pending": ["A actual independence confirmation; original return flag preserved"],
        "numerical_validation_sha256": sha(OUT / "NUMERICAL_VALIDATION.json"),
        "delivery_manifest_sha256": sha(OUT / "DELIVERY_MANIFEST.json"),
        "old_results_and_source_files_unchanged": True,
        "report": "FINDINGS.md",
    }
    put(OUT / "ACCEPTANCE.json", acceptance)
    print(acceptance)


if __name__ == "__main__":
    main()
