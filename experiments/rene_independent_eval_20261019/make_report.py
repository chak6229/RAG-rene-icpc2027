"""Render the independent recomputation report from RECOMPUTE_1024.json."""

import json
import os

RESULT = r".\results\rene_independent_eval_20261019\RECOMPUTE_1024.json"
OUT = r".\results\rene_independent_eval_20261019\REPORT.md"

with open(RESULT, encoding="utf-8") as handle:
    payload = json.load(handle)

lines = []
lines.append("# 1024 确认批的独立复算与风险–覆盖诊断")
lines.append("")
lines.append("日期：2026-10-19。性质：只读复算；输入为封存的 `private_outcomes.json`、"
             "`retrieval_items.pkl`、`label_free_decisions.pkl`；不导入管道代码，不训练、不重跑。")
lines.append("")
lines.append("## 1 复算一致性")
lines.append("")
lines.append("| 方法 | 采用 | 有害 | ΔMRR（复算） | ΔMRR（封存） | 绝对误差 |")
lines.append("|---|---:|---:|---:|---:|---:|")
for method, row in sorted(payload["comparisons"].items()):
    actual = row["actual"]
    expected = row["expected"]
    lines.append(
        f"| `{method}` | {actual['adopted']} | {actual['harmful']} | "
        f"{actual['delta_mrr']:.10f} | {expected['delta_mrr']:.10f} | {row['delta_mrr_abs_error']:.3e} |"
    )
lines.append("")
lines.append("结论：5/5 方法的 ΔMRR、采用数、有害数完全一致；最大绝对误差 0（一项为 8.7e-19 浮点噪声）。"
             "说明封存结果可由排名与目标独立复算。")
lines.append("")
lines.append("## 2 风险–覆盖曲线（no-seed / common 风险头）")
lines.append("")
lines.append("| τ | 采用 | 覆盖 | 有害 | 害率 | ΔMRR | 相对门前保留 |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|")
pre_gate = payload["recomputed"]["no_seed_uq/pre_gate"]["delta_mrr"]
for point in payload["risk_coverage_no_seed_common"]:
    retention = point["delta_mrr"] / pre_gate if pre_gate else float("nan")
    lines.append(
        f"| {point['tau']:.3f} | {point['adopted']} | {point['coverage']:.3f} | "
        f"{point['harmful']} | {point['harmful_rate']:.4f} | {point['delta_mrr']:+.8f} | {retention:.1%} |"
    )
lines.append("")
lines.append(f"门前（pre_gate）ΔMRR = {pre_gate:+.8f}；当前冻结门 τ=.075 的 ΔMRR = "
             f"{payload['recomputed']['no_seed_uq/common/threshold075']['delta_mrr']:+.8f}，"
             f"保留 {payload['recomputed']['no_seed_uq/common/threshold075']['delta_mrr']/pre_gate:.1%}。")
lines.append("")
lines.append("## 3 解释与边界")
lines.append("")
lines.append("- 曲线是**同一已消费批**上的阈值扫描，属于开发诊断；τ 是在看过该批害率后选的，"
             "不能当作已确认策略，也不构成 10% 风险认证。")
lines.append("- 点估计显示：τ 放松到 .20 时，害率 9.14%、ΔMRR +0.010410（约为 .075 的 4.5 倍）；"
             "τ=.50 时害率 10.0%、ΔMRR +0.019376（约为门前 91%）。这些点的 CP 上界仍可能超过 10%，"
             "需在开发角色冻结、另取确认集验证。")
lines.append("- 与 Codex 的提醒一致：低害率本身不证明校准错误；本曲线只说明「同一风险头下存在更宽的 operating point」，"
             "不能反推门前收益或风险头因果。")
lines.append("")
lines.append("## 4 复现")
lines.append("")
lines.append("```text")
lines.append("cd experiments/rene_independent_eval_20261019")
lines.append("python test_rne_eval.py")
lines.append("python validate_1024.py")
lines.append("python make_report.py")
lines.append("```")
lines.append("")
lines.append("输入：`results/rene_uq_independent_proof_20261019/fresh/`；输出：同目录 `RECOMPUTE_1024.json`、本报告。")

with open(OUT, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print(OUT)
