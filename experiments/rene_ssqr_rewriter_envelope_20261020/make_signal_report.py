"""Render the signal + positive-control report from SIGNAL_POSITIVE_CONTROL.json."""

import json
import os

ROOT = r"."
SRC = os.path.join(ROOT, r"results\rene_ssqr_rewriter_envelope_20261020\SIGNAL_POSITIVE_CONTROL.json")
OUT = os.path.join(ROOT, r"results\rene_ssqr_rewriter_envelope_20261020\SIGNAL_POSITIVE_CONTROL.md")

with open(SRC, encoding="utf-8") as handle:
    data = json.load(handle)

lines = []
lines.append("# 正向对照与信号级分析（SSQR bridge 冻结缓存）")
lines.append("")
lines.append("日期：2026-10-20。性质：post-hoc，只读冻结候选/排名/qrels；未训练、未生成新候选。")
lines.append("")

lines.append("## 1 正向对照：候选集有价值，但 Top-1 有害")
lines.append("")
lines.append("| 数据 | 查询 | NoRewrite | ENTR Top-1 | ENTR best3 | PROB best3 | Oracle-all |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for key in ("source_main", "cosqa_main"):
    row = data[key]
    m = row["mrr"]
    lines.append(
        f"| {row['label']} | {row['queries']} | {m['nowrite']:.4f} | {m['entr_top1']:.4f} | "
        f"{m['entr_best3']:.4f} | {m['prob_best3']:.4f} | {m['oracle']:.4f} |"
    )
lines.append("")
lines.append("| 数据 | Δ Top-1 | Δ best3 | Δ PROB best3 | Δ Oracle |")
lines.append("|---|---:|---:|---:|---:|")
for key in ("source_main", "cosqa_main"):
    row = data[key]
    d = row["delta"]
    lines.append(
        f"| {row['label']} | {d['entr_top1']:+.4f} | {d['entr_best3']:+.4f} | "
        f"{d['prob_best3']:+.4f} | {d['oracle']:+.4f} |"
    )
lines.append("")
lines.append("**读法**：Oracle 远高于 NoRewrite，说明生成器足够好、候选集里确实有收益；"
             "best3 捕获了一部分；Top-1 反而低于 NoRewrite。**问题在单选择，不在生成器。**")
lines.append("")

lines.append("## 2 信号级相关：弱信息，但在极端点反向")
lines.append("")
lines.append("| 数据 | Spearman(ENTR, RR) | Spearman(PROB, RR) | 随机置换 | ENTR 负相关 query 占比 |")
lines.append("|---|---:|---:|---:|---:|")
for key in ("source_main", "cosqa_main"):
    row = data[key]
    s = row["spearman"]
    lines.append(
        f"| {row['label']} | {s['entropy']:+.3f} | {s['prob']:+.3f} | {s['random']:+.3f} | "
        f"{s['entropy_negative_fraction']:.1%} |"
    )
lines.append("")
lines.append("- ENTR 的 Spearman 为负，表示“熵越低、RR 越高”，方向与 SSQR 一致；"
             "但量级很弱（源域约 −0.03，CoSQA 约 −0.13），且 Top-1 仍低于 NoRewrite。")
lines.append("- PROB 的 Spearman 为正，方向也符合“概率越高越好”，但同样很弱。")
lines.append("- 这说明信号在**整体排序**上有微弱信息，但在**最低熵/最高概率的极端点**上不成立；"
             "Top-1 恰好取的是极端点。")
lines.append("")

lines.append("## 3 解释与边界")
lines.append("")
lines.append("- 这是 post-hoc 诊断：候选、排名、qrels 都来自已冻结/已消费的 bridge，不是新确认。")
lines.append("- `Oracle-all` 是读数上界，不是可部署方法。")
lines.append("- Spearman 只在 ≥3 个位置且非全并列的 query 上计算；缺失/退化 query 已计入 `missing`。")
lines.append("- 结论只针对这套重建（本地 T5 + CodeBERT pooler）；多 rewriter 包络另报。")
lines.append("")
lines.append("## 4 复现")
lines.append("")
lines.append("```text")
lines.append("cd experiments/rene_ssqr_rewriter_envelope_20261020")
lines.append("python analyze_signal.py")
lines.append("python make_signal_report.py")
lines.append("```")

with open(OUT, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print(OUT)
