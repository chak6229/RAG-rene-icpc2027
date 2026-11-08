"""Render the multi-rewriter envelope table from per-run JSON files."""

import json
import os

ROOT = r"."
OUT = os.path.join(ROOT, r"results\rene_ssqr_rewriter_envelope_20261020")
FILES = [
    ("source", "local", "source_local.json"),
    ("source", "t5base", "source_t5base.json"),
    ("cosqa", "local", "cosqa_local.json"),
    ("cosqa", "t5base", "cosqa_t5base.json"),
]

rows = []
for domain, rewriter, name in FILES:
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        rows.append((domain, rewriter, None))
        continue
    with open(path, encoding="utf-8") as handle:
        rows.append((domain, rewriter, json.load(handle)))

lines = []
lines.append("# 多 rewriter 包络（512-query sensitivity subset）")
lines.append("")
lines.append("日期：2026-10-20。本地 CQC T5 与 t5-base 在同一批查询、同一插入位置、同一 greedy 设置、"
             "同一全词表 teacher-forcing 熵、同一 CodeBERT pooler 检索器下比较。")
lines.append("")
lines.append("| 数据 | ReWriter | 查询 | NoRewrite | ENTR Top-1 | ENTR best3 | Oracle-all | Δ Top-1 | Δ best3 | Δ Oracle | Spearman(ENTR,RR) |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
missing = []
for domain, rewriter, data in rows:
    if data is None:
        missing.append(f"{domain}/{rewriter}")
        lines.append(f"| {domain} | {rewriter} | — | — | — | — | — | — | — | — | — |")
        continue
    m, d = data["mrr"], data["delta"]
    lines.append(
        f"| {domain} | {rewriter} | {data['queries']} | {m['nowrite']:.4f} | {m['entr_top1']:.4f} | "
        f"{m['entr_best3']:.4f} | {m['oracle']:.4f} | {d['entr_top1']:+.4f} | {d['entr_best3']:+.4f} | "
        f"{d['oracle']:+.4f} | {data['spearman_entropy']:+.3f} |"
    )
lines.append("")
lines.append("**读法**：")
lines.append("- 如果两个 rewriter 都出现 `Top-1 < NoRewrite`、`best3 > NoRewrite`、`Oracle >> NoRewrite`，"
             "说明失败在选择信号，不是某一个 T5 的偶然。")
lines.append("- Spearman 为负表示“熵越低、RR 越高”，方向与 SSQR 一致；量级反映信号强弱，不是单调可用性的证明。")
lines.append("- Oracle-all 是读数上界，不是可部署方法；Top-1 是单选择，best3 是候选集指标。")
lines.append("")
if missing:
    lines.append(f"未完成：{', '.join(missing)}。")
    lines.append("")
lines.append("## 复现")
lines.append("")
lines.append("```text")
lines.append("cd experiments/rene_ssqr_rewriter_envelope_20261020")
lines.append("python generate_envelope.py --domain cosqa --rewriter local")
lines.append("python generate_envelope.py --domain cosqa --rewriter t5base")
lines.append("python generate_envelope.py --domain source --rewriter local")
lines.append("python generate_envelope.py --domain source --rewriter t5base")
lines.append("python make_envelope_report.py")
lines.append("```")

with open(os.path.join(OUT, "ENVELOPE.md"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print(os.path.join(OUT, "ENVELOPE.md"))
