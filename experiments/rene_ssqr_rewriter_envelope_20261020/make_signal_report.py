"""Render the signal + positive-control report from SIGNAL_POSITIVE_CONTROL.json."""

import json
import os

ROOT = r"."
SRC = os.path.join(ROOT, r"results\rene_ssqr_rewriter_envelope_20261020\SIGNAL_POSITIVE_CONTROL.json")
OUT = os.path.join(ROOT, r"results\rene_ssqr_rewriter_envelope_20261020\SIGNAL_POSITIVE_CONTROL.md")

with open(SRC, encoding="utf-8") as handle:
    data = json.load(handle)

lines = []
lines.append('# Contrast and signal level analysis (SSQR bridge freeze cache)')
lines.append("")
lines.append('Date: 2026 - 10 - 20. Nature: post-hoc, reading only frozen candidate/rank/qrels; No training, no new candidates.')
lines.append("")

lines.append('## 1 Contrast: Candidates are valuable, but Top-1 Harmful')
lines.append("")
lines.append('Data Query ENTR Top-1 ENTR best3 PROB best3  O Oracle-all')
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for key in ("source_main", "cosqa_main"):
    row = data[key]
    m = row["mrr"]
    lines.append(
        f"| {row['label']} | {row['queries']} | {m['nowrite']:.4f} | {m['entr_top1']:.4f} | "
        f"{m['entr_best3']:.4f} | {m['prob_best3']:.4f} | {m['oracle']:.4f} |"
    )
lines.append("")
lines.append('| Data Δ Top-1| best3| PROB best3|Oracle|')
lines.append("|---|---:|---:|---:|---:|")
for key in ("source_main", "cosqa_main"):
    row = data[key]
    d = row["delta"]
    lines.append(
        f"| {row['label']} | {d['entr_top1']:+.4f} | {d['entr_best3']:+.4f} | "
        f"{d['prob_best3']:+.4f} | {d['oracle']:+.4f} |"
    )
lines.append("")
lines.append('** Reader**: Oracle is much higher than NoRewrite, indicating that the generator is good enough and there are real gains in the candidate pool;'
             'best3 Captured a portion; Top-1 is lower than NoRewrite. ** Problem is single, not generator. **')
lines.append("")

lines.append('## 2 Signal level is relevant: weak information, but reverse at extreme point')
lines.append("")
lines.append('| Data Spearman (ENTR, RR) Spearman (PROB, RR) | Random replacement of | ENTR Negative relevance query')
lines.append("|---|---:|---:|---:|---:|")
for key in ("source_main", "cosqa_main"):
    row = data[key]
    s = row["spearman"]
    lines.append(
        f"| {row['label']} | {s['entropy']:+.3f} | {s['prob']:+.3f} | {s['random']:+.3f} | "
        f"{s['entropy_negative_fraction']:.1%} |"
    )
lines.append("")
lines.append('- ENTR Spearman is negative and means “the lower the entropy, the higher the RR”, in the same direction as SSQR;'
             'However, the volume level is weak (source domain approximately −0.03, CoSQA about −0.13 and Top-1 is still below NoRewrite.')
lines.append('- PROB Spearman is positive, and the direction corresponds to "the higher the probability, the better", but it is equally weak.')
lines.append('- This indicates that the signal has weak information in its overall ranking**, but is not valid at ** the extreme point of lowest entropy/highest probability**;'
             'Top-1 That just happens to be the extreme point.')
lines.append("")

lines.append('## 3 Interpretation and boundary')
lines.append("")
lines.append('- This is post-hoc diagnostic: candidate, ranking, qrels all from frozen/consumption bridge, not new confirmation.')
lines.append('- `Oracle-all` is upper reading, not deployable.')
lines.append('- Spearman is calculated only on the query of ≥3 at a non-exhaustive parallel location; Missing/degraded query is included in `missing`.')
lines.append('- The conclusions are only for this set of reconstructions (local T5 + CodeBERT pooler); Many rewriter packages are reported separately.')
lines.append("")
lines.append('## 4 Recurring')
lines.append("")
lines.append("```text")
lines.append("cd experiments/rene_ssqr_rewriter_envelope_20261020")
lines.append("python analyze_signal.py")
lines.append("python make_signal_report.py")
lines.append("```")

with open(OUT, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print(OUT)
