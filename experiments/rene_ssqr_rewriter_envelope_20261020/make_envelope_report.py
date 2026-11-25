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
lines.append('# Multiple rewriter Package (512 - query sensitivity subset)')
lines.append("")
lines.append('Date: 2026 - 10 - 20. Local CQC T5 with t5-base batch query, insert in the same position, greedy settings,'
             'A comparison under the same full-word list teacher-forcing entropy, the same CodeBERT pooler searcher.')
lines.append("")
lines.append('| Data ReWriter| Query NoRewrite| ENTR  Top- 1  |  ENTR  best3 | Oracle-all | Δ Top- 1  | Δ best3 | Δ Oracle | Spearman(ENTR, RR) |')
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
lines.append('**Read:**')
lines.append('- If both rewriter appear `Top-1 < NoRewrite`, `best3 > NoRewrite`, `Oracle >> NoRewrite`,'
             'This indicates that the failure in selecting a signal is not an accident of one T5.')
lines.append('- Spearman is the negative expression "the lower the entropy, the higher the RR" and in the same direction as SSQR; Quantities reflect a strong signal and are not proof of monotonous usability.')
lines.append('- Oracle-all is the upper margin of reading and not a deployable method; Top-1 is a single choice, best3 is a candidate set of indicators.')
lines.append("")
if missing:
    lines.append(f"Not completed:{', '.join(missing)}。")
    lines.append("")
lines.append('## Revert')
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
