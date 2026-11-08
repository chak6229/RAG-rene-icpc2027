"""Audit pair annotations and duplicate structure, without ranking a model."""
from __future__ import annotations

import collections
import hashlib
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

from discover_sources import ROOT, OUT, SOURCES


def sha_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def query_key(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def token_key(value) -> str:
    return " ".join(value.split() if isinstance(value, str) else map(str, value))


class ReadableHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.suppressed = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.suppressed += 1
        if not self.suppressed and tag in {"p", "h1", "h2", "h3", "h4", "li", "tr", "div", "section", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.suppressed = max(0, self.suppressed - 1)
        if not self.suppressed and tag in {"p", "h1", "h2", "h3", "h4", "li", "tr", "div", "section"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(data)


def main():
    base = ROOT / "imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA"
    splits = {name: json.loads((base / f"cosqa-{name}.json").read_text(encoding="utf-8")) for name in ("all", "train", "dev")}
    report = {"scope": "annotation/provenance only; no predictions, retrieval scores, or effects read", "splits": {}}
    for name, rows in splits.items():
        by_query = collections.defaultdict(list)
        pair_labels = collections.defaultdict(set)
        for row in rows:
            q = query_key(row["doc"])
            c = sha_bytes(token_key(row["code_tokens"]))
            by_query[q].append(row)
            pair_labels[(q, c)].add(row["label"])
        positives = {q: sum(r["label"] == 1 for r in group) for q, group in by_query.items()}
        lengths = sorted(len(row["doc"].split()) for row in rows)
        report["splits"][name] = {
            "pairs": len(rows),
            "label_counts": dict(collections.Counter(str(row["label"]) for row in rows)),
            "unique_queries_raw": len({r["doc"] for r in rows}),
            "unique_queries_normalized": len(by_query),
            "unique_code_raw": len({r["code"] for r in rows}),
            "unique_code_tokens": len({sha_bytes(token_key(r["code_tokens"])) for r in rows}),
            "unique_pair_query_code_tokens": len(pair_labels),
            "duplicate_pair_rows": len(rows) - len(pair_labels),
            "conflicting_normalized_pairs": sum(len(labels) > 1 for labels in pair_labels.values()),
            "queries_with_known_positive": sum(n > 0 for n in positives.values()),
            "queries_without_known_positive": sum(n == 0 for n in positives.values()),
            "queries_with_multiple_positive_pairs": sum(n > 1 for n in positives.values()),
            "positive_count_per_query": dict(sorted(collections.Counter(positives.values()).items())),
            "labeled_pair_count_per_query": dict(sorted(collections.Counter(len(v) for v in by_query.values()).items())),
            "query_word_lengths": {"min": lengths[0], "median": lengths[len(lengths)//2], "p95": lengths[int(.95*(len(lengths)-1))], "max": lengths[-1]},
        }
    train, dev = splits["train"], splits["dev"]
    report["train_dev_overlaps"] = {
        "normalized_queries": len({query_key(r["doc"]) for r in train} & {query_key(r["doc"]) for r in dev}),
        "raw_codes": len({r["code"] for r in train} & {r["code"] for r in dev}),
        "code_tokens": len({sha_bytes(token_key(r["code_tokens"])) for r in train} & {sha_bytes(token_key(r["code_tokens"])) for r in dev}),
    }
    def records(rows):
        return collections.Counter((r["idx"], r["doc"], r["code"], r["label"]) for r in rows)
    report["all_equals_train_plus_dev_as_multiset"] = records(splits["all"]) == records(train + dev)
    report["input_files"] = [{"path": (base / f"cosqa-{name}.json").relative_to(ROOT).as_posix(), "sha256": hashlib.sha256((base/f"cosqa-{name}.json").read_bytes()).hexdigest()} for name in splits]
    (OUT / "DATASET_STATISTICS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for name in ("cosqa_arxiv_html",):
        parser = ReadableHTML()
        parser.feed((SOURCES / f"{name}.html").read_text(encoding="utf-8"))
        text = "\n".join(re.sub(r"\s+", " ", line).strip() for line in "".join(parser.parts).splitlines() if line.strip())
        (SOURCES / f"{name}.txt").write_text(text, encoding="utf-8")
    for row in splits["all"][:2]:
        print(json.dumps({k: row[k] if k in {"idx", "label"} else str(row[k])[:220] for k in row}, ensure_ascii=False))


if __name__ == "__main__":
    main()
