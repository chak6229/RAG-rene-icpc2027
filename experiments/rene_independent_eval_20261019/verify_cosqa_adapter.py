"""Read-only verification of the CoSQA adapter export.

Checks the three interface claims (query=doc, no character-level join, qrels
separated from queries) plus counts, ID alignment, split overlap and manifest
hashes. Does not import the adapter or run any model.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

ROOT = r"."
BASE = os.path.join(ROOT, r"results\rene_ssqr_alignment_20261019\cosqa_adapter")
RAW = os.path.join(ROOT, r"imports\CodeXGLUE-main\Text-Code\NL-code-search-WebQuery\CoSQA\cosqa-all.json")
OUT = os.path.join(ROOT, r"results\rene_independent_eval_20261019\COSQA_ADAPTER_CHECK.json")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def char_fraction(text):
    tokens = text.split()
    if not tokens:
        return 1.0
    return sum(1 for token in tokens if len(token) == 1) / len(tokens)


def percentile(values, ratio):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * ratio))]


def main():
    queries = load_jsonl(os.path.join(BASE, "public", "all_positive_queries.jsonl"))
    corpus = load_jsonl(os.path.join(BASE, "public", "corpus.jsonl"))
    qrels = load_jsonl(os.path.join(BASE, "private", "all_positive_qrels.jsonl"))
    dev_q = load_jsonl(os.path.join(BASE, "public", "official_dev_queries.jsonl"))
    test_q = load_jsonl(os.path.join(BASE, "public", "official_test_queries.jsonl"))
    dev_r = load_jsonl(os.path.join(BASE, "private", "official_dev_qrels.jsonl"))
    test_r = load_jsonl(os.path.join(BASE, "private", "official_test_qrels.jsonl"))

    query_fractions = [char_fraction(row["query"]) for row in queries]
    corpus_fractions = [char_fraction(row["code_text"]) for row in corpus]
    raw = json.load(open(RAW, encoding="utf-8"))
    raw_by_idx = {row["idx"]: row for row in raw}
    mismatches = [
        row["query_id"]
        for row in queries
        if raw_by_idx.get(row["query_id"], {}).get("doc") != row["query"]
    ]

    manifest = json.load(open(os.path.join(BASE, "MANIFEST.json"), encoding="utf-8"))
    bad_hashes = []
    for item in manifest["files"]:
        path = os.path.join(ROOT, item["path"].replace("/", os.sep))
        if not os.path.exists(path):
            bad_hashes.append({"path": item["path"], "status": "missing"})
        elif sha256(path) != item["sha256"]:
            bad_hashes.append({"path": item["path"], "status": "hash-mismatch"})

    qids = {row["query_id"] for row in queries}
    rids = {row["query_id"] for row in qrels}
    dev_ids = {row["query_id"] for row in dev_q}
    test_ids = {row["query_id"] for row in test_q}
    dev_map = {row["query_id"]: row for row in dev_r}
    test_map = {row["query_id"]: row for row in test_r}

    payload = {
        "schema": "cosqa-adapter-independent-check-20261019-v1",
        "counts": {
            "queries": len(queries),
            "corpus": len(corpus),
            "qrels": len(qrels),
            "dev_queries": len(dev_q),
            "test_queries": len(test_q),
            "dev_qrels": len(dev_r),
            "test_qrels": len(test_r),
        },
        "interface": {
            "query_keys": sorted(queries[0].keys()),
            "corpus_keys": sorted(corpus[0].keys()),
            "qrels_keys": sorted(qrels[0].keys()),
            "public_query_has_label_or_doc": any(
                key not in ("query_id", "query") for row in queries for key in row
            ),
            "query_char_fraction_p50": percentile(query_fractions, 0.50),
            "query_char_fraction_max": max(query_fractions),
            "queries_over_50pct_single_char": sum(1 for value in query_fractions if value > 0.5),
            "corpus_char_fraction_p50": percentile(corpus_fractions, 0.50),
            "raw_doc_mismatches": len(mismatches),
            "raw_doc_mismatch_examples": mismatches[:5],
        },
        "qrels": {
            "unique_query_ids": len(rids),
            "same_query_set": qids == rids,
            "relevance_values": dict(Counter(row["relevance"] for row in qrels)),
            "doc_id_min": min(row["doc_id"] for row in qrels),
            "doc_id_max": max(row["doc_id"] for row in qrels),
            "missing_docs": len({row["doc_id"] for row in qrels} - {row["doc_id"] for row in corpus}),
        },
        "splits": {
            "dev_test_query_overlap": len(dev_ids & test_ids),
            "dev_qrels_agree_all": all(dev_map.get(row["query_id"]) == row for row in qrels if row["query_id"] in dev_map),
            "test_qrels_agree_all": all(test_map.get(row["query_id"]) == row for row in qrels if row["query_id"] in test_map),
        },
        "manifest": {
            "files": len(manifest["files"]),
            "bad_hashes": bad_hashes,
            "query_field": manifest["query_field"],
            "corpus_field": manifest["corpus_field"],
            "paired_documentation_used_as_query": manifest["paired_documentation_used_as_query"],
            "unjudged_is_negative": manifest["unjudged_is_negative"],
            "official_splits_are_group_disjoint": manifest["official_splits_are_group_disjoint"],
        },
        "verdict": {
            "query_is_doc": not mismatches and not any(key not in ("query_id", "query") for row in queries for key in row),
            "no_character_level_join": max(query_fractions) < 0.5 and sum(1 for value in query_fractions if value > 0.5) == 0,
            "qrels_separated": qids == rids and "relevance" not in queries[0] and "doc_id" not in queries[0],
            "counts_match": (len(queries), len(corpus), len(qrels)) == (10020, 6267, 10020),
            "manifest_hashes_match": not bad_hashes,
        },
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
