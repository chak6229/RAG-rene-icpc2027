"""CoSQA adapter: Bing doc queries, code-only corpus, and separate positive qrels."""
from __future__ import annotations

import json
from pathlib import Path

from .common import OUT, ROOT, read_json, sha256, write_json

ALL_SOURCE = ROOT / "imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-all.json"
SEARCH_SOURCE = ROOT / "results/rene_cosqa_ssqr_audit_20261019/selected_source/coclr/data/search"


def token_text(value, *, field):
    if isinstance(value, str):
        result = value.strip()
    elif isinstance(value, list) and all(isinstance(x, str) for x in value):
        result = " ".join(value).strip()
    else:
        raise ValueError(f"{field} must be a string or a list of strings")
    if not result:
        raise ValueError(f"{field} is empty")
    return result


def query_input(row):
    query = row.get("doc")
    query_id = row.get("idx")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("CoSQA requires nonempty doc; paired code documentation is not a query fallback")
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError("CoSQA requires a stable idx")
    # Whitelist fields so labels, target IDs and paired documentation cannot enter inference inputs.
    return {"query_id": query_id, "query": query.strip()}


def label_value(row):
    value = row.get("label")
    if value not in (0, 1, "0", "1"):
        raise ValueError("CoSQA label must be binary")
    return int(value)


def adapt(rows, code_id_map):
    if not isinstance(code_id_map, dict) or not code_id_map:
        raise ValueError("official raw-code to corpus-ID mapping is required")
    if set(code_id_map.values()) != set(range(len(code_id_map))):
        raise ValueError("official corpus IDs must be unique and contiguous")
    corpus_text = {}
    queries, qrels, negatives = [], [], 0
    seen_ids = set()
    for row in rows:
        public = query_input(row)
        if public["query_id"] in seen_ids:
            raise ValueError("duplicate query IDs require explicit grouping, not silent overwrites")
        seen_ids.add(public["query_id"])
        code = row.get("code")
        if not isinstance(code, str) or code not in code_id_map:
            raise ValueError("CoSQA raw code does not map to official corpus")
        doc_id = code_id_map[code]
        if "retrieval_idx" in row and row["retrieval_idx"] != doc_id:
            raise ValueError("retrieval_idx conflicts with official corpus ID")
        text = token_text(row.get("code_tokens"), field="code_tokens")
        if doc_id in corpus_text and corpus_text[doc_id] != text:
            raise ValueError("inconsistent code-only token text for one corpus ID")
        corpus_text[doc_id] = text
        if label_value(row) == 1:
            queries.append(public)
            qrels.append({"query_id": public["query_id"], "doc_id": doc_id, "relevance": 1})
        else:
            negatives += 1
    corpus = [{"doc_id": doc_id, "code_text": corpus_text[doc_id]} for doc_id in sorted(corpus_text)]
    return queries, corpus, qrels, negatives


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def export():
    rows = read_json(ALL_SOURCE)
    code_map = read_json(SEARCH_SOURCE / "code_idx_map.txt")
    queries, corpus, qrels, negatives = adapt(rows, code_map)
    if (len(rows), len(corpus), len(queries), negatives) != (20604, 6267, 10020, 10584):
        raise ValueError("CoSQA structural counts differ from the completed audit")
    if {r["doc_id"] for r in corpus} != set(range(6267)):
        raise ValueError("incomplete corpus")
    output = OUT / "cosqa_adapter"
    files = []

    def emit(name, items):
        path = output / name
        write_jsonl(path, items)
        files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "rows": len(items)})

    emit("public/corpus.jsonl", corpus)
    emit("public/all_positive_queries.jsonl", queries)
    emit("private/all_positive_qrels.jsonl", qrels)
    sources = [ALL_SOURCE, SEARCH_SOURCE / "code_idx_map.txt"]
    qrel_by_id = {row["query_id"]: row for row in qrels}
    query_by_id = {row["query_id"]: row for row in queries}
    code_by_id = {row["doc_id"]: row for row in corpus}
    for split in ("dev", "test"):
        path = SEARCH_SOURCE / f"cosqa-retrieval-{split}-500.json"
        sources.append(path)
        official = read_json(path)
        official_queries, official_corpus, official_qrels, official_negatives = adapt(official, code_map)
        if len(official_queries) != 500 or official_negatives:
            raise ValueError("official retrieval split must contain 500 positive queries")
        if any(qrel_by_id[row["query_id"]] != row for row in official_qrels):
            raise ValueError("official retrieval qrels disagree with CoSQA-all")
        if any(query_by_id[row["query_id"]] != row for row in official_queries):
            raise ValueError("official retrieval queries disagree with CoSQA-all")
        if any(code_by_id[row["doc_id"]] != row for row in official_corpus):
            raise ValueError("official retrieval code tokens disagree with CoSQA-all")
        emit(f"public/official_{split}_queries.jsonl", official_queries)
        emit(f"private/official_{split}_qrels.jsonl", official_qrels)
    manifest = {"status": "structurally_validated_no_effect_evaluation", "query_field": "doc",
                "corpus_field": "code_tokens", "corpus_ids": "official code_idx_map.txt",
                "paired_documentation_used_as_query": False, "query_input_fields": ["query_id", "query"],
                "corpus_input_fields": ["doc_id", "code_text"], "unjudged_is_negative": False,
                "negative_pairs_are_no_answer_queries": False, "roles_assigned": False,
                "official_splits_are_group_disjoint": False, "other_positives_may_exist": True,
                "private_is_file_separation_not_os_access_control": True,
                "raw_pairs": len(rows), "positive_queries": len(queries), "negative_pairs_excluded": negatives,
                "corpus_size": len(corpus), "files": files,
                "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in sources],
                "adapter_sha256": sha256(Path(__file__))}
    write_json(output / "MANIFEST.json", manifest)
    return manifest


if __name__ == "__main__":
    result = export()
    print(json.dumps({key: result[key] for key in ("status", "raw_pairs", "positive_queries", "corpus_size")}, ensure_ascii=False))
