"""Streaming query/code/repository overlap audit; no retrieval or outcomes."""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import pickle
import re
import time
from pathlib import Path

from discover_sources import ROOT, OUT
from dataset_statistics import query_key, token_key, sha_bytes


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(2**22), b""):
            digest.update(data)
    return digest.hexdigest()


def load_pickle(relative):
    with (ROOT / relative).open("rb") as handle:
        return pickle.load(handle)


def word_query(value):
    return " ".join(re.findall(r"\w+", str(value).casefold()))


def get_repo(row):
    repo = row.get("repo") or row.get("nwo")
    if repo:
        return str(repo).casefold()
    match = re.match(r"https?://github\.com/([^/]+/[^/]+)", str(row.get("url", "")))
    return match.group(1).casefold() if match else None


def main():
    started = time.monotonic()
    raw_path = ROOT / "imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-all.json"
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    code_rows = {sha_bytes(token_key(row["code_tokens"])): row for row in rows}
    targets = {kind: collections.defaultdict(set) for kind in ("raw", "whitespace", "code_tokens")}
    for cid, row in code_rows.items():
        targets["raw"][sha_bytes(row["code"])].add(cid)
        targets["whitespace"][sha_bytes(" ".join(row["code"].split()))].add(cid)
        targets["code_tokens"][cid].add(cid)
    queries = {query_key(row["doc"]) for row in rows}
    queries_lex = collections.defaultdict(set)
    for q in queries:
        queries_lex[word_query(q)].add(q)
    query_to_code = {query_key(row["doc"]): sha_bytes(token_key(row["code_tokens"])) for row in rows}
    positives = {query_key(row["doc"]) for row in rows if row["label"] == 1}
    coderepos = collections.defaultdict(set)
    source_details = []
    input_files = [{"path": raw_path.relative_to(ROOT).as_posix(), "sha256": file_sha(raw_path)}]
    matched_source_records = []

    historical = {}
    roles_by_train_index = collections.defaultdict(set)
    for name, path in (
        ("historical_action_4096", "results/uq_pilot_20261014/action_witness_uq/data/action_code_corpus.pkl"),
        ("historical_semantic_4096", "results/uq_pilot_20261014/action_witness_uq/data/semantic_code_corpus.pkl"),
    ):
        items = load_pickle(path)
        historical[name] = {"path": path, "rows": len(items), "matched_codes": set(),
                            "repos": {get_repo(item) for item in items if get_repo(item)}, "urls_by_index": {int(item["source_index"]): item["url"] for item in items}}
        for item in items:
            roles_by_train_index[int(item["source_index"])].add(name)
        input_files.append({"path": path, "sha256": file_sha(ROOT/path)})
    eligible_path = "results/rene_strict_confirmation_20261019/fresh/eligible_repositories.pkl"
    eligible = load_pickle(eligible_path)
    code_to_eligible = {item["code_hash"]: item for item in eligible}
    eligible_by_repo = {item["repo"]: item for item in eligible}
    assert len(eligible_by_repo) == len(eligible)
    corpus_path = "results/rene_strict_confirmation_20261019/fresh/corpus/block0.pkl"
    corpus = load_pickle(corpus_path)
    resolved = [code_to_eligible[sha_bytes(" ".join(item["code_text"].split()))] for item in corpus]
    assert len(resolved) == 4096
    historical["strict_4096_original_source"] = {"path": corpus_path, "rows": len(resolved), "matched_codes": set(),
        "repos": {item["repo"] for item in resolved}, "urls_by_index": {int(item["source_index"]): item["source_url"] for item in resolved}}
    for item in resolved:
        roles_by_train_index[int(item["source_index"])].add("strict_4096_original_source")
    for path in (eligible_path, corpus_path):
        input_files.append({"path": path, "sha256": file_sha(ROOT/path)})
    for name, path in (
        ("strict_1536_query_targets", "results/rene_strict_confirmation_20261019/fresh/queries.json"),
        ("consumed_1024_query_targets", "results/rene_uq_independent_proof_20261019/fresh/queries.json"),
    ):
        items = json.loads((ROOT/path).read_text(encoding="utf-8"))
        historical[name] = {"path": path, "rows": len(items), "matched_codes": set(),
                            "repos": {item["repo"] for item in items}, "urls_by_index": {}}
        for item in items:
            source = eligible_by_repo[item["repo"]]
            if source["query"] != item["query"]:
                raise ValueError("query metadata differs from repository inventory")
            roles_by_train_index[int(source["source_index"])].add(name)
        input_files.append({"path": path, "sha256": file_sha(ROOT/path)})
        exact = {query_key(item["query"]) for item in items} & queries
        lex = {q for item in items for q in queries_lex.get(word_query(item["query"]), set())}
        historical[name]["query_exact_overlap"] = len(exact)
        historical[name]["query_lexical_overlap"] = len(lex)

    def scan(name, paths, *, original_training=False):
        counts = collections.Counter()
        hits = {kind: set() for kind in targets}
        qhits, qlexhits, repos = set(), set(), set()
        for path in paths:
            input_files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha(path)})
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as stream:
                for index, line in enumerate(stream):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    counts["rows"] += 1
                    code = row.get("code") or row.get("function") or ""
                    tokens = row.get("code_tokens") or row.get("function_tokens") or []
                    fingerprints = {"raw": sha_bytes(code), "whitespace": sha_bytes(" ".join(code.split())),
                                    "code_tokens": sha_bytes(token_key(tokens))}
                    matches = {kind: targets[kind].get(value, set()) for kind, value in fingerprints.items()}
                    matched = set().union(*matches.values())
                    repo = get_repo(row)
                    if repo:
                        repos.add(repo)
                    for kind, cids in matches.items():
                        hits[kind].update(cids)
                        counts[f"matched_rows_{kind}"] += bool(cids)
                    if matched:
                        for cid in matched:
                            if repo:
                                coderepos[cid].add(repo)
                        matched_source_records.append({"source": name, "file": path.relative_to(ROOT).as_posix(), "line_index": index,
                            "repo": repo, "url": row.get("url"), "code_ids": sorted(matched),
                            "match_types": sorted(kind for kind, cids in matches.items() if cids)})
                    if original_training and index in roles_by_train_index:
                        for role in roles_by_train_index[index]:
                            expected_url = historical[role]["urls_by_index"].get(index)
                            if expected_url is not None and row.get("url") != expected_url:
                                raise ValueError(f"historical source_index/url mismatch in {role}: {index}")
                            historical[role]["matched_codes"].update(matched)
                    raw_doc = row.get("docstring") or row.get("docstring_summary") or ""
                    token_doc = row.get("docstring_tokens") or []
                    if isinstance(token_doc, list):
                        token_doc = " ".join(token_doc)
                    for doc in (raw_doc, token_doc):
                        q = query_key(str(doc))
                        if q in queries:
                            qhits.add(q)
                        qlexhits.update(queries_lex.get(word_query(doc), set()))
            print(json.dumps({"stage": "scanned", "source": name, "file": path.name, "cumulative_rows": counts["rows"], "matched_codes": len(set().union(*hits.values()))}), flush=True)
        matched = set().union(*hits.values())
        entry = {"name": name, "files": [p.relative_to(ROOT).as_posix() for p in paths], "rows": counts["rows"],
                 "matched_cosqa_codes": {kind: len(value) for kind, value in hits.items()}, "matched_cosqa_codes_any": len(matched),
                 "matched_cosqa_positive_queries_by_code": sum(cid in matched for q,cid in query_to_code.items() if q in positives),
                 "query_overlap_normalized": len(qhits), "query_overlap_lexical": len(qlexhits), "source_repositories": len(repos),
                 "matched_code_ids": sorted(matched), "source_repo_names": sorted(repos), "row_counts": dict(counts)}
        source_details.append(entry)

    scan("local_CodeXGLUE_retriever_training", [ROOT/"data/codexglue/training/train.jsonl"], original_training=True)
    scan("local_AdvTest_validation", [ROOT/"data/codexglue/advtest/valid.jsonl"])
    scan("local_AdvTest_consumed_test", [ROOT/"data/codexglue/advtest/test.jsonl"])
    scan("original_AdvTest_test_code", [ROOT/"imports/dataset/test_code.jsonl"])
    for split in ("train", "valid", "test"):
        paths = sorted((ROOT/f"imports/python/final/jsonl/{split}").glob("*.jsonl.gz"))
        if not paths:
            raise ValueError(f"missing CodeSearchNet split: {split}")
        scan(f"CodeSearchNet_Python_original_{split}", paths)

    for name, relative in (("T5_CODEnn_train", "data/codenn_cqc/train.txt"), ("T5_CODEnn_validation", "data/codenn_cqc/valid.txt")):
        path = ROOT/relative
        input_files.append({"path": relative, "sha256": file_sha(path)})
        qhits, lexhits, n = set(), set(), 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                n += 1
                q = query_key(line)
                if q in queries:
                    qhits.add(q)
                lexhits.update(queries_lex.get(word_query(line), set()))
        source_details.append({"name": name, "rows": n, "query_overlap_normalized": len(qhits), "query_overlap_lexical": len(lexhits)})
        print(json.dumps({"stage": "scanned", "source": name, "rows": n, "query_overlap": len(qhits), "query_lexical_overlap": len(lexhits)}), flush=True)

    for name, item in historical.items():
        matched = item.pop("matched_codes")
        repos = item.pop("repos")
        item.pop("urls_by_index")
        item.update(matched_cosqa_codes=len(matched), matched_cosqa_code_ids=sorted(matched),
                    positive_queries_with_matching_target_code=sum(cid in matched for q,cid in query_to_code.items() if q in positives),
                    cosqa_codes_with_any_recovered_repository_overlap=sum(bool(coderepos[cid]&repos) for cid in code_rows),
                    cosqa_positive_queries_with_any_recovered_repository_overlap=sum(bool(coderepos[cid]&repos) for q,cid in query_to_code.items() if q in positives),
                    reference_repository_count=len(repos))

    # Link identical code and every recovered repository before suggesting group splits.
    parent = {}
    def find(node):
        parent.setdefault(node,node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    def union(a,b):
        parent[find(a)] = find(b)
    for cid in code_rows:
        find("code:"+cid)
        for repo in coderepos[cid]:
            union("code:"+cid, "repo:"+repo)
    groups = collections.defaultdict(list)
    positive_groups = collections.Counter()
    for q,cid in query_to_code.items():
        group = find("code:"+cid)
        groups[group].append(q)
        if q in positives:
            positive_groups[group] += 1
    group_ids = {node:sha_bytes(node)[:20] for node in groups}
    group_report = {"definition": "connected components of exact/normalized matching CoSQA code and all recovered source repositories; no role assignment",
        "all_query_groups": len(groups), "positive_query_groups": len(positive_groups),
        "largest_group_all_queries": max(map(len,groups.values())), "largest_group_positive_queries": max(positive_groups.values()),
        "codes_with_recovered_repository": sum(bool(coderepos[cid]) for cid in code_rows),
        "codes_with_multiple_possible_repositories": sum(len(coderepos[cid])>1 for cid in code_rows),
        "total_recovered_repositories": len(set().union(*coderepos.values())) if coderepos else 0,
        "top_positive_group_sizes": sorted(positive_groups.values(),reverse=True)[:20]}
    group_path = OUT/"QUERY_GROUP_INVENTORY.jsonl"
    with group_path.open("w",encoding="utf-8") as handle:
        for row in rows:
            q=query_key(row["doc"]);cid=query_to_code[q]
            handle.write(json.dumps({"idx":row["idx"],"query_sha256":sha_bytes(q),"code_tokens_sha256":cid,
                "known_positive":row["label"]==1,"group_id":group_ids[find("code:"+cid)],"source_repositories":sorted(coderepos[cid])},ensure_ascii=False)+"\n")
    with (OUT/"MATCHED_SOURCE_RECORDS.jsonl").open("w",encoding="utf-8") as handle:
        for item in matched_source_records:
            handle.write(json.dumps(item,ensure_ascii=False)+"\n")
    result={"scope":"source/annotation metadata audit; no method outputs or retrieval metrics computed", "elapsed_seconds":time.monotonic()-started,
        "cosqa_unique_codes":len(code_rows),"cosqa_queries":len(rows),"cosqa_known_positive_queries":len(positives),
        "fingerprint_contract":{"raw":"SHA256 of unchanged code string", "whitespace":"SHA256 of code with whitespace collapsed, preserving case", "code_tokens":"SHA256 of existing source code_tokens/function_tokens joined by one space, preserving case", "query_normalized":"NFKC+casefold+whitespace", "query_lexical":"casefold+Unicode word tokens"},
        "sources":source_details,"historical_references":historical,"grouping":group_report,"inputs":input_files,
        "limits":["No semantic clone exclusion is established by exact/token fingerprints.","Repository mappings recovered from code matches can have multiple origins.","CoSQA query-code labels remain sparse; code overlap is not evidence of query-label training exposure."]}
    (OUT/"OVERLAP_AUDIT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    summary={"sources":[{k:v for k,v in item.items() if k not in {"matched_code_ids","source_repo_names","files","row_counts"}} for item in source_details],
             "historical_references":{key:{k:v for k,v in item.items() if k!="matched_cosqa_code_ids"} for key,item in historical.items()},"grouping":group_report,"elapsed_seconds":result["elapsed_seconds"]}
    (OUT/"OVERLAP_SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
