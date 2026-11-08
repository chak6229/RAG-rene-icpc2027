"""Recheck source identities and audit contracts; never evaluate a search method."""
from __future__ import annotations

import collections
import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from discover_sources import ROOT, OUT


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**22), b""):
            h.update(block)
    return h.hexdigest()


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_query(value):
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def main():
    checks = []

    def check(name, condition, **detail):
        checks.append({"name": name, "passed": bool(condition), **detail})
        if not condition:
            raise AssertionError(name)

    snapshots = {row["key"]: row for row in load("REPOSITORY_SNAPSHOTS.json")}
    repo_keys = {row["repository"]: key for key, row in snapshots.items()}
    resolved = []
    for record in load("OFFICIAL_SOURCE_FILES.json"):
        key = repo_keys[record["repository"]]
        entry = next(row for row in snapshots[key]["files"] if row["path"] == record["upstream_path"])
        path = OUT / "selected_source" / key / record["upstream_path"]
        data = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        check("upstream_blob:" + key + "/" + record["upstream_path"], actual == entry["sha"])
        if "content_sha256" in record:
            check("capture_sha256:" + key + "/" + record["upstream_path"], sha256(path) == record["content_sha256"])
        resolved.append({"repository": record["repository"], "commit": snapshots[key]["commit"],
                         "upstream_path": entry["path"], "git_blob_sha1": actual,
                         "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                         "initial_capture_failed": "error" in record,
                         "resolved_capture": "SOURCE_RETRY_dev500_raw.json" if "error" in record else "OFFICIAL_SOURCE_FILES.json"})
    retry = load("SOURCE_RETRY_dev500_raw.json")
    check("retrieval_dev_successful_retry", retry["verified"] and any(row["sha256"] == retry["sha256"] for row in resolved))
    for record in load("LOCAL_UPSTREAM_MATCHES.json"):
        path = ROOT / record["path"]
        data = path.read_bytes()
        check("local_upstream:" + record["path"],
              hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == record["upstream_blob_sha1"]
              and sha256(path) == record["sha256"])
    cuda = load("CUDA_LICENSE_SOURCES.json")
    for record in cuda["files"]:
        path = ROOT / record["path"]
        data = path.read_bytes()
        check("cuda_blob:" + record["upstream_path"],
              hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == record["git_blob_sha1"]
              and sha256(path) == record["sha256"])

    base = ROOT / "imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA"
    rows = json.loads((base / "cosqa-all.json").read_text(encoding="utf-8"))
    by_id = {row["idx"]: row for row in rows}
    check("all_rows_unique_ids_and_queries", len(by_id) == len({normalized_query(r["doc"]) for r in rows}) == 20604)
    labels = collections.Counter(row["label"] for row in rows)
    check("annotation_conservation", labels == {0: 10584, 1: 10020})
    qa_train = json.loads((base / "cosqa-train.json").read_text(encoding="utf-8"))
    qa_dev = json.loads((base / "cosqa-dev.json").read_text(encoding="utf-8"))
    check("qa_all_is_disjoint_train_dev", {r["idx"] for r in qa_train}.isdisjoint({r["idx"] for r in qa_dev})
          and {r["idx"] for r in qa_train + qa_dev} == set(by_id)
          and all(by_id[r["idx"]] == r for r in qa_train + qa_dev))
    corpus = json.loads((OUT / "selected_source/coclr/data/search/code_idx_map.txt").read_text(encoding="utf-8"))
    check("corpus_code_set_and_ids", set(corpus) == {r["code"] for r in rows} and set(corpus.values()) == set(range(6267)))
    official = {}
    for split in ("dev", "test"):
        split_rows = json.loads((OUT / f"selected_source/coclr/data/search/cosqa-retrieval-{split}-500.json").read_text(encoding="utf-8"))
        official[split] = split_rows
        check("official_retrieval_" + split, len(split_rows) == 500
              and all(row["label"] == 1 and corpus[row["code"]] == row["retrieval_idx"]
                      and all(row[k] == by_id[row["idx"]][k] for k in by_id[row["idx"]]) for row in split_rows))

    inventory = [json.loads(line) for line in (OUT / "QUERY_GROUP_INVENTORY.jsonl").open(encoding="utf-8")]
    inv = {row["idx"]: row for row in inventory}
    check("group_inventory_complete", len(inv) == 20604 and set(inv) == set(by_id))
    for row in rows:
        item = inv[row["idx"]]
        check_hash = item["query_sha256"] == digest_text(normalized_query(row["doc"]))
        check_hash &= item["code_tokens_sha256"] == digest_text(" ".join(row["code_tokens"].split()))
        if not check_hash or item["known_positive"] != (row["label"] == 1):
            raise AssertionError("inventory row mismatch:" + row["idx"])
    check("all_inventory_rows_hash_and_label_match", True, rows=len(rows))
    repo_groups, code_groups = collections.defaultdict(set), collections.defaultdict(set)
    for item in inventory:
        code_groups[item["code_tokens_sha256"]].add(item["group_id"])
        for repo in item["source_repositories"]:
            repo_groups[repo].add(item["group_id"])
    positive = [item for item in inventory if item["known_positive"]]
    check("connected_groups_preserve_known_code_and_repo", all(len(g) == 1 for g in code_groups.values())
          and all(len(g) == 1 for g in repo_groups.values()))
    check("group_counts", len({x["group_id"] for x in inventory}) == 3156
          and len({x["group_id"] for x in positive}) == 2475 and len(repo_groups) == 2933)
    unknown_codes = {x["code_tokens_sha256"] for x in inventory if not x["source_repositories"]}
    check("unresolved_source_counts", len(unknown_codes) == 223 and sum(not x["source_repositories"] for x in positive) == 252)

    overlap = load("OVERLAP_AUDIT.json")
    train_source = next(x for x in overlap["sources"] if x["name"] == "local_CodeXGLUE_retriever_training")
    matched_codes = set(train_source["matched_code_ids"])
    train_repos = set(train_source["source_repo_names"])
    strata = collections.Counter()
    for item in positive:
        if item["code_tokens_sha256"] in matched_codes:
            name = "target_code_matches_local_finetune_training"
        elif not item["source_repositories"]:
            name = "no_match_but_repository_unknown"
        elif set(item["source_repositories"]) & train_repos:
            name = "target_no_exact_match_but_repository_seen"
        else:
            name = "target_and_recovered_repository_absent_from_local_finetune_training"
        strata[name] += 1
    check("local_exposure_strata_reconstructed", dict(strata) == load("RETRIEVAL_CONTRACT.json")["positive_query_local_finetune_exposure_strata"])
    test_groups = {inv[row["idx"]]["group_id"] for row in official["test"]}
    development_eligible = [item for item in positive if item["group_id"] not in test_groups]
    group_feasibility = {
        "status": "availability audit only, no role assignment or power claim",
        "scenario": "reserve official test500 and exclude all its recovered code/repository connected groups from fitting",
        "official_test_queries": 500, "official_test_positive_groups": len(test_groups),
        "remaining_positive_queries": len(development_eligible),
        "remaining_positive_groups": len({x["group_id"] for x in development_eligible}),
        "remaining_queries_with_known_source_repository": sum(bool(x["source_repositories"]) for x in development_eligible),
        "remaining_groups_with_known_source_repository": len({x["group_id"] for x in development_eligible if x["source_repositories"]}),
        "queries_excluded_outside_official_test_for_group_overlap": 10020 - len(development_eligible) - 500,
        "qualification": "unknown-repository codes only form code groups; true repository disjointness is not established for them",
        "next": "prospectively freeze development/confirmation roster after upfront clustered power and event assessment; no method effects have been computed",
    }
    (OUT / "ROLE_FEASIBILITY.json").write_text(json.dumps(group_feasibility, ensure_ascii=False, indent=2), encoding="utf-8")

    inputs = load("INPUT_CONTRACT.json")
    check("input_schema_contract", inputs["schema_contract"]["query_field"] == "doc"
          and inputs["schema_contract"]["unique_encoder_code_texts"] == 6267)
    for name, stats in inputs["t5"].items():
        check("input_sentinel_contract:" + name, stats["paper_all_boundaries"]["count"] == 156603
              and stats["paper_all_boundaries"]["max"] == 36
              and stats["untruncated_sentinel_count_not_one"] == 0
              and all(value == 0 for value in stats["sentinel_count_not_one_after_truncation"].values()))
    for item in inputs["tokenizer_files"]:
        check("tokenizer_sha:" + item["path"], sha256(ROOT / item["path"]) == item["sha256"])
    protected = load("LOCAL_CHECKPOINTS.json")["protected_snapshot"]
    for path, expected in protected.items():
        check("protected_pointer:" + path, sha256(ROOT / path) == expected)
    # Record the public implementation's effective seed call, independently of
    # the unused seed=101 configuration field. This is source evidence, not a run.
    source = (OUT / "selected_source/ssqr/Pretraining/pretrain.py").read_text(encoding="utf-8")
    check("public_t5_seed_discrepancy_present", "seed=101" in source and "set_seed(42)" in source)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
              "scope": "source/data/schema/recorded tokenization/pointer integrity; no method inference or retrieval-effect verification",
              "passed_checks": len(checks), "checks": checks, "resolved_official_source_files": resolved,
              "protected_files_unchanged": len(protected),
              "large_input_hashes": "recorded during original full streaming overlap audit, not redundantly rescanned here",
              "weights": "actual hashes/step metadata captured in LOCAL_CHECKPOINTS.json; no checkpoint replaced"}
    (OUT / "VERIFICATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": len(checks), "verified_official_files": len(resolved),
                      "protected_files_unchanged": len(protected), "role_feasibility": group_feasibility}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
