"""Verify the authors' retrieval data and exposure strata, without inference."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from discover_sources import ROOT, OUT, SOURCES
from dataset_statistics import query_key, token_key, sha_bytes


def main():
    base=OUT/"selected_source/coclr/data/search"
    code_map=json.loads((base/"code_idx_map.txt").read_text(encoding="utf-8"))
    all_rows=json.loads((ROOT/"imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-all.json").read_text(encoding="utf-8"))
    all_by_id={row["idx"]:row for row in all_rows}
    all_by_code={row["code"]:row for row in all_rows}
    assert len(code_map)==len(set(code_map.values()))==6267
    assert set(code_map.values())==set(range(6267))
    assert set(code_map)==set(all_by_code)
    official={split:json.loads((base/f"cosqa-retrieval-{split}-500.json").read_text(encoding="utf-8")) for split in ("dev","test")}
    held_ids={r["idx"] for rows in official.values() for r in rows}
    assert len(held_ids)==1000
    official["train_derived_identity_only"]=[row for row in all_rows if row["idx"] not in held_ids]
    assert len(official["train_derived_identity_only"])==19604
    source=json.loads((OUT/"OVERLAP_AUDIT.json").read_text(encoding="utf-8"))
    sources={item["name"]:item for item in source["sources"]}
    train_codes=set(sources["local_CodeXGLUE_retriever_training"]["matched_code_ids"])
    strict_codes=set(source["historical_references"]["strict_4096_original_source"]["matched_cosqa_code_ids"])
    training_repos=set(sources["local_CodeXGLUE_retriever_training"]["source_repo_names"])
    inventory=[json.loads(line) for line in (OUT/"QUERY_GROUP_INVENTORY.jsonl").open(encoding="utf-8")]
    inv_by_hash={row["query_sha256"]:row for row in inventory}
    def detail(rows):
        cids={sha_bytes(token_key(r["code_tokens"])) for r in rows}
        pos=[r for r in rows if r["label"]==1]
        repos=set()
        for row in rows:
            repos.update(inv_by_hash[sha_bytes(query_key(row["doc"]))]["source_repositories"])
        return {"pairs":len(rows),"labels":dict(collections.Counter(str(r["label"]) for r in rows)),
            "unique_target_codes":len(cids),"known_positive_queries":len(pos),
            "code_overlap_with_local_retriever_training":len(cids&train_codes),
            "positive_queries_target_in_local_retriever_training":sum(sha_bytes(token_key(r["code_tokens"])) in train_codes for r in pos),
            "code_overlap_with_strict_4096":len(cids&strict_codes),
            "recovered_repositories":len(repos),"recovered_repository_overlap_with_local_retriever_training":len(repos&training_repos),
            "positive_query_groups":len({inv_by_hash[sha_bytes(query_key(r["doc"]))]["group_id"] for r in pos}),
            "positive_queries_unknown_source_repository":sum(not inv_by_hash[sha_bytes(query_key(r["doc"]))]["source_repositories"] for r in pos)}
    stats={}
    for split,rows in official.items():
        if not split.startswith("train_"):
            for row in rows:
                assert row["label"]==1
                assert code_map[row["code"]]==row["retrieval_idx"]
                original=all_by_id[row["idx"]]
                assert all(row[key]==original[key] for key in original)
        stats[split]=detail(rows)
    overlaps={}
    for a,b in (("dev","test"),("train_derived_identity_only","dev"),("train_derived_identity_only","test")):
        ca={r["code"] for r in official[a]};cb={r["code"] for r in official[b]}
        qa={query_key(r["doc"]) for r in official[a]};qb={query_key(r["doc"]) for r in official[b]}
        ga={inv_by_hash[sha_bytes(query_key(r["doc"]))]["group_id"] for r in official[a]}
        gb={inv_by_hash[sha_bytes(query_key(r["doc"]))]["group_id"] for r in official[b]}
        overlaps[a+"__"+b]={"query_overlap":len(qa&qb),"code_overlap":len(ca&cb),"connected_group_overlap":len(ga&gb)}
    positive_rows=[row for row in all_rows if row["label"]==1]
    strata=collections.Counter()
    for row in positive_rows:
        cid=sha_bytes(token_key(row["code_tokens"]))
        repos=set(inv_by_hash[sha_bytes(query_key(row["doc"]))]["source_repositories"])
        if cid in train_codes:
            strata["target_code_matches_local_finetune_training"]+=1
        elif not repos:
            strata["no_match_but_repository_unknown"]+=1
        elif repos&training_repos:
            strata["target_no_exact_match_but_repository_seen"]+=1
        else:
            strata["target_and_recovered_repository_absent_from_local_finetune_training"]+=1
    assert sum(strata.values())==10020
    # Primary data copies match both the Microsoft distribution and author QA blobs.
    author_tree=json.loads((SOURCES/"coclr_tree.json").read_text(encoding="utf-8"))["tree"]
    author_matches=[]
    for split in ("all","train","dev"):
        p=ROOT/f"imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-{split}.json"
        entry=next(x for x in author_tree if x["path"]==f"data/qa/cosqa-{split}.json")
        data=p.read_bytes();actual=hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
        author_matches.append({"split":split,"git_blob_sha1":actual,"upstream_git_blob_sha1":entry["sha"],"matches":actual==entry["sha"]})
    report={"corpus_documents":len(code_map),"corpus_matches_local_cosqa_code_set":True,"corpus_ids_contiguous":True,
        "official_splits":stats,"split_overlaps":overlaps,"positive_query_local_finetune_exposure_strata":dict(strata),
        "local_QA_copies_match_author_tree":author_matches,
        "official_retrieval_train_note":"row identities/labels derived from official all minus official 500 dev and 500 test; original train JSON bytes not downloaded or claimed verified",
        "qrels_contract":"one human-judged positive per evaluation query; other query-document combinations unjudged, not verified negatives",
        "code_map_sha256":hashlib.sha256((base/"code_idx_map.txt").read_bytes()).hexdigest(),
        "split_file_sha256":{split:hashlib.sha256((base/f"cosqa-retrieval-{split}-500.json").read_bytes()).hexdigest() for split in ("dev","test")}}
    (OUT/"RETRIEVAL_CONTRACT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
