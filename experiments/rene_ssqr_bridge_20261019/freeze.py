"""Construct source identities and freeze finite diagnostics before new effects."""
from __future__ import annotations

import importlib.metadata
import platform
from collections import defaultdict

from .common import (
    ROOT, PACKAGE, OUT, DEVELOPMENT, ADAPTER, INVENTORY, SOURCE, T5, CODEBERT,
    PROTECTED, PROFILES, now, read, readl, write, writel, record, priority, require,
)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def choose_one_per_group(ids, groups, *, namespace, limit=None):
    grouped = defaultdict(list)
    for query_id in ids:
        grouped[groups[query_id]].append(query_id)
    selected_groups = sorted(grouped, key=lambda value: priority(namespace + "/group", value))
    if limit is not None:
        require(len(selected_groups) >= limit, "too few independent groups for frozen sensitivity sample")
        selected_groups = selected_groups[:limit]
    return [min(grouped[g], key=lambda value: priority(namespace + "/query", value)) for g in selected_groups]


def prepare():
    if (OUT / "PROTOCOL.json").exists():
        from .common import verify_frozen
        return verify_frozen(private=True)
    require(not (OUT / "metrics").exists(), "new effects already exist before design freeze")
    require(read(DEVELOPMENT / "FINAL_ACCEPTANCE.json")["passed"], "development parent did not pass")
    source = readl(SOURCE)
    require(len(source) == 19210 and len({r["url"] for r in source}) == 19210, "wrong source test corpus")
    official_urls = (ROOT / "imports/dataset/test.txt").read_text(encoding="utf-8-sig").splitlines()
    require([r["url"] for r in source] == official_urls, "canonical test rows differ from author split order")
    author_rows = {r["url"]: r for r in readl(ROOT / "imports/dataset/test_code.jsonl")}
    for row in source:
        author = author_rows[row["url"]]
        require(row["function_tokens"] == author["function_tokens"] and
                row["docstring_tokens"] == author["docstring_tokens"], "source texts differ from author data")
        require(isinstance(row["nwo"], str) and row["nwo"], "source repository is missing")
    source_index = {f'advtest-{row["idx"]}': index for index, row in enumerate(source)}
    source_ids = sorted(source_index, key=lambda value: priority("main-source", value))[:2048]
    source_groups = {qid: source[source_index[qid]]["nwo"].casefold().strip() for qid in source_ids}
    source_sensitivity = choose_one_per_group(source_ids, source_groups, namespace="source-sensitivity", limit=512)
    cosqa_queries = readl(DEVELOPMENT / "data/queries.jsonl")
    cosqa_ids = [row["query_id"] for row in cosqa_queries]
    cosqa_groups = read(DEVELOPMENT / "data/groups.json")
    roles = read(DEVELOPMENT / "ROLE_MANIFEST.json")
    require(len(cosqa_ids) == 1490 and set(cosqa_ids) == set(roles["development_query_ids"]), "wrong CoSQA development role")
    require(not set(cosqa_ids) & set(roles["confirmation_pool_query_ids"]), "confirmation query entered sensitivity input")
    cosqa_sensitivity = choose_one_per_group(cosqa_ids, cosqa_groups, namespace="cosqa-sensitivity")
    require(len(cosqa_sensitivity) == 512, "unexpected CoSQA group count")
    inventory = {row["idx"]: row for row in readl(INVENTORY)}
    uf = UnionFind()
    for qid in source_ids:
        uf.union("source:" + qid, "repo:" + source_groups[qid])
    for qid in cosqa_ids:
        uf.union("cosqa:" + qid, "cosqa-group:" + cosqa_groups[qid])
        for repository in inventory[qid]["source_repositories"]:
            uf.union("cosqa:" + qid, "repo:" + repository.casefold().strip())
    joint = {domain: {qid: uf.find(domain + ":" + qid) for qid in ids}
             for domain, ids in (("source", source_ids), ("cosqa", cosqa_ids))}
    source_queries = [{"query_id": qid, "query": " ".join(source[source_index[qid]]["docstring_tokens"])} for qid in source_ids]
    require(all(q["query"].strip() and "<extra_id_" not in q["query"] for q in source_queries), "invalid source query")
    writel(OUT / "data/source/queries.jsonl", source_queries)
    writel(OUT / "data/source/corpus.jsonl", [{"doc_id": index, "code_text": " ".join(row["function_tokens"])}
                                            for index, row in enumerate(source)])
    writel(OUT / "data/private/source_qrels.jsonl", [{"query_id": qid, "doc_id": source_index[qid], "relevance": 1}
                                                  for qid in source_ids])
    writel(OUT / "data/private/source_lineage.jsonl", [{"query_id": qid, "url": source[source_index[qid]]["url"],
                                                      "source_row": source_index[qid], "nwo": source_groups[qid]} for qid in source_ids])
    write(OUT / "data/joint_groups.json", joint)
    samples = {
        "source": {"main_ids": source_ids, "sensitivity_ids": source_sensitivity, "groups": source_groups,
                   "available_queries": 19210, "corpus_documents": 19210},
        "cosqa": {"main_ids": cosqa_ids, "sensitivity_ids": cosqa_sensitivity, "groups": cosqa_groups,
                  "available_queries": 1490, "corpus_documents": 6267},
    }
    dependencies = [ROOT / "experiments/rene_ssqr_alignment_20261019" / name for name in
                    ("common.py", "t5.py", "pipeline.py", "data.py", "retriever.py")]
    dependencies += [ROOT / "experiments/rene_cosqa_pilot_20261019" / name for name in
                     ("common.py", "selection.py", "ranking.py")]
    paths = list(PACKAGE.glob("*.py")) + dependencies + [SOURCE, ROOT / "imports/dataset/test.txt",
             ROOT / "imports/dataset/test_code.jsonl", INVENTORY,
             DEVELOPMENT / "data/queries.jsonl", DEVELOPMENT / "data/groups.json", DEVELOPMENT / "ROLE_MANIFEST.json",
             DEVELOPMENT / "data/private/development_qrels.jsonl", DEVELOPMENT / "candidates.jsonl",
             DEVELOPMENT / "rankings/CodeBERT.jsonl", DEVELOPMENT / "rankings/BM25.jsonl",
             DEVELOPMENT / "FINAL_ACCEPTANCE.json", ADAPTER / "public/corpus.jsonl",
             ROOT / "results/rene_cosqa_confirmation_20261019/PROTOCOL.json",
             ROOT / "results/rene_cosqa_confirmation_20261019/PRIMARY_RESULT.json",
             ROOT / "results/rene_cosqa_confirmation_20261019/FINAL_ACCEPTANCE.json",
             ROOT / "results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy",
             ROOT / "models/t5-base/spiece.model"]
    for folder in (T5, CODEBERT):
        paths += [p for p in folder.iterdir() if p.is_file() and p.suffix != ".pt"]
    paths += list((OUT / "data").rglob("*.json*"))
    protocol = {
        "schema_version": 1, "status": "FROZEN", "created_utc": now(),
        "role": "post_confirmation_supplementary_diagnostic",
        "purpose": "separate reconstruction sensitivity from dataset dependence; strengthen external-replication fidelity evidence",
        "prior_confirmation": {"unchanged": True, "all_H_consumed": True,
                               "not_new_confirmation": True, "sensitivity_uses_H": False},
        "source_status": "historically exposed CodeXGLUE AdvTest; supplementary diagnostic only",
        "samples": samples, "profiles": PROFILES,
        "primary_bridge_profile": "greedy20_source",
        "sensitivity_population": "one deterministically hash-selected query per group; 512 groups per dataset; same queries in all profiles",
        "sampling_seed": 101, "sampling_rng": "SHA256(seed,query_id,position), one sequence per position; shared positions cached once across position policies",
        "sampling_parameters": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "num_beams": 1},
        "length_interpretation": "m=10 is operationalized as total decoder length including start=10, max_new_tokens=9; original exact counting was not recoverable",
        "generation_precision": "float32; TF32 disabled; eval/dropout off", "baseline_generation_batch_size": 8,
        "sensitivity_greedy_batch_size": 8, "sensitivity_sampling_batch_size": 1,
        "scoring": "full raw-vocabulary teacher forcing; primary all nonpad sentinels+EOS; secondary lexical span mask (empty placed last)",
        "candidate_k": 3, "rand_seeds": list(range(2026101900, 2026101920)), "ranking_depth": 100,
        "tie_rule": "score descending, corpus doc_id ascending", "no_original_fallback_in_best3": True,
        "retrievers": ["CodeBERT", "BM25"],
        "bm25": {"k1": 1.2, "b": .75, "implementation": "local lexical control, not claimed Lucene parity"},
        "inference": {"estimand": "query-weighted ENTR best3 minus mean of 20 RAND best3 repetitions; PROB control",
                      "groups": "source repositories / CoSQA connected groups; union across datasets through recovered repository identities",
                      "intervals": "cluster t95 and 10000 shared-connected-group bootstrap; descriptive, no new confirmatory tests",
                      "domain_difference": "direct source effect minus CoSQA effect; never significance-vs-nonsignificance inference",
                      "sensitivity_contrasts": ["greedy20_all - greedy20_source", "greedy10_source - greedy20_source",
                                                "sample10_source - greedy10_source", "sample10_all - greedy10_all",
                                                "sample10_all - greedy20_source", "lexical_entropy - full_entropy"],
                      "no_outcome_based_profile_promotion": True},
        "models": {"T5": str(T5.relative_to(ROOT)), "CodeBERT": str(CODEBERT.relative_to(ROOT)),
                   "author_checkpoint_available": False, "retraining_authorized_in_this_run": False},
        "bound_files": [record(p) for p in sorted(set(paths))],
        "protected_files": [record(ROOT / path) for path in PROTECTED],
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                        for name in ("torch", "transformers", "numpy", "scipy", "sentencepiece")}},
    }
    write(OUT / "PROTOCOL.json", protocol)
    print({"frozen": True, "source_queries": len(source_ids), "source_groups": len(set(source_groups.values())),
           "cosqa_queries": len(cosqa_ids), "sensitivity_each": 512,
           "joint_groups": len(set(joint["source"].values()) | set(joint["cosqa"].values()))}, flush=True)
    return protocol


if __name__ == "__main__":
    prepare()
