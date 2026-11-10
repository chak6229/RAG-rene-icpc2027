"""Freeze group roles and settings without computing any CoSQA method effects."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from .common import ADAPTER, CODEBERT, INVENTORY, OUT, ROOT, T5, now, read, readl, sha, write, writel


def order(value, salt):
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()


def main():
    if (OUT / "PROTOCOL.json").exists():
        raise ValueError("protocol already frozen; do not replace it")
    commands = [[sys.executable, "-X", "utf8", "-m", "unittest", "experiments.rene_cosqa_pilot_20261019.test_contract", "-v"],
                [sys.executable, "-X", "utf8", "experiments/rene_ssqr_alignment_20261019/test_rne_eval.py"]]
    tests = []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        tests.append({"command": command, "exit_code": result.returncode, "output": result.stdout + result.stderr})
        if result.returncode:
            write(OUT / "TEST_FAILURE.json", tests)
            raise ValueError("contract checks failed before protocol freeze")
    write(OUT / "TEST_RESULTS.json", {"passed": True, "strict_regressions": 27, "legacy_valid_case_tests": 4, "runs": tests}, immutable=True)

    queries = readl(ADAPTER / "public/all_positive_queries.jsonl")
    query_map = {row["query_id"]: row for row in queries}
    inventory = {row["idx"]: row for row in readl(INVENTORY)}
    groups = {query: inventory[query]["group_id"] for query in query_map}
    by_group = defaultdict(list)
    for query, group in groups.items():
        by_group[group].append(query)
    official_test = {row["query_id"] for row in readl(ADAPTER / "public/official_test_queries.jsonl")}
    test_groups = {groups[query] for query in official_test}
    eligible_groups = sorted(set(by_group) - test_groups, key=lambda value: order(value, "rene-cosqa-development-20261019"))
    development_groups = set(eligible_groups[:512])
    confirmation_groups = set(by_group) - development_groups
    assert len(development_groups) == 512 and not development_groups & test_groups
    development_ids = sorted(query for query in query_map if groups[query] in development_groups)
    confirmation_ids = sorted(set(query_map) - set(development_ids))
    # Whole connected groups, selected by a prespecified hash order; never split a pilot group.
    pilot_groups, pilot_ids = [], []
    for group in eligible_groups[:512]:
        group_queries = sorted(by_group[group], key=lambda query: order(query, "rene-cosqa-pilot-query-20261019"))
        if len(pilot_ids) + len(group_queries) > 256:
            break
        pilot_groups.append(group)
        pilot_ids.extend(group_queries)
    assert 128 <= len(pilot_ids) <= 256
    assert set(pilot_groups) <= development_groups
    assert not set(pilot_ids) & set(confirmation_ids)
    pilot_queries = [query_map[query] for query in pilot_ids]
    all_roles = {query: "development" if query in set(development_ids) else "confirmation_pool" for query in query_map}
    role_manifest = {
        "status": "FROZEN", "created_utc": now(), "group_definition": "audited code/recovered-repository connected components",
        "development_query_ids": development_ids, "development_group_ids": sorted(development_groups),
        "confirmation_pool_query_ids": confirmation_ids, "confirmation_pool_group_ids": sorted(confirmation_groups),
        "confirmation_sampling_group_order": sorted(confirmation_groups, key=lambda value: order(value, "rene-cosqa-confirmation-20261019")),
        "official_test_query_ids": sorted(official_test), "official_test_group_ids": sorted(test_groups),
        "pilot_query_ids": pilot_ids, "pilot_group_ids": pilot_groups, "roles_by_query": all_roles,
        "confirmation_not_scheduled": True, "group_partition_without_predictions": True,
        "unknown_repository_identities_preserved": True,
        "selection_rule": "first512 non-test groups by SHA256 salt; pilot whole-group prefix fitting <=256 queries; stop before first overshoot",
    }
    write(OUT / "ROLE_MANIFEST.json", role_manifest, immutable=True)
    writel(OUT / "data/queries.jsonl", pilot_queries)
    write(OUT / "data/groups.json", {query: groups[query] for query in pilot_ids}, immutable=True)
    write(OUT / "data/PILOT_ROLE.json", {"status": "FROZEN", "role": "development", "query_ids": pilot_ids,
                                        "group_ids": pilot_groups, "roles_manifest_sha256": sha(OUT / "ROLE_MANIFEST.json")}, immutable=True)
    # Partitioning labels is trusted data preparation, separate from opening method effects.
    # Annotation provenance has already been audited; do not claim these labels were never read.
    qrels = readl(ADAPTER / "private/all_positive_qrels.jsonl")
    pilot_set = set(pilot_ids)
    pilot_qrels = [row for row in qrels if row["query_id"] in pilot_set]
    assert {row["query_id"] for row in pilot_qrels} == pilot_set
    writel(OUT / "data/private/pilot_qrels.jsonl", pilot_qrels)
    write(OUT / "DATA_ACCESS_LEDGER.json", {"created_utc": now(), "purpose": "role allocation and label partition, no model results",
                                           "annotations_previously_read_for_provenance_audit": True,
                                           "private_export_partitioned_this_step": True,
                                           "inference_reads_private": False, "confirmation_effects_opened": False,
                                           "pilot_queries_permanently_development": len(pilot_ids)}, immutable=True)

    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    source_paths += [ROOT / "experiments/rene_ssqr_alignment_20261019/rne_eval.py",
                     ROOT / "experiments/rene_ssqr_alignment_20261019/t5.py",
                     ROOT / "experiments/rene_ssqr_alignment_20261019/retriever.py",
                     ROOT / "experiments/rene_ssqr_alignment_20261019/pipeline.py",
                     ROOT / "experiments/rene_ssqr_alignment_20261019/common.py"]
    data_paths = [ADAPTER / "manifest.json", ADAPTER / "public/corpus.jsonl", ADAPTER / "public/all_positive_queries.jsonl",
                  ADAPTER / "public/official_test_queries.jsonl", INVENTORY, T5 / "model.safetensors",
                  T5 / "config.json", ROOT / "models/t5-base/spiece.model", CODEBERT / "model.safetensors",
                  CODEBERT / "config.json", CODEBERT / "training_complete.json", OUT / "ROLE_MANIFEST.json",
                  OUT / "data/queries.jsonl", OUT / "data/groups.json", OUT / "data/PILOT_ROLE.json",
                  OUT / "data/private/pilot_qrels.jsonl", OUT / "TEST_RESULTS.json"]
    protected_paths = []  # Author-workspace registries are outside this artifact.
    def record(path):
        return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
    protocol = {
        "schema": "rene-cosqa-source-reconstruction-development-v1", "status": "FROZEN", "created_utc": now(),
        "role": "development", "query_ids": pilot_ids, "queries": len(pilot_ids), "groups": len(pilot_groups),
        "role_manifest_sha256": sha(OUT / "ROLE_MANIFEST.json"), "candidate_k": 3,
        "metric_k": 100, "ranking_depth": 100, "retrievers": ["CodeBERT", "BM25"],
        "generation": {"seed": 101, "do_sample": False, "max_new_tokens": 19, "temperature": 1.0,
                       "top_k": 0, "top_p": 1.0, "position_policy": "source_1_to_n", "batch_size": 8},
        "reconstruction": {
            "target": "source-guided reconstruction of SSQR position-scoring transfer, not exact original checkpoint replication",
            "primary_basis": "published main JSON path positions1..n, explicit greedy reconstruction of unspecified generate defaults",
            "paper_differences": "paper describes positions0..n, sampled m10; original post-training generation config and weights unavailable",
            "max_new_tokens_rationale": "explicit 19 new tokens corresponding to modern default total length20 incl decoder start; not recovered original decoding",
            "t5_checkpoint": "local best92000 from completed 93750-step CODEnn run, train seed101; author code actual set_seed42",
            "codebert": "new completed2epoch pooler/raw-dot/full batch CE; source linear warmup/decay; source valid only checkpoint selection",
            "prob": "descending teacher-forced mean log probability of wrapped label including sentinels/EOS; declared operationalization",
            "entr": "ascending raw full-model-vocabulary mean entropy; sentinels/EOS included, padding excluded",
            "same_pool": True, "quality_filter": "none", "deduplicate_positions": False,
            "short_query_rule": "min(3, number of available positions), no identity padding",
        },
        "rand_seeds": list(range(2026101900, 2026101920)),
        "rand_estimand": "per-query average of best3 RR over20 frozen uniform position permutations, no new T5 sampling",
        "bm25": {"parameters": {"k1": 1.2, "b": .75}, "idf": "log(1+(N-df+.5)/(df+.5))",
                 "tokenization": "ASCII lowercase alphanumeric words starting with a letter, camel/snake split",
                 "query_tf": "unique terms", "document_truncation": "none"},
        "dense": {"representation": "pooler_output", "normalization": "none", "similarity": "raw dot",
                  "precision": "float32, TF32 disabled", "max_tokens": 256, "batch_size": 16},
        "ties": "score_desc_doc_id_asc", "missing_input_policy": "reject; no implicit deletion or fallback",
        "main_contrasts": [f"{retriever}:ENTR_minus_{control}_best3" for retriever in ("CodeBERT", "BM25") for control in ("RAND", "PROB")],
        "multiplicity": "four RQ1 contrasts across two retrievers; Holm cluster-t p-values; Bonferroni98.75% cluster percentile intervals",
        "bootstrap_repetitions": 10000, "statistics_seed": 20261019, "planning_effect": .005,
        "inference": "pilot is descriptive development only; independent connected groups assumed; no confirmatory equivalence declaration",
        "power_plan": {"alpha_family": .05, "family_size": 4, "power_targets": [.8, .9],
                       "effect_targets": [.0025, .005, .01],
                       "method": "group-weighted delta cluster-sandwich variance; normal planning with conservative95% bootstrap variance",
                       "minimum_changed_groups_for_variance_plan": 20, "bootstrap_variance_repetitions": 2000,
                       "confirmation_group_selection": "frozen SHA256 order of reserved groups, sample size decided before any confirmation method effects",
                       "confirmation_currently_authorized": False},
        "deployment_policy_promoted": False, "risk_certified": False,
        "bound_files": [record(path) for path in source_paths + data_paths],
        "protected_files": [record(path) for path in protected_paths],
    }
    write(OUT / "PROTOCOL.json", protocol, immutable=True)
    write(OUT / "FREEZE_RECEIPT.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                                        "method_effects_computed": False, "confirmation_group_count": len(confirmation_groups),
                                        "pilot_query_count": len(pilot_ids), "pilot_group_count": len(pilot_groups)}, immutable=True)
    print(json.dumps({"status": "FROZEN", "pilot_queries": len(pilot_ids), "pilot_groups": len(pilot_groups),
                      "development_queries": len(development_ids), "development_groups": len(development_groups),
                      "confirmation_pool_queries": len(confirmation_ids), "confirmation_pool_groups": len(confirmation_groups),
                      "official_test_groups_reserved": len(test_groups), "protocol_sha256": sha(OUT / "PROTOCOL.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
