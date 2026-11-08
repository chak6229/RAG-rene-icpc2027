"""Materialize the filled development manifest from frozen artifacts."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version as package_version, PackageNotFoundError
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "ResearchVault/Handoffs"
OUT = HANDOFF / "RENE-统一评测manifest模板.json"
DATA = ROOT / "results/uq_pilot_20261014/action_witness_uq/data"
EVIDENCE = ROOT / "results/strong_flow_uq_20261014/evidence"
FLOW = ROOT / "results/strong_flow_uq_20261014"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def combined(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=str):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(bytes.fromhex(sha(path)))
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    protocol = json.loads((FLOW / "PROTOCOL.json").read_text(encoding="utf8"))
    roles = protocol["roles"][0]
    counts = {name: len(values) for name, values in roles.items()}
    source_files = [
        ROOT / "experiments/strong_flow_uq_20261014/run.py",
        ROOT / "experiments/strong_flow_uq_20261014/flow.py",
        ROOT / "experiments/strong_flow_uq_20261014/build_evidence.py",
        ROOT / "experiments/gain_model_registry_20261014/registry.py",
        ROOT / "experiments/verification_policy_registry_20261015/registry.py",
        ROOT / "experiments/risk_module_registry_20261016/registry.py",
        ROOT / "build_rewrite_utility_dataset.py",
        ROOT / "build_enhanced_retrieval_feedback_dataset.py",
    ]
    t5 = ROOT / "models/t5-cqc-codenn-1m/best"
    retriever = ROOT / "models/codebert-codexglue-2epoch/checkpoint-3000"
    tokenizer_files = [p for p in t5.iterdir() if p.is_file() and p.suffix in (".json", ".txt", ".model")]
    try:
        torch_version = package_version("torch")
    except PackageNotFoundError:
        torch_version = "not_installed"
    try:
        transformers_version = package_version("transformers")
    except PackageNotFoundError:
        transformers_version = "not_installed"
    manifest = {
        "schema": "rene-evaluation-manifest",
        "schema_version": "1.0",
        "status": "FROZEN_DEVELOPMENT_REPRODUCTION",
        "created_for": "ICPC 2027 RENE controlled development reproduction",
        "claim_scope": "Development-only replay; no independent confirmation or risk certificate.",
        "protocol": {
            "protocol_id": "rene-dev-strong84-20261019",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": "strong_flow_uq_20261014-replay",
            "git_revision": git_revision(),
            "source_tree_hash": combined(source_files),
            "random_seeds": {"python": 261014, "numpy": 261014, "torch": 261014,
                             "t5_generation": [101, 202, 303, 404, 505, 606, 707, 808],
                             "data_order": "frozen action_items.pkl order"},
            "test_policy": "historical AdvTest test is consumed and forbidden",
        },
        "data": {
            "dataset_name": "VAC-MQR action-witness development pool",
            "dataset_root": rel(DATA),
            "splits": {
                "train": {"path": rel(DATA / "action_items.pkl"), "count": counts["selector_fit"], "sha256": sha(DATA / "action_items.pkl")},
                "valid": {"path": rel(DATA / "action_items.pkl"), "count": sum(counts.values()), "sha256": sha(DATA / "action_items.pkl")},
                "confirmation": {"path": "NOT_APPLICABLE_DEVELOPMENT_ONLY", "count": 0, "sha256": None},
            },
            "repository_cluster_key": "repo; one query per repository",
            "url_overlap": {"train_valid": 0, "train_confirmation": "not_applicable", "valid_confirmation": "not_applicable"},
            "test_exposure": {"used_before_this_run": True, "allowed_in_this_run": False,
                              "exposure_log": "ResearchVault/Handoffs/RENE-评测来源与test暴露审计-20261018.md"},
            "code_representation": "code_tokens/function_tokens only; raw function/docstring forbidden",
            "query_representation": "frozen query text from action_items.pkl",
        },
        "candidate_generation": {
            "generator_name": "CODEnn CQC T5 with multi-seed span insertion",
            "generator_checkpoint": rel(t5),
            "checkpoint_sha256": sha(t5 / "model.safetensors"),
            "tokenizer_manifest_sha256": combined(tokenizer_files),
            "script_path": "run_t5_seed_uq_pilot.py",
            "script_sha256": sha(ROOT / "run_t5_seed_uq_pilot.py"),
            "candidate_policy": "historical frozen multi-seed candidate pool; no label-based candidate filtering",
            "candidate_count_per_query": "variable union, max 16",
            "position_policy": "all whitespace token boundaries",
            "sampling": {"enabled": True, "seed": [101, 202, 303, 404, 505, 606, 707, 808], "temperature": 1.0,
                          "top_p": 1.0, "top_k": 50, "num_return_sequences": 1},
            "max_input_tokens": 512, "max_span_tokens": 12, "quality_filter_before_topk": True,
            "candidate_cache": {"path": rel(DATA / "action_items.pkl"), "sha256": sha(DATA / "action_items.pkl"), "row_count": 512},
        },
        "retrieval": {
            "retriever_name": "CodeBERT CodeXGLUE checkpoint-3000",
            "checkpoint_path": rel(retriever),
            "weights_sha256": sha(retriever / "model.safetensors"),
            "config_sha256": sha(retriever / "config.json"),
            "tokenizer_files_sha256": {rel(p): sha(p) for p in sorted(retriever.iterdir()) if p.is_file() and p.suffix in (".json", ".txt", ".model")},
            "vector_extraction": {"layer": "last_hidden_state", "pooling": "CLS index 0", "normalization": "L2", "precision": "float32 output from BF16 autocast when supported", "autocast_enabled": True, "autocast_dtype": "bfloat16", "max_length": 256, "batch_size": 48},
            "corpus": {"path": rel(DATA / "action_code_corpus.pkl"), "count": 4096, "sha256": sha(DATA / "action_code_corpus.pkl")},
            "similarity": "cosine dot product after L2 normalization",
            "top_k": 100,
            "tie_breaking": "score descending, document ID ascending",
            "retrieval_cache": {"path": rel(DATA / "ce_large_dense_4096_scores_action.pkl"), "sha256": sha(DATA / "ce_large_dense_4096_scores_action.pkl")},
        },
        "scoring": {"rr_definition": "1 / (1 + count(scores > relevant_score)) if rank <= 100 else 0", "strict_greater_than": True,
                    "k": 100, "relevant_index_source": "action_items.pkl query_index", "fusion_definition": "equal-weight original/candidate RRF, offset 60",
                    "candidate_pool_sha256": sha(DATA / "action_items.pkl"), "label_definition": "candidate RR minus original RR",
                    "missing_candidate_policy": "no missing slots in this action pool; historical missing slots remain unavailable"},
        "roles_and_selection": {"utility_train": {"count": counts["selector_fit"], "sha256": sha(DATA / "action_items.pkl")},
                                "selection_valid": {"count": counts["selection_valid"], "sha256": sha(DATA / "action_items.pkl")},
                                "risk_calibration": {"count": counts["risk_calibration"], "sha256": sha(DATA / "action_items.pkl")},
                                "confirmation": {"count": 0, "sha256": None, "status": "not_applicable_development_replay"},
                                "thresholds_frozen_before_confirmation": False, "model_selection_inputs": ["frozen development protocol only"],
                                "risk_policy": "primary CE full UQ research policy; risk checks retained as exploratory",
                                "fallback_arm": "original query / No-Rewrite"},
        "evaluation": {"query_ordering": "exact action_items.pkl order", "metrics": ["absolute_MRR@100", "delta_MRR", "coverage", "harm_frequency", "harm_magnitude", "verification_cost", "UQ_cost"],
                       "bootstrap_unit": "repository cluster", "interval_method": "repository-cluster bootstrap in existing report", "effect_margin": "not used in development replay",
                       "multiple_comparison_correction": "eight exploratory risk checks at alpha=.05/8", "raw_outputs": [{"path": rel(FLOW / "summary.json"), "sha256": sha(FLOW / "summary.json")}]},
        "environment": {"os": platform.platform(), "python": sys.version.split()[0], "torch": torch_version,
                        "transformers": transformers_version, "cuda": "runtime_not_imported_during_freeze",
                        "device": "runtime_recorded_in_existing_protocol", "dependency_lockfile": {"path": "not_available_legacy_development", "sha256": None}},
        "artifact_acceptance": {"clean_directory_command": "python experiments/rene_artifact_acceptance_20261018/accept.py", "smoke_test_command": "python experiments/strong_flow_uq_20261014/audit_flow.py --actual", "rebuild_core_tables": True, "hash_match_required": True, "acceptance_report": "results/rene_artifact_acceptance_20261018/ACCEPTANCE.json"},
        "required_before_confirmation": ["fill independent confirmation manifest", "repository-disjoint confirmation rows", "freeze thresholds and statistical decision rule", "do not read consumed AdvTest test"],
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(OUT)


if __name__ == "__main__":
    main()
