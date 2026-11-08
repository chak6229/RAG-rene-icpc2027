"""Freeze the remaining development groups, model baselines, and cache reuse."""
from __future__ import annotations

import copy
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

from .common import (ADAPTER, INVENTORY, OUT, PILOT, ROOT, T5, CODEBERT, group_folds, now, read, readl,
                     record, require, sha, verify_parent, write, writel)
from .features import FEATURE_NAMES, GEN_NAMES


MODEL_SPECS = {
    "GEN_HGB": {"features": "generation", "algorithm": "HistGradientBoostingRegressor"},
    "GAIN_RIDGE": {"features": "all", "algorithm": "StandardScaler+Ridge", "alpha": 10.0},
    "GAIN_HGB": {"features": "all", "algorithm": "HistGradientBoostingRegressor", "primary": True},
}
HGB = {"loss": "squared_error", "max_iter": 200, "learning_rate": .05, "max_leaf_nodes": 7,
       "max_depth": 3, "min_samples_leaf": 30, "l2_regularization": 10.0,
       "max_bins": 63, "early_stopping": False, "random_state": 20261019}


def main():
    require(not (OUT / "PROTOCOL.json").exists(), "extension already frozen")
    verify_parent()
    tests = []
    for module in ("experiments.rene_cosqa_development_20261019.test_contract",
                   "experiments.rene_cosqa_pilot_20261019.test_contract"):
        command = [sys.executable, "-X", "utf8", "-m", "unittest", module, "-v"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8", text=True)
        tests.append({"command": command, "exit_code": result.returncode, "output": result.stdout + result.stderr})
        require(result.returncode == 0, f"pre-freeze tests failed: {module}\n{tests[-1]['output']}")
    write(OUT / "TEST_RESULTS.json", {"passed": True, "runs": tests}, immutable=True)
    parent = read(PILOT / "PROTOCOL.json")
    roles = read(PILOT / "ROLE_MANIFEST.json")
    development = set(roles["development_query_ids"])
    pilot = roles["pilot_query_ids"]
    new_ids = sorted(development - set(pilot))
    query_ids = pilot + new_ids
    groups = {r["idx"]: r["group_id"] for r in readl(INVENTORY)}
    require(len(development) == 1490 and len(new_ids) == 1234, "unexpected frozen development allocation")
    require(len({groups[q] for q in new_ids}) == 426, "remaining groups do not match allocation")
    require(not {groups[q] for q in pilot} & {groups[q] for q in new_ids}, "pilot group was split")
    require(not development & set(roles["confirmation_pool_query_ids"]), "confirmation query entered D")
    public = {r["query_id"]: r for r in readl(ADAPTER / "public/all_positive_queries.jsonl")}
    writel(OUT / "data/queries.jsonl", [public[q] for q in query_ids])
    write(OUT / "data/groups.json", {q: groups[q] for q in query_ids}, immutable=True)
    write(OUT / "data/DEVELOPMENT_ROLE.json", {"status": "FROZEN", "role": "development", "query_ids": query_ids,
                                               "group_ids": roles["development_group_ids"],
                                               "roles_manifest_sha256": sha(PILOT / "ROLE_MANIFEST.json")}, immutable=True)
    shutil.copyfile(PILOT / "ROLE_MANIFEST.json", OUT / "ROLE_MANIFEST.json")
    folds = group_folds(roles["development_group_ids"])
    write(OUT / "FOLDS.json", {"folds": 5, "group_to_fold": folds,
                               "query_to_fold": {q: folds[groups[q]] for q in query_ids},
                               "assignment_without_labels": True}, immutable=True)
    # This is trusted label partitioning, not method-effect evaluation.
    qrels = [r for r in readl(ADAPTER / "private/all_positive_qrels.jsonl") if r["query_id"] in development]
    require({r["query_id"] for r in qrels} == development, "development qrels incomplete")
    writel(OUT / "data/private/development_qrels.jsonl", qrels)
    write(OUT / "DATA_ACCESS_LEDGER.json", {"created_utc": now(), "purpose": "development-only label partition",
                                           "labels_previously_read_for_provenance": True, "method_effects_computed_this_step": False,
                                           "confirmation_effects_opened": False, "public_private_is_file_separation": True}, immutable=True)
    protocol = copy.deepcopy(parent)
    protocol.update({"schema": "rene-cosqa-development-extension-v1", "created_utc": now(),
                     "query_ids": query_ids, "queries": len(query_ids), "groups": 512,
                     "parent_protocol_sha256": sha(PILOT / "PROTOCOL.json"),
                     "parent_decision_seal_sha256": sha(PILOT / "DECISIONS_SEALED.json"),
                     "pilot_query_ids": pilot, "new_query_ids": new_ids, "new_groups": 426,
                     "confirmation_authorized": False,
                     "extension_reason": "improve development variance/event estimates and evaluate same-pool direct gain selection",
                     "previous_pilot_effects_known_before_extension_design": True,
                     "inference": "development-only, group OOF; pilot and remaining groups also reported separately; no confirmation inference",
                     "reuse": {"pilot_candidates_and_rankings": "exact reuse after seal verification",
                              "dense_corpus_vectors": "reuse frozen pilot vectors; no code re-encoding",
                              "query_text_cache": "reuse every exact existing text; encode new texts only",
                              "new_t5_seeds_or_masks": 0},
                     "direct_gain": {"pool": "same ENTR min(3,N) candidates for every selection comparison",
                                     "label": "candidate RR@100 minus original RR@100",
                                     "folds": 5, "group_isolation": "audited code/recovered-repository connected components",
                                     "fold_assignment": "SHA256 group order round-robin; no outcomes",
                                     "models": MODEL_SPECS, "hgb_parameters": HGB,
                                     "feature_names": FEATURE_NAMES, "generation_feature_names": GEN_NAMES,
                                     "row_weight": "1 / query candidate count, normalized mean1 within training fold",
                                     "threshold": 0.0, "threshold_tuning": False, "hyperparameter_tuning": False,
                                     "forced_rule": "maximum predicted gain within fixed ENTR pool; tie native selected rank then position",
                                     "selective_rule": "include NoRewrite with known score0; adopt only if maximum candidate score>0",
                                     "structural_zero": "candidate ranking identical to original implies known gain0; override model score to0",
                                     "adoption": "actual Top100 ranking differs from original; fallback restores original full ranking",
                                     "prediction_clip": [-1.0, 1.0],
                                     "primary_model": "GAIN_HGB", "other_models": "prespecified development controls, not post-hoc winners",
                                     "future_full_fit": "save research-only all-D models after OOF; never use full-fit predictions to evaluate D"},
                     "rq2_contrasts": [f"{r}:{c}" for r in parent["retrievers"] for c in
                                       ("GAIN_HGB_forced_minus_ENTR_top1", "GAIN_HGB_selective_minus_NoRewrite")],
                     "rq2_inference": "descriptive OOF intervals conditional on fitted folds; not a proof of algorithm novelty or certified risk",
                     "confirmation_multiplicity": "final joint RQ1/RQ2 family and policy must be bound before opening confirmation",
                     "power_transport": "also report known D/H positive-query group-size distributions and homogeneous group-mean variance design-factor sensitivity; no guaranteed transport",
                     "runtime_versions": {name: importlib.metadata.version(name) for name in
                                          ("numpy", "scipy", "scikit-learn", "torch", "transformers", "joblib")},
                     "deployment_policy_promoted": False, "risk_certified": False})
    sources = sorted(Path(__file__).parent.glob("*.py"))
    sources += [ROOT / r["path"] for r in parent["bound_files"] if r["path"].endswith(".py")]
    paths = [PILOT / name for name in ("PROTOCOL.json", "ROLE_MANIFEST.json", "DECISIONS_SEALED.json", "FINAL_ACCEPTANCE.json")]
    paths += [PILOT / row["path"] for row in read(PILOT / "DECISIONS_SEALED.json")["files"]]
    paths += [ROOT / r["path"] for r in parent["bound_files"] if not r["path"].endswith(".py")
              and not r["path"].startswith("results/rene_cosqa_pilot_20261019/data/")]
    paths += [path for directory in (T5, CODEBERT) for path in directory.iterdir()
              if path.is_file() and path.suffix in (".json", ".txt", ".model")]
    paths += [OUT / name for name in ("ROLE_MANIFEST.json", "FOLDS.json", "data/queries.jsonl", "data/groups.json",
                                     "data/DEVELOPMENT_ROLE.json", "data/private/development_qrels.jsonl", "TEST_RESULTS.json")]
    protocol["bound_files"] = [record(path) for path in dict.fromkeys(sources + paths)]
    write(OUT / "PROTOCOL.json", protocol, immutable=True)
    write(OUT / "FREEZE_RECEIPT.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                                        "new_generation_started": False, "new_effects_opened": False,
                                        "confirmation_effects_opened": False}, immutable=True)
    print({"frozen": True, "all_D_queries": len(query_ids), "new_queries": len(new_ids), "new_groups": 426,
           "features": len(FEATURE_NAMES), "protocol_sha256": sha(OUT / "PROTOCOL.json")}, flush=True)


if __name__ == "__main__":
    main()
