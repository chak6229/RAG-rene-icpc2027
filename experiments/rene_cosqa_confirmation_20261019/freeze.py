from __future__ import annotations

import importlib.metadata
import subprocess
import sys

from experiments.rene_cosqa_development_20261019.features import FEATURE_NAMES
from .common import (ADAPTER, CE, CODEBERT, INVENTORY, OUT, PARENT, PROTECTED, ROOT, SOURCE, T5,
                     now, read, record, require, sha, write)
from .policy import CE_NAMES, RISK_NAMES


def main():
    require(not (OUT / "DESIGN.json").exists(), "design already frozen")
    require(not (OUT / "OUTCOME_OPENING.json").exists(), "cannot freeze after opening")
    require(read(PARENT / "FINAL_ACCEPTANCE.json")["passed"], "parent acceptance missing")
    from .planning import prepare
    if not (OUT / "PRECISION_PLAN.json").exists():
        prepare()
    planning = read(OUT / "PRECISION_PLAN.json")
    require(planning["selected"]["eligible"] and planning["selected"]["planned_power_at_capacity"] >= .8, "precision not resolved")
    command = [sys.executable, "-X", "utf8", "-m", "unittest", "experiments.rene_cosqa_confirmation_20261019.test_contract", "-v"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, encoding="utf-8")
    write(OUT / "TEST_RESULTS.json", {"command": command, "exit_code": result.returncode,
                                      "output": result.stdout + result.stderr, "passed": result.returncode == 0}, immutable=True)
    require(result.returncode == 0, "pre-freeze integrity tests failed: " + result.stderr)
    roles = read(PARENT / "ROLE_MANIFEST.json")
    parent = read(PARENT / "PROTOCOL.json")
    p = {"status": "FROZEN", "created_utc": now(), "role": "independent_one_shot_H_confirmation",
         "data_source": "CoSQA positive user queries; same audited6267-code corpus",
         "H_query_ids": roles["confirmation_pool_query_ids"], "H_group_ids": roles["confirmation_pool_group_ids"],
         "H_queries": 8530, "H_groups": 1963, "all_D_queries_previously_consumed": 1490,
         "primary_sample": "all1963 groups, maximum16 queries per group, SHA256 order,7718 queries; standard query-weighted MRR",
         "primary_comparison": "CodeBERT:ENTR_minus_RAND20_best3:capped16",
         "primary_family_size": 1, "primary_alpha_two_sided": .05,
         "primary_inference": "paired connected-group t95CI, with group-bootstrap95CI sensitivity; no other null rejection claims",
         "equivalence_rule": "only if both t95CI and bootstrap95CI lie strictly inside[-.005,.005] and >=20 changed groups; otherwise unresolved. No promise of80%equivalence power",
         "auxiliary_comparisons": ["CodeBERT:ENTR_minus_PROB_best3", "BM25:ENTR_minus_RAND20_best3", "BM25:ENTR_minus_PROB_best3",
                                   "per-retriever:RIDGE_gain_minus_NoRewrite", "per-retriever:RIDGE_verified_minus_RIDGE_gain",
                                   "per-retriever:RIDGE_risk_minus_RIDGE_verified", "per-retriever:RIDGE_risk_minus_NoRewrite",
                                   "per-retriever:RIDGE_gain_minus_HGB_gain_control", "CodeBERT-minus-BM25 paired stage-effect differences"],
         "auxiliary_role": "predefined descriptive estimation with95%intervals; not additional confirmatory superiority/equivalence tests",
         "full_H_role": "predefined conventional8530-query sensitivity; not an equally powered co-primary",
         "single_adoption_policy": "RIDGE_risk", "policy_selection": "new D-selected study policy; parent development HGB remains original primary and retained control, never renamed",
         "study_adaptation": "pure rewrite actions over existing actual71features; no historical strong84 padding, no fusion claim; separateD gain/verification/risk roles",
         "gain_model": "StandardScaler+Ridge alpha10; positive max over exact ENTR min(3,N) plus NoRewrite0",
         "linear_control_retained": True, "hgb_control": parent["direct_gain"]["hgb_parameters"],
         "verification": {"budget": "highest positive base-gain priority min(floor(n/4),eligible) queries per retriever, queryID breaks ties",
                          "plan": "eight distinct original success anchors, winner/rival and largest RR-position changes; public rankings only",
                          "scorer": "frozen historical real code-query CE; original userquery conditioning; source training exposure retained",
                          "input": "query<=64 tokens, code<=160 tokens plus actual model specials",
                          "update": "separate-role weighted Ridge residual on71+2+18real witness features; mean+=.5residual; same ENTR pool, NoRewrite0",
                          "no_threshold_tuning": True},
         "risk": {"algorithm": "threeway zero-reference offset logistic; historical algorithm adapted to20real stage/anchor features",
                  "training": "separateD risk role; actual selected actions after exact25%verification allocation; no class weights",
                  "classes": ["negative", "zero", "positive"], "native_harm": "separate gain-role logistic C.1",
                  "penalty": "intercept1, verified coefficient50, other coefficients10; duplicate CEavailability masked",
                  "threshold": .075, "target_harm": .1, "fallback": "restore original fullTop100; no reject-and-reselect; retain sunkCEcost",
                  "certification": "no certificate from dependent query-level binomial counts; cluster intervals are descriptive",
                  "deployment_policy_promoted": False},
         "feature_names": FEATURE_NAMES, "CE_feature_names": CE_NAMES, "risk_feature_names": RISK_NAMES,
         "generation": parent["generation"], "bm25": parent["bm25"], "candidate_k": 3,
         "rand_seeds": parent["rand_seeds"], "ranking_depth": 100, "metric_k": 100,
         "retrievers": ["CodeBERT", "BM25"], "ranking_tie": "score descending, corpus docID ascending",
         "freeze_order": "design and precision -> separateD training -> complete model+policy seal -> H label-free generation/retrieval/CE/decisions -> independent pre-audit -> one outcome opening -> independent RR replay",
         "private_is_OS_isolated": False, "external_preregistration": False,
         "risk_certified": False, "official_pointers_changed": False,
         "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "joblib")}}
    sources = sorted(SOURCE.glob("*.py"))
    sources += [ROOT / "experiments/rene_cosqa_pilot_20261019/selection.py",
                ROOT / "experiments/rene_cosqa_pilot_20261019/evaluation.py",
                ROOT / "experiments/rene_cosqa_pilot_20261019/ranking.py",
                ROOT / "experiments/rene_cosqa_development_20261019/features.py",
                ROOT / "experiments/rene_cosqa_development_20261019/freeze.py",
                ROOT / "experiments/rene_ssqr_alignment_20261019/t5.py",
                ROOT / "experiments/rene_ssqr_alignment_20261019/retriever.py",
                ROOT / "experiments/rene_ssqr_alignment_20261019/pipeline.py"]
    sources += [ROOT / row["path"] for row in parent["bound_files"] if row["path"].endswith(".py")]
    inputs = [OUT / name for name in ("PRIMARY_SAMPLE.json", "STAGE_ROLES.json", "PRECISION_PLAN.json", "TEST_RESULTS.json",
                                     "public/H_queries.jsonl", "public/H_groups.json")]
    inputs += [PARENT / name for name in ("PROTOCOL.json", "ROLE_MANIFEST.json", "DECISIONS_SEALED.json", "FINAL_ACCEPTANCE.json",
                                        "data/private/development_qrels.jsonl", "metrics/CodeBERT_per_query.jsonl")]
    inputs += [PARENT / f"{sub}/{r}.jsonl" for r in ("CodeBERT", "BM25") for sub in ("features", "rankings")]
    inputs += [ADAPTER / "manifest.json", ADAPTER / "public/corpus.jsonl", ADAPTER / "public/all_positive_queries.jsonl", INVENTORY,
               CE, ROOT / "models/codebert-base/config.json", ROOT / "models/t5-base/spiece.model",
               T5 / "model.safetensors", T5 / "config.json", CODEBERT / "model.safetensors", CODEBERT / "config.json",
               ROOT / "results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy"]
    inputs += [path for directory in (T5, CODEBERT, ROOT / "models/codebert-base") for path in directory.iterdir()
               if path.is_file() and path.suffix in (".json", ".txt", ".model")]
    p["bound_files"] = [record(path) for path in dict.fromkeys(sources + inputs)]
    p["protected_files"] = [record(ROOT / path) for path in PROTECTED]
    write(OUT / "DESIGN.json", p, immutable=True)
    write(OUT / "DESIGN_FREEZE_RECEIPT.json", {"created_utc": now(), "design_sha256": sha(OUT / "DESIGN.json"),
                                               "H_predictions_started": False, "H_outcomes_opened": False}, immutable=True)
    print({"design_frozen": True, "H_outcomes_opened": False, "precision_required_groups": planning["selected"]["required_groups"]}, flush=True)


if __name__ == "__main__":
    main()
