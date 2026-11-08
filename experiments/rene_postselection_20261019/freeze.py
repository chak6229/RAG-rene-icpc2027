from __future__ import annotations

import subprocess
import sys

from experiments.rene_cosqa_development_20261019.common import verify_decisions, verify_frozen
from .common import (ALGORITHMS, BLOCKS, CALIBRATORS, OUT, PARENT, ROOT, SOURCE, now, read, record,
                     require, sha, write)


def freeze():
    require(not (OUT / "PROTOCOL.json").exists(), "protocol already frozen")
    parent = verify_frozen(private=True)
    verify_decisions()
    require(read(PARENT / "FINAL_ACCEPTANCE.json")["passed"], "parent development not accepted")
    for retriever in parent["retrievers"]:
        seal = read(PARENT / f"OOF_SEALED_{retriever}.json")
        for row in seal["files"]:
            require(sha(PARENT / row["path"]) == row["sha256"], "parent OOF changed")
    test = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest",
                           "experiments.rene_postselection_20261019.test_contract",
                           "experiments.rene_cosqa_development_20261019.test_contract",
                           "experiments.rene_cosqa_pilot_20261019.test_contract"],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    require(test.returncode == 0, f"contract tests failed\n{test.stdout}\n{test.stderr}")
    write(OUT / "TESTS_PRE_FREEZE.json", {"created_utc": now(), "returncode": test.returncode,
                                       "output": test.stdout + test.stderr}, immutable=True)
    files = list(SOURCE.glob("*.py")) + [SOURCE / "README.md", OUT / "TESTS_PRE_FREEZE.json"]
    files += [PARENT / name for name in ("PROTOCOL.json", "ROLE_MANIFEST.json", "FOLDS.json", "FINAL_ACCEPTANCE.json",
                                        "data/groups.json", "data/queries.jsonl", "data/private/development_qrels.jsonl",
                                        "selections.jsonl", "DECISIONS_SEALED.json", "READOUT.json")]
    for retriever in parent["retrievers"]:
        files += [PARENT / f"{prefix}/{retriever}.jsonl" for prefix in ("features", "rankings", "oof_predictions", "decisions")]
        files += [PARENT / f"OOF_SEALED_{retriever}.json", PARENT / f"TRAINING_LEDGER_{retriever}.json"]
        files += list((PARENT / f"models/{retriever}").glob("*-fold*.joblib"))
    # Imported immutable source is bound explicitly, without touching the parent's seal.
    files += list((ROOT / "experiments/rene_cosqa_development_20261019").glob("*.py"))
    files += [ROOT / "experiments/rene_cosqa_pilot_20261019/common.py",
              ROOT / "experiments/rene_cosqa_pilot_20261019/evaluation.py"]
    unique = sorted(set(files))
    protocol = {
        "created_utc": now(), "status": "FROZEN", "role": "consumed_development_only",
        "parent_protocol_sha256": sha(PARENT / "PROTOCOL.json"),
        "prior_effects_known": True, "design": "follow-up to observed selection-conditional optimism; no new confirmation claim",
        "confirmation_authorized": False, "new_generation_or_retrieval": False, "new_seeds_or_masks": 0,
        "queries": len(parent["query_ids"]), "groups": 512, "retrievers": parent["retrievers"],
        "denominator": "all 1490 permanent-D queries, never only changed or adopted queries",
        "candidate_pool": "the parent's exactly three cached ENTR candidates per query",
        "outer_folds": "reuse frozen five whole-connected-group folds",
        "inner_folds": "four remaining original fold IDs inside each outer training set; no outer labels enter inner fitting",
        "base_models": {"algorithms": ALGORITHMS, "blocks": BLOCKS,
                        "hgb_parameters": parent["direct_gain"]["hgb_parameters"], "ridge_alpha": 10.0,
                        "reused": ["FULL_HGB", "FULL_RIDGE", "GEN_HGB"],
                        "new_refits": "all other block/algorithm combinations; same hyperparameters",
                        "primary_in_parent": "FULL_HGB; the Ridge control is not retroactively promoted"},
        "calibration": {
            "base_models": ["FULL_HGB", "FULL_RIDGE"], "methods": CALIBRATORS,
            "training_cohort": "raw positive, non-identical winners selected from inner group-OOF predictions",
            "test_cohort": "raw positive, non-identical winners selected from outer group-OOF predictions",
            "AFFINE": "StandardScaler + Ridge(alpha=10), winner score only",
            "ISOTONIC": "monotone increasing, clipped extrapolation, y bounded [-1,1]",
            "CONTEXT_RIDGE": "StandardScaler + Ridge(alpha=10), winner score, runner-up incl NoRewrite, gap, and 52 retrieval/anchor/lexical features",
            "small_sample_fallback": "<30 training selected actions: constant training mean; no actions: zero",
            "winner_preserved": True, "may_rescue_raw_nonpositive_action": False,
            "hyperparameter_or_threshold_tuning": False,
            "adoption": "raw eligible and calibrated expected gain strictly >0; otherwise complete NoRewrite ranking",
            "primary_check": "AFFINE vs RAW selected-cohort MSE and mean bias, utility reported separately",
        },
        "feature_blocks": {"GEN": "19 generation/text", "GS": "GEN +36 score summaries",
                           "GSA": "GS +9 anchor/ranking correspondence", "GSL": "GS +7 lexical code-coverage",
                           "FULL": "all71; anchors are retrieved documents, never gold target ranks"},
        "fixed_coverage": {
            "base_ablation": "top floor(.20 * outer-held queries) changed-ranking winners, no score sign gate",
            "calibration": "top floor(.50 * raw-eligible outer-held queries); exact same count for RAW and three calibrators per family",
            "tie": "descending policy score, descending original raw winner score, then query_id; isotonic plateaus preserve raw order",
            "purpose": "separate ordering evidence from adopting fewer queries; calibration half-eligible is not 50% of all queries",
        },
        "metrics": ["query-weighted delta MRR@100", "adoption count and coverage", "harmful/adopted", "positive gain and loss",
                    "fixed-cohort selected-action MSE/MAE/bias", "fixed-bin calibration", "paired equal-coverage utility"],
        "calibration_bins": [-1, -.10, -.02, 0, .005, .01, .02, .05, .10, .25, 1.000001],
        "statistics": {"bootstrap": 5000, "seed": 20261019, "unit": "whole connected group",
                       "interval": "descriptive paired 95% percentile; conditional on fitted OOF models; no familywise or confirmation claims",
                       "model_refit_inside_bootstrap": False},
        "gold_evidence_audit": "D-only original/candidate top1 relevance and RR strata; audit-only, never predictive inputs or rule selection",
        "outcome_summary_after_predictions_sealed": True,
        "deployment": "NoRewrite", "risk_certified": False, "promote_formal_model": False,
        "bound_files": [record(path) for path in unique], "protected_files": parent["protected_files"],
    }
    write(OUT / "PROTOCOL.json", protocol, immutable=True)
    print({"protocol_sha256": sha(OUT / "PROTOCOL.json"), "bound_files": len(unique),
           "tests": test.stdout + test.stderr, "confirmation_opened": False}, flush=True)


if __name__ == "__main__":
    freeze()
