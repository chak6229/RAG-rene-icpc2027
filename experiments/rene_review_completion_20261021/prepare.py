"""Accept a separately confirmed return while preserving all original bytes."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from experiments.rene_relevance_sensitivity_20261021.common import ROOT, frozen, put, read, rows, sha
from experiments.rene_relevance_sensitivity_20261021.collect import validate_return
from experiments.rene_review_final_reanalysis_20261021.analyze import normalize_supplement

OUT = ROOT / "results/rene_review_completion_20261021"
PRIOR = ROOT / "results/rene_review_final_reanalysis_20261021"
PAPER = ROOT / "paper/rene2027"
EDITED = ["abstract.md", "intro_draft.md", "study_design.md", "main_results_narrative.md",
          "discussion.md", "conclusion.md", "README.md", "relevance_sensitivity_results.md",
          "review_disagreement_discussion.md", "integrated_results.md", "contribution_evidence.md",
          "knowledge_increment.md", "outline.md", "results_contract.md", "manuscript.md"]


def binding(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def main():
    assert not OUT.exists(), "Create a new version rather than overwrite a completed delivery."
    protocol = frozen()
    prior = read(PRIOR / "RESULTS.json")
    numeric = read(PRIOR / "NUMERICAL_VALIDATION.json")
    assert numeric["passed"] and numeric["results_sha256"] == sha(PRIOR / "RESULTS.json")
    assert numeric["query_bounds_sha256"] == sha(PRIOR / "query_bounds.jsonl")
    assert numeric["independent_exact_paired_bounds"] == 6144
    receipt = read(PRIOR / "RECEIPT_MANIFEST.json")
    for item in receipt["files"]:
        assert sha(Path(item["source"])) == item["sha256"]
        assert sha(PRIOR / item["copy"]) == item["sha256"]

    OUT.mkdir()
    before = OUT / "before"
    before.mkdir()
    for name in EDITED:
        shutil.copyfile(PAPER / name, before / name)
    put(OUT / "PAPER_BEFORE.json", {"files": [binding(PAPER / n) for n in EDITED]})

    protected = {p for p in PRIOR.rglob("*") if p.is_file()}
    protected.update(Path(i["source"]) for i in receipt["files"])
    protected.update(Path(i["path"]) for i in read(PRIOR / "PROTECTED_FILES.json")["files"]
                     if Path(i["path"]) not in {PAPER / n for n in EDITED})
    protected.update(ROOT / i["path"] for i in protocol["bound_inputs_and_design"])
    protected.update(p for p in (ROOT / "results/rene_manuscript_integration_20261021").rglob("*") if p.is_file())
    protected.update(p for p in (ROOT / "experiments/rene_review_final_reanalysis_20261021").glob("*.py"))
    put(OUT / "PROTECTED_FILES.json", {"scope": "returns, prior numerical evidence, frozen inputs and original 15-table bundle",
                                         "files": [binding(p) for p in sorted(protected)]})

    shutil.copytree(PRIOR / "received", OUT / "received")
    for name in ["query_bounds.jsonl", "supplement_labels.jsonl", "supplement_disagreements.jsonl",
                 "REPRESENTATIVE_DISAGREEMENTS.json"]:
        shutil.copyfile(PRIOR / name, OUT / name)
    put(OUT / "RECEIPT_MANIFEST.json", {"prior_receipt": binding(PRIOR / "RECEIPT_MANIFEST.json"),
                                         "files": receipt["files"]})
    a_path = OUT / "received/补审A/relevance-supplement-A-complete.json"
    a_payload = read(a_path)
    assert a_payload["independence_confirmed"] is False
    confirmation = {
        "reviewer": "A", "phase": "relevance_supplement", "packet_id": a_payload["packet_id"],
        "original_return": binding(a_path), "original_independence_flag": False,
        "evidence_source": "explicit human user reply in the current conversation",
        "question": "补审 A 的文件中“独立完成”未勾选。请确认：A 是否独立完成了补审，且未查看原实验效果、旧评分或 B 的判断？",
        "answer_verbatim": "是，A 确实独立完成，只是漏勾选",
        "confirmed_scope": {"completed_independently": True, "did_not_view_experimental_outcomes": True,
                            "did_not_view_previous_ratings": True, "did_not_view_reviewer_B_judgments": True},
        "record_type": "separate organizer confirmation; not a rewritten reviewer return",
        "return_bytes_modified": False, "judgments_modified": False,
    }
    put(OUT / "INDEPENDENCE_CONFIRMATION.json", confirmation)
    keys = read(ROOT / "results/rene_relevance_sensitivity_20261021/private/reviewer_keys.json")
    accepted = {}
    for reviewer in ("A", "B"):
        payload = read(OUT / f"received/补审{reviewer}/relevance-supplement-{reviewer}-complete.json")
        normalized = normalize_supplement(payload, keys[reviewer], reviewer)
        assert len(normalized) == 344
        if reviewer == "A":
            try:
                validate_return(payload, reviewer, keys[reviewer])
                raise AssertionError("The unchanged collector must preserve its original decision.")
            except ValueError as error:
                assert str(error) == "Complete independent return required"
        else:
            assert len(validate_return(payload, reviewer, keys[reviewer])) == 344
        accepted[reviewer] = {"content_complete": True, "formal_acceptance": True, "accepted_labels": 344,
                              "unchanged_collector_accepts_raw_return": reviewer == "B",
                              "independence_confirmed_in_return": reviewer == "B",
                              "acceptance_basis": "explicit_user_confirmation_sidecar" if reviewer == "A" else "original_return_declaration"}

    current = copy.deepcopy(prior)
    current["analysis_version"] = "completed_review_with_separate_user_confirmation"
    current["status"] = "formal_dual_independent_supplement_complete"
    current["supplement_status"] = accepted
    current["supplement_labels_formally_accepted"] = 688
    current["formal_dual_independent_supplement_completion"] = True
    current["independence_confirmation"] = binding(OUT / "INDEPENDENCE_CONFIRMATION.json")
    current["numerical_evidence_source"] = binding(PRIOR / "RESULTS.json")
    current["prior_independent_numerical_validation"] = binding(PRIOR / "NUMERICAL_VALIDATION.json")
    current.pop("source_and_prior_files_unchanged")
    current["source_returns_and_prior_numerical_results_unchanged"] = True
    put(OUT / "RESULTS.json", current)
    put(OUT / "REVIEW_ACCEPTANCE.json", {
        "passed": True, "formal_dual_independent_supplement_completion": True,
        "returned_labels": 688, "formally_accepted_labels": 688,
        "accepted_by_unchanged_collector": 344, "accepted_with_explicit_separate_confirmation": 344,
        "reviewers": accepted, "confirmation": binding(OUT / "INDEPENDENCE_CONFIRMATION.json"),
        "source_returns_preserved": True, "existing_labels_overwritten": False,
        "unknowns_and_disagreements_retained": True, "new_queries": 0,
    })
    scope = read(PRIOR / "ANALYSIS_SCOPE.json")
    scope.pop("formal_completion_depends_on_return_independence_confirmation")
    put(OUT / "ANALYSIS_SCOPE.json", {
        **scope, "formal_dual_independent_supplement_completion": True,
        "independence_evidence": {"A": "separate_explicit_user_confirmation", "B": "original_return_declaration"},
        "acceptance_record": "REVIEW_ACCEPTANCE.json", "confirmation_record": "INDEPENDENCE_CONFIRMATION.json",
        "paper_revision_sources": EDITED, "original_15_table_bundle_preserved": True,
    })
    print({"accepted_supplement_labels": 688, "source_returns_preserved": True,
           "protected_files": len(protected), "paper_sources_backed_up": len(EDITED), "output": str(OUT)})


if __name__ == "__main__":
    main()
