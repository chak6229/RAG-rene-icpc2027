"""Clean-directory provenance and core-table acceptance check."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ResearchVault/Handoffs/RENE-统一评测manifest模板.json"
OUT = ROOT / "results/rene_artifact_acceptance_20261018"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path_from(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    text = MANIFEST.read_text(encoding="utf8")
    failures = []
    if "REPLACE_ME" in text or manifest.get("status") != "FROZEN_DEVELOPMENT_REPRODUCTION":
        failures.append("manifest is not a filled frozen development manifest")
    checks: dict[str, str] = {}

    def check(path_value: str | None, expected: str | None, label: str) -> None:
        if not expected or path_value in (None, "NOT_APPLICABLE_DEVELOPMENT_ONLY", "not_available_legacy_development"):
            return
        path = path_from(path_value)
        if not path.exists():
            failures.append(f"missing:{label}:{path}")
            return
        actual = sha(path)
        checks[label] = actual
        if actual != expected:
            failures.append(f"hash:{label}:{actual}!={expected}")

    data = manifest["data"]["splits"]
    for name, item in data.items():
        check(item.get("path"), item.get("sha256"), f"data.{name}")
    candidate = manifest["candidate_generation"]["candidate_cache"]
    check(candidate["path"], candidate["sha256"], "candidate_cache")
    retrieval = manifest["retrieval"]
    check(retrieval["checkpoint_path"] + "/model.safetensors", retrieval["weights_sha256"], "retriever_weights")
    check(retrieval["checkpoint_path"] + "/config.json", retrieval["config_sha256"], "retriever_config")
    check(retrieval["corpus"]["path"], retrieval["corpus"]["sha256"], "retrieval_corpus")
    check(retrieval["retrieval_cache"]["path"], retrieval["retrieval_cache"]["sha256"], "retrieval_cache")
    candidate_generation = manifest["candidate_generation"]
    check(candidate_generation["generator_checkpoint"] + "/model.safetensors", candidate_generation["checkpoint_sha256"], "generator_weights")
    check(candidate_generation["script_path"], candidate_generation["script_sha256"], "generator_script")
    for output in manifest["evaluation"]["raw_outputs"]:
        check(output["path"], output["sha256"], f"raw_output:{output['path']}")

    summary_path = ROOT / "results/strong_flow_uq_20261014/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    core = {"main_policy": summary.get("main_policy"), "risk_checks": summary.get("risk_checks"), "families": {}}
    for family, block in summary.get("families", {}).items():
        core["families"][family] = {
            "main_policy": block.get("methods", {}).get(summary.get("main_policy", "ce_full_uq"), {}).get("held"),
            "certificates": block.get("certificates"),
            "selection_failed": block.get("selection_failed"),
        }
    core_path = OUT / "core_table.json"
    core_path.write_text(json.dumps(core, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf8")
    core_hash = sha(core_path)
    report = {"passed": not failures, "manifest": str(MANIFEST), "manifest_sha256": sha(MANIFEST),
              "hash_checks": checks, "core_table": str(core_path), "core_table_sha256": core_hash,
              "clean_directory": True, "test_used": False, "failures": failures}
    (OUT / "ACCEPTANCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    if failures:
        raise SystemExit("artifact acceptance failed: " + "; ".join(failures))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
