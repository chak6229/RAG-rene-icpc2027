"""Bind the completed audit documents and local evidence to exact file hashes."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from discover_sources import ROOT, OUT


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    verification = json.loads((OUT / "VERIFICATION.json").read_text(encoding="utf-8"))
    decision = json.loads((OUT / "AUDIT_DECISION.json").read_text(encoding="utf-8"))
    assert verification["status"] == "PASS" and verification["passed_checks"] == 91
    assert decision["audit_completed"] and not decision["formal_replication_execution_ready"]
    report = (OUT / "REPORT.md").read_text(encoding="utf-8")
    for required in ("6,267", "4,011", "6,490", "set_seed(42)", "docstring_tokens", "91"):
        assert required in report
    output_path = OUT / "AUDIT_DELIVERY.json"
    paths = [p for p in OUT.rglob("*") if p.is_file() and p != output_path and "__pycache__" not in p.parts]
    paths.extend((ROOT / "experiments/rene_cosqa_ssqr_audit_20261019").glob("*.py"))
    paths.extend([
        ROOT / 'ResearchVault/Handoffs/ RENE - CoSQA and SSQR review -20261019.md',
        ROOT / "results/rene_thesis_review_20261019/SSQR.html",
        ROOT / "results/rene_thesis_review_20261019/SSQR.txt",
    ])
    files = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(set(paths))]
    protected = json.loads((OUT / "LOCAL_CHECKPOINTS.json").read_text(encoding="utf-8"))["protected_snapshot"]
    assert all(sha256(ROOT / path) == expected for path, expected in protected.items())
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "AUDIT_COMPLETE",
                "scope": "data/source/checkpoint and input-contract audit; not scientific-effect confirmation",
                "verification_status": "PASS", "verification_checks": verification["passed_checks"],
                "audit_decision": "CoSQA accepted for new-query single-target replication; aligned execution/protocol still next",
                "files": files, "file_count": len(files), "protected_snapshot_rechecked": protected,
                "upstream_large_inputs": "hashes recorded by original streaming audit in OVERLAP_AUDIT.json; these datasets are not duplicated here",
                "mutable_research_memory": "daily/project/agent notes are linked but intentionally not bound as immutable scientific artifacts"}
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "files": len(files), "checks": 91,
                      "manifest_sha256": sha256(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
