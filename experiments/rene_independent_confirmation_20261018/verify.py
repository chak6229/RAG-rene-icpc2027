"""Verify the sealed one-time RENE confirmation without rerunning models."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rene_independent_confirmation_20261018"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_keys(value, names: set[str], found: set[str]) -> None:
    if isinstance(value, dict):
        found.update(set(value) & names)
        for child in value.values():
            find_keys(child, names, found)
    elif isinstance(value, list):
        for child in value:
            find_keys(child, names, found)


def main() -> None:
    protocol_path = OUT / "PROTOCOL.json"
    seal_path = OUT / "DECISION_SEAL.json"
    labels_path = OUT / "label_free_decisions.pkl"
    result_path = OUT / "RESULTS.json"
    manifest_path = OUT / "CONFIRMATION_MANIFEST.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    seal = json.loads(seal_path.read_text(encoding="utf8"))
    result = json.loads(result_path.read_text(encoding="utf8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    with labels_path.open("rb") as stream:
        label_free = pickle.load(stream)

    assert protocol["status"] == "FROZEN_BEFORE_LABELS"
    assert manifest["status"] == "COMPLETED_ONE_TIME_CONFIRMATION"
    assert protocol["test_used"] is False and manifest["test_used"] is False
    assert result["test_used"] is False and result["one_time_confirmation"] is True
    assert seal["protocol_sha256"] == sha(protocol_path)
    assert seal["label_free_sha256"] == sha(labels_path)
    assert result["protocol_sha256"] == sha(protocol_path)
    assert result["decision_seal_sha256"] == sha(seal_path)
    assert manifest["label_free_decisions_sha256"] == sha(labels_path)
    assert manifest["results_sha256"] == sha(result_path)
    assert label_free["protocol_sha256"] == sha(protocol_path)
    forbidden = set()
    find_keys(label_free, {"rr_at_100", "delta_rr", "original_rr", "harm"}, forbidden)
    assert not forbidden, f"label-free cache contains outcome keys: {sorted(forbidden)}"
    methods = result["methods"]
    assert set(methods) >= {"no_uq", "seed_uq", "oracle", "seed101_best"}
    assert methods["no_uq"]["query_count"] == protocol["query_count"]
    assert methods["seed_uq"]["query_count"] == protocol["query_count"]
    report = {"passed": True, "test_used": False, "one_time_confirmation": True,
              "query_count": protocol["query_count"], "forbidden_label_keys": sorted(forbidden),
              "protocol_sha256": sha(protocol_path), "label_free_sha256": sha(labels_path),
              "results_sha256": sha(result_path)}
    (OUT / "VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
