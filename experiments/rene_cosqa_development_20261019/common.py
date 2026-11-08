from __future__ import annotations

from pathlib import Path

from experiments.rene_cosqa_pilot_20261019.common import (
    ROOT, ADAPTER, INVENTORY, T5, CODEBERT, now, read, readl, sha, write, writel,
    verify_frozen as verify_pilot,
)

PILOT = ROOT / "results/rene_cosqa_pilot_20261019"
OUT = ROOT / "results/rene_cosqa_development_20261019"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_parent():
    verify_pilot()
    seal = read(PILOT / "DECISIONS_SEALED.json")
    require(seal["protocol_sha256"] == sha(PILOT / "PROTOCOL.json"), "pilot protocol changed")
    for record in seal["files"]:
        require(sha(PILOT / record["path"]) == record["sha256"], "sealed pilot cache changed")
    require(read(PILOT / "FINAL_ACCEPTANCE.json")["passed"], "pilot acceptance missing")


def verify_frozen(*, private=False, parent=False):
    if parent:
        verify_parent()
    p = read(OUT / "PROTOCOL.json")
    require(p["status"] == "FROZEN" and p["role"] == "development", "not a frozen development run")
    require(p["confirmation_authorized"] is False, "confirmation is outside this entry point")
    for row in p["bound_files"] + p["protected_files"]:
        if "/data/private/" in row["path"] and not private:
            continue
        require(sha(ROOT / row["path"]) == row["sha256"], f'frozen input drift: {row["path"]}')
    return p


def verify_decisions():
    seal = read(OUT / "DECISIONS_SEALED.json")
    require(seal["protocol_sha256"] == sha(OUT / "PROTOCOL.json"), "extension seal protocol mismatch")
    for row in seal["files"]:
        require(sha(OUT / row["path"]) == row["sha256"], f'sealed cache changed: {row["path"]}')
    return seal


def group_folds(group_ids, folds=5):
    import hashlib
    ordered = sorted(set(group_ids), key=lambda value: hashlib.sha256(
        f"rene-cosqa-direct-gain-fivefold-20261019\0{value}".encode()).hexdigest())
    return {group: index % folds for index, group in enumerate(ordered)}


def record(path):
    return {"path": Path(path).relative_to(ROOT).as_posix(), "sha256": sha(path)}
