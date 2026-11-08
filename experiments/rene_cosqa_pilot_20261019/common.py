from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rene_cosqa_pilot_20261019"
ADAPTER = ROOT / "results/rene_ssqr_alignment_20261019/cosqa_adapter"
INVENTORY = ROOT / "results/rene_cosqa_ssqr_audit_20261019/QUERY_GROUP_INVENTORY.jsonl"
T5 = ROOT / "models/t5-cqc-codenn-1m/best"
CODEBERT = ROOT / "models/codebert-ssqr-pooler-2epoch-20261019"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(2 ** 23):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def readl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value, *, immutable=False):
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"frozen artifact already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if immutable else "w", encoding="utf-8") as handle:
        handle.write(encoded)


def writel(path, rows, *, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if immutable else "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def verify_frozen(*, include_private=False):
    protocol = read(OUT / "PROTOCOL.json")
    if protocol["status"] != "FROZEN" or protocol["role"] != "development":
        raise ValueError("only the frozen development pilot is authorized here")
    for row in protocol["bound_files"] + protocol["protected_files"]:
        if "/data/private/" in row["path"] and not include_private:
            continue
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"frozen input drift: {row['path']}")
    return protocol
