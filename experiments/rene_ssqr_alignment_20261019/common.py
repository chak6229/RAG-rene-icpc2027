from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rene_ssqr_alignment_20261019"
MODEL_OUTPUT = ROOT / "models/codebert-ssqr-pooler-2epoch-20261019"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**23), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value, *, immutable: bool = False):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if immutable and path.exists():
        if read_json(path) != value:
            raise ValueError(f"existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    # Every replace is restricted to the same explicitly resolved parent.
    if temporary.resolve().parent != path.resolve().parent:
        raise ValueError("atomic write escaped target directory")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
