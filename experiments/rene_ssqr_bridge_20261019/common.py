from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / "results/rene_ssqr_bridge_20261019"
DEVELOPMENT = ROOT / "results/rene_cosqa_development_20261019"
ADAPTER = ROOT / "results/rene_ssqr_alignment_20261019/cosqa_adapter"
INVENTORY = ROOT / "results/rene_cosqa_ssqr_audit_20261019/QUERY_GROUP_INVENTORY.jsonl"
SOURCE = ROOT / "data/codexglue/advtest/test.jsonl"
T5 = ROOT / "models/t5-cqc-codenn-1m/best"
CODEBERT = ROOT / "models/codebert-ssqr-pooler-2epoch-20261019"
PROTECTED = []  # Unrelated author-workspace registries are not part of this release.
PROFILES = {
    "greedy20_source": {"bank": "baseline", "positions": "source_1_to_n", "max_new_tokens": 19, "sample": False},
    "greedy20_all": {"bank": "baseline", "positions": "paper_0_to_n", "max_new_tokens": 19, "sample": False},
    "greedy10_source": {"bank": "greedy10", "positions": "source_1_to_n", "max_new_tokens": 9, "sample": False},
    "greedy10_all": {"bank": "greedy10", "positions": "paper_0_to_n", "max_new_tokens": 9, "sample": False},
    "sample10_source": {"bank": "sample10", "positions": "source_1_to_n", "max_new_tokens": 9, "sample": True},
    "sample10_all": {"bank": "sample10", "positions": "paper_0_to_n", "max_new_tokens": 9, "sample": True},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def readl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value, *, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def writel(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def priority(namespace, value):
    return hashlib.sha256(f"rene-bridge-20261019/{namespace}\0{value}".encode("utf-8")).hexdigest()


def text_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def position_seed(query_id, position, seed=101):
    return int.from_bytes(hashlib.sha256(f"{seed}\0{query_id}\0{position}".encode()).digest()[:8], "big") % (2**31 - 1)


def verify_frozen(*, private=False):
    protocol = read(OUT / "PROTOCOL.json")
    require(protocol["status"] == "FROZEN", "bridge design is not frozen")
    require(protocol["role"] == "post_confirmation_supplementary_diagnostic", "wrong evidence role")
    for row in protocol["bound_files"] + protocol["protected_files"]:
        if "/private/" in row["path"] and not private:
            continue
        require(sha(ROOT / row["path"]) == row["sha256"], f'frozen input changed: {row["path"]}')
    return protocol


def verify_seal(name):
    seal = read(OUT / name)
    require(seal["protocol_sha256"] == sha(OUT / "PROTOCOL.json"), f"{name} protocol mismatch")
    for row in seal["files"]:
        require(sha(ROOT / row["path"]) == row["sha256"], f'sealed file changed: {row["path"]}')
    return seal


@contextlib.contextmanager
def deny_private_reads():
    """Fail if inference accidentally opens embedded source labels or private files."""
    original_builtin, original_io, original_path = builtins.open, io.open, Path.open
    source_path = SOURCE.resolve()

    def check(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, bytes, Path)) and "r" in mode:
            path = Path(file).resolve()
            if path == source_path or "private" in path.parts:
                raise PermissionError(f"label-bearing input is prohibited during inference: {path.name}")

    def guarded_builtin(file, mode="r", *args, **kwargs):
        check(file, mode, *args, **kwargs)
        return original_builtin(file, mode, *args, **kwargs)

    def guarded_io(file, mode="r", *args, **kwargs):
        check(file, mode, *args, **kwargs)
        return original_io(file, mode, *args, **kwargs)

    def guarded_path(file, mode="r", *args, **kwargs):
        check(file, mode, *args, **kwargs)
        return original_path(file, mode, *args, **kwargs)

    builtins.open, io.open, Path.open = guarded_builtin, guarded_io, guarded_path
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = original_builtin, original_io, original_path


def pools_for(domain, profile, *, sensitivity=False):
    from collections import defaultdict

    protocol = read(OUT / "PROTOCOL.json")
    ids = protocol["samples"][domain]["sensitivity_ids" if sensitivity else "main_ids"]
    allowed = set(ids)
    setting = PROFILES[profile]
    baseline = DEVELOPMENT / "candidates.jsonl" if domain == "cosqa" else OUT / "candidates/source/baseline.jsonl"
    if setting["bank"] == "baseline":
        paths = [baseline]
        if setting["positions"] == "paper_0_to_n":
            paths.append(OUT / f"candidates/{domain}/greedy20_position0.jsonl")
    else:
        paths = [OUT / f'candidates/{domain}/{setting["bank"]}.jsonl']
    grouped = defaultdict(list)
    for path in paths:
        for row in readl(path):
            if row["query_id"] not in allowed:
                continue
            if setting["positions"] == "source_1_to_n" and row["position"] == 0:
                continue
            grouped[row["query_id"]].append(row)
    require(set(grouped) == allowed, "profile has incomplete query coverage")
    for query_id, rows in grouped.items():
        require(len({r["position"] for r in rows}) == len(rows), "profile has duplicate insertion positions")
        grouped[query_id] = sorted(rows, key=lambda r: r["position"])
    return {query_id: grouped[query_id] for query_id in ids}
