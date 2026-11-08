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
OUT = ROOT / "results/rene_envelope_completion_20261020"
BRIDGE = ROOT / "results/rene_ssqr_bridge_20261019"
OLD = ROOT / "results/rene_ssqr_rewriter_envelope_20261020"
MODEL = ROOT / "models/codebert-ssqr-pooler-2epoch-20261019"
DOMAINS = ("source", "cosqa")
PROTECTED = [
    "configs/vac_mqr_gain_model/CURRENT.json",
    "configs/vac_mqr_verification_policy/CURRENT.json",
    "configs/vac_mqr_risk_module/CURRENT.json",
    "configs/uq_candidate_selection/CURRENT.json",
    "results/vac_mqr_v5/STABLE.json",
    "results/rene_cosqa_confirmation_20261019/PROTOCOL.json",
    "results/rene_cosqa_confirmation_20261019/PRIMARY_RESULT.json",
    "results/rene_cosqa_confirmation_20261019/FINAL_ACCEPTANCE.json",
    "results/rene_core_step_a_20261020/RESULTS.json",
    "results/rene_core_step_b_20261020/RESULTS.json",
    "results/rene_multicandidate_fusion_20261020/RESULTS.json",
    "results/rene_multicandidate_fusion_20261020/FINAL_ACCEPTANCE.json",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**23), b""):
            value.update(block)
    return value.hexdigest()


def text_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record(path):
    return {"path": Path(path).relative_to(ROOT).as_posix(), "sha256": sha(path)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def writel(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def paths(domain):
    require(domain in DOMAINS, "Unknown domain")
    return {
        "queries": BRIDGE / f"data/{domain}/queries.jsonl",
        "corpus": BRIDGE / "data/source/corpus.jsonl" if domain == "source" else ROOT / "results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl",
        "qrels": BRIDGE / f"data/private/{domain}_qrels.jsonl",
        "base_candidates": OLD / f"candidates/{domain}/t5base.jsonl",
        "local_candidates": BRIDGE / f"candidates/{domain}/baseline.jsonl" if domain == "source" else ROOT / "results/rene_cosqa_development_20261019/candidates.jsonl",
        "local_rankings": BRIDGE / f"rankings/{domain}/CodeBERT.jsonl",
        "codes": BRIDGE / "retrieval_cache/source/CodeBERT/code_vectors.npy" if domain == "source" else ROOT / "results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy",
    }


def verify():
    p = read(OUT / "PROTOCOL.json")
    require(p["role"] == "post_hoc_evidence_completion_not_new_confirmation", "Wrong role")
    for item in p["bound_files"] + p["protected_files"]:
        require(sha(ROOT / item["path"]) == item["sha256"], f'Frozen file changed: {item["path"]}')
    return p


def verify_seal(name):
    seal = read(OUT / name)
    require(seal["protocol_sha256"] == sha(OUT / "PROTOCOL.json"), "Protocol seal mismatch")
    for item in seal["files"]:
        require(sha(ROOT / item["path"]) == item["sha256"], f'Sealed file changed: {item["path"]}')
    return seal


@contextlib.contextmanager
def deny_labels():
    """Inference cannot open source test labels or any private/qrels file."""
    original_builtin, original_io, original_path = builtins.open, io.open, Path.open
    source = (ROOT / "data/codexglue/advtest/test.jsonl").resolve()

    def check(path, mode):
        if isinstance(path, (str, bytes, Path)) and "r" in mode:
            p = Path(path).resolve()
            if p == source or "private" in p.parts or "qrels" in p.name.lower():
                raise PermissionError("Label-bearing input prohibited during inference")

    def guarded_builtin(file, mode="r", *args, **kwargs):
        check(file, mode)
        return original_builtin(file, mode, *args, **kwargs)

    def guarded_io(file, mode="r", *args, **kwargs):
        check(file, mode)
        return original_io(file, mode, *args, **kwargs)

    def guarded_path(file, mode="r", *args, **kwargs):
        check(file, mode)
        return original_path(file, mode, *args, **kwargs)

    builtins.open, io.open, Path.open = guarded_builtin, guarded_io, guarded_path
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = original_builtin, original_io, original_path
