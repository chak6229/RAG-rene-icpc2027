from __future__ import annotations

import contextlib
import contextvars
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

from experiments.rene_cosqa_pilot_20261019.common import (
    ROOT, ADAPTER, INVENTORY, T5, CODEBERT, now, sha, read, readl, write, writel,
)

SOURCE = Path(__file__).parent
PARENT = ROOT / "results/rene_cosqa_development_20261019"
OUT = ROOT / "results/rene_cosqa_confirmation_20261019"
CE = ROOT / "models/ssqr-verifier/ce_large_dense_4096.pt"
RETRIEVERS = ("CodeBERT", "BM25")
TOL = 1e-10
RELEVANCE_GUARD = contextvars.ContextVar("rene_confirmation_relevance_guard", default=False)
PROTECTED = ()  # Unrelated author-workspace registries are not part of this release.


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value, salt):
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()


def record(path):
    return {"path": Path(path).relative_to(ROOT).as_posix(), "sha256": sha(path)}


def seal(name, paths, **extra):
    write(OUT / name, {"created_utc": now(), "files": [record(p) for p in paths], **extra}, immutable=True)


def check_seal(name):
    p = read(OUT / name)
    for row in p["files"]:
        require(sha(ROOT / row["path"]) == row["sha256"], f"sealed artifact drift: {row['path']}")
    return p


def verify_design():
    p = read(OUT / "DESIGN.json")
    require(p["status"] == "FROZEN", "design not frozen")
    for row in p["bound_files"] + p["protected_files"]:
        # Training labels are fully hashed before entering inference; guarded inference
        # rechecks public inputs/source/models without opening even D relevance files.
        if RELEVANCE_GUARD.get() and "/private/" in row["path"]:
            continue
        require(sha(ROOT / row["path"]) == row["sha256"], f"frozen input drift: {row['path']}")
    return p


@contextlib.contextmanager
def deny_relevance_reads():
    """Process-local guard, not a claim of OS-level or external preregistration."""
    original_builtin = __import__("builtins").open
    original_io = io.open
    original_path = Path.open

    def checked(opener):
        def wrapped(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, bytes, Path)) and "r" in mode:
                name = str(file).replace("\\", "/").lower()
                forbidden = ("/private/" in name or "qrels" in name or "private_outcomes" in name)
                require(not forbidden, f"inference attempted a relevance read: {name}")
            return opener(file, mode, *args, **kwargs)
        return wrapped
    __import__("builtins").open = checked(original_builtin)
    io.open = checked(original_io)
    # Python3.10 pathlib caches its own accessor; patching io.open alone misses it.
    Path.open = checked(original_path)
    token = RELEVANCE_GUARD.set(True)
    try:
        yield
    finally:
        __import__("builtins").open = original_builtin
        io.open = original_io
        Path.open = original_path
        RELEVANCE_GUARD.reset(token)


def capped_ids(query_ids, groups, cap):
    pools = defaultdict(list)
    for q in query_ids:
        pools[groups[q]].append(q)
    return sorted(q for g in sorted(pools) for q in sorted(
        pools[g], key=lambda v: digest(v, "rene-confirmation-cap-20261019"))[:cap])


def status(stage, **kwargs):
    write(OUT / "RUN_STATUS.json", {"updated_utc": now(), "stage": stage,
                                    "confirmation_opened": (OUT / "OUTCOME_OPENING.json").exists(), **kwargs})
