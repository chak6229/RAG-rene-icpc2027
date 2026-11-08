from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from experiments.rene_cosqa_development_20261019.common import (
    ROOT, OUT as PARENT, now, read, readl, require, sha, write, writel,
)
from experiments.rene_cosqa_development_20261019.features import FEATURE_NAMES
from experiments.rene_cosqa_pilot_20261019.evaluation import rr

OUT = ROOT / "results/rene_postselection_20261019"
SOURCE = ROOT / "experiments/rene_postselection_20261019"
BLOCKS = {
    "GEN": list(range(19)),
    "GS": list(range(55)),
    "GSA": list(range(64)),
    "GSL": list(range(55)) + list(range(64, 71)),
    "FULL": list(range(71)),
}
ALGORITHMS = ("HGB", "RIDGE")
CALIBRATORS = ("AFFINE", "ISOTONIC", "CONTEXT_RIDGE")


def record(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def verify():
    p = read(OUT / "PROTOCOL.json")
    require(p["status"] == "FROZEN" and p["role"] == "consumed_development_only", "wrong experiment role")
    require(not p["confirmation_authorized"], "confirmation is not authorized")
    for row in p["bound_files"] + p["protected_files"]:
        require(sha(ROOT / row["path"]) == row["sha256"], f'input/source drift: {row["path"]}')
    return p


@dataclass
class Data:
    retriever: str
    qids: list
    rows: list
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    folds: np.ndarray
    query_rows: dict
    actions: dict
    positives: dict
    group_map: dict
    fold_map: dict


def load_data(retriever):
    p = read(PARENT / "PROTOCOL.json")
    qids = p["query_ids"]
    rows = readl(PARENT / f"features/{retriever}.jsonl")
    require(Counter(r["query_id"] for r in rows) == Counter({q: 3 for q in qids}), "not the exact ENTR3 pool")
    groups = read(PARENT / "data/groups.json")
    folds = read(PARENT / "FOLDS.json")["query_to_fold"]
    roles = read(PARENT / "ROLE_MANIFEST.json")
    require(set(qids) == set(roles["development_query_ids"]), "development denominator changed")
    require(not set(qids) & set(roles["confirmation_pool_query_ids"]), "confirmation included")
    require(not set(roles["development_group_ids"]) & set(roles["confirmation_pool_group_ids"]), "D/H group overlap")
    actions = {(r["query_id"], r["position"]): r for r in readl(PARENT / f"rankings/{retriever}.jsonl")}
    positives = {q: set() for q in qids}
    seen = set()
    for r in readl(PARENT / "data/private/development_qrels.jsonl"):
        key = r["query_id"], r["doc_id"]
        require(r["query_id"] in positives and key not in seen, "qrels outside D or duplicate")
        seen.add(key)
        if r.get("relevance", 1) > 0:
            positives[r["query_id"]].add(r["doc_id"])
    require(all(positives.values()), "missing D relevance")
    query_rows = defaultdict(list)
    for i, r in enumerate(rows):
        query_rows[r["query_id"]].append(i)
        require(r["structural_zero"] == (actions[r["query_id"], r["position"]]["ranked_doc_ids"]
                                        == actions[r["query_id"], None]["ranked_doc_ids"]), "structural zero differs")
    selections = {r["query_id"]: [v["position"] for v in sorted(r["selected"], key=lambda v: v["selected_rank"])]
                  for r in readl(PARENT / "selections.jsonl") if r["policy"] == "ENTR"}
    for q in qids:
        require([rows[i]["position"] for i in query_rows[q]] == selections[q], "candidate pool/order differs")
    x = np.asarray([r["values"] for r in rows], dtype=float)
    require(x.shape == (3 * len(qids), len(FEATURE_NAMES)) and np.isfinite(x).all(), "bad feature cache")
    y = np.asarray([rr(actions[r["query_id"], r["position"]]["ranked_doc_ids"], positives[r["query_id"]], 100)
                    - rr(actions[r["query_id"], None]["ranked_doc_ids"], positives[r["query_id"]], 100) for r in rows])
    return Data(retriever, qids, rows, x, y, np.asarray([groups[r["query_id"]] for r in rows]),
                np.asarray([folds[r["query_id"]] for r in rows]), dict(query_rows), actions, positives, groups, folds)


def parent_predictions(data, algorithm, block="FULL"):
    name = ("GEN_HGB" if block == "GEN" else f"GAIN_{algorithm}")
    old = {(r["query_id"], r["position"]): r for r in readl(PARENT / f"oof_predictions/{data.retriever}.jsonl")}
    return np.asarray([old[r["query_id"], r["position"]]["predictions"][name]["raw_clipped"] for r in data.rows])
