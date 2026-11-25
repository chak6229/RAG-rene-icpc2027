from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.rene_cosqa_development_20261019.features import FEATURE_NAMES, feature_vector
from experiments.rene_cosqa_pilot_20261019.ranking import lexical_tokens

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / "results/rene_core_step_b_20261020"
PARENT = ROOT / "results/rene_cosqa_development_20261019"
CORPUS = ROOT / "results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl"
DESIGN = ROOT / 'ResearchVault/Handoffs/RENE- Core contribution validation program -20261020. md'
RETRIEVERS = ("CodeBERT", "BM25")
HEADS = ("M0", "MT", "MR", "MRT")
POLICIES = ("positive", "fixed20")
# Token counts are common text evidence. Probability/entropy-derived ranks stay T.
M0 = [4, 5] + list(range(8, 15)) + list(range(31, 43)) + list(range(65, 71))
T = [0, 1, 2, 3, 6, 7, 15, 16, 17, 18]
R = list(range(19, 31)) + list(range(43, 65))
BLOCKS = {"M0": sorted(M0), "MT": sorted(M0 + T),
          "MR": sorted(M0 + R), "MRT": sorted(M0 + T + R)}
PROTECTED = []  # Unrelated author-workspace registries are not part of this release.


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def readl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 23), b""):
            value.update(chunk)
    return value.hexdigest()


def record(path):
    return {"path": Path(path).relative_to(ROOT).as_posix(), "sha256": sha(path)}


def write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def writel(name, rows):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def verify():
    protocol = read(OUT / "PROTOCOL.json")
    require(protocol["role"] == "consumed_development_OOF_diagnostic", "Wrong role")
    require(protocol["new_confirmation"] is False, "Not a confirmation")
    for item in protocol["bound_files"] + protocol["protected_files"]:
        require(sha(ROOT / item["path"]) == item["sha256"], f'Frozen file drift: {item["path"]}')
    return protocol


@dataclass
class Features:
    retriever: str
    qids: list
    groups: np.ndarray
    folds: np.ndarray
    x: np.ndarray
    positions: list
    eligible: np.ndarray
    identity: np.ndarray
    same_ranking: np.ndarray
    originals: list
    candidates: list
    metadata: dict


def load_features(retriever):
    """No qrels/targets or outcome files are opened by this function."""
    qids = read(PARENT / "PROTOCOL.json")["query_ids"]
    roles = read(PARENT / "ROLE_MANIFEST.json")
    group_map = read(PARENT / "data/groups.json")
    folds = read(PARENT / "FOLDS.json")
    require(len(qids) == len(set(qids)) == 1490, "Unexpected development population")
    require(set(qids) == set(roles["development_query_ids"]) == set(group_map), "D role mismatch")
    require(not set(qids) & set(roles["confirmation_pool_query_ids"]), "H queries included")
    require(not set(roles["development_group_ids"]) & set(roles["confirmation_pool_group_ids"]), "D/H group overlap")
    require(set(group_map.values()) == set(roles["development_group_ids"]), "Group role mismatch")
    require(all(folds["query_to_fold"][q] == folds["group_to_fold"][group_map[q]] for q in qids), "Broken grouped folds")
    queries = {row["query_id"]: row["query"] for row in readl(PARENT / "data/queries.jsonl")}
    selections = {row["query_id"]: [v["position"] for v in sorted(row["selected"], key=lambda v: v["selected_rank"])]
                  for row in readl(PARENT / "selections.jsonl") if row["policy"] == "ENTR"}
    candidate_rows = readl(PARENT / "candidates.jsonl")
    candidate_map = {(r["query_id"], r["position"]): r for r in candidate_rows}
    require(len(candidate_map) == len(candidate_rows), "Duplicate candidate identity")
    feature_rows = readl(PARENT / f"features/{retriever}.jsonl")
    require(Counter(r["query_id"] for r in feature_rows) == Counter({q: 3 for q in qids}), "Not exact ENTR3 cache")
    feature_map = {(r["query_id"], r["position"]): r for r in feature_rows}
    rank_rows = readl(PARENT / f"rankings/{retriever}.jsonl")
    ranks = {(r["query_id"], r["position"]): r for r in rank_rows}
    require(len(ranks) == len(rank_rows), "Duplicate ranking action")
    code_words = {row["doc_id"]: set(lexical_tokens(row["code_text"])) for row in readl(CORPUS)}
    vectors, positions, originals, rewritten, identity, same = [], [], [], [], [], []
    max_feature_error = 0.0
    for q in qids:
        pool = [candidate_map[q, p] for p in selections[q]]
        all_positions = [candidate_map[q, p] for p in range(1, len(queries[q].split()) + 1)]
        expected = [r["position"] for r in sorted(all_positions, key=lambda r: (r["full_vocab_entropy"], r["position"]))[:3]]
        require(selections[q] == expected, "ENTR selection changed")
        candidate = pool[0]
        p = candidate["position"]
        cached = feature_map[q, p]
        require(cached["entr_rank"] == 1, "Not fixed ENTR Top1")
        original, action = ranks[q, None], ranks[q, p]
        require(original["text"] == queries[q] and action["text"] == candidate["rewrite"], "Ranking text mismatch")
        for row in (original, action):
            require(len(row["ranked_doc_ids"]) == len(set(row["ranked_doc_ids"])) == 100, "Top100 identity invalid")
        rebuilt = feature_vector(queries[q], candidate, original, action, pool, code_words)
        error = float(np.max(np.abs(rebuilt - np.asarray(cached["values"]))))
        max_feature_error = max(error, max_feature_error)
        require(error < 1e-12, "Feature reconstruction differs")
        is_identity = candidate["rewrite"] == queries[q]
        is_same = original["ranked_doc_ids"] == action["ranked_doc_ids"]
        require(is_identity == candidate["is_identity"] and is_same == cached["structural_zero"], "Identity/structural-zero mismatch")
        require(not is_identity or is_same, "Identical text ranks differ")
        vectors.append(cached["values"])
        positions.append(p)
        originals.append(original["ranked_doc_ids"])
        rewritten.append(action["ranked_doc_ids"])
        identity.append(is_identity)
        same.append(is_same)
    x = np.asarray(vectors, dtype=np.float64)
    require(x.shape == (1490, 71) and np.isfinite(x).all(), "Feature shape/nonfinite")
    require(sorted(M0 + T + R) == list(range(71)) and len(set(M0 + T + R)) == 71, "Feature blocks must partition schema")
    eligible = ~(np.asarray(identity) | np.asarray(same))
    fold_array = np.asarray([folds["query_to_fold"][q] for q in qids])
    return Features(retriever, qids, np.asarray([group_map[q] for q in qids]), fold_array, x, positions,
                    eligible, np.asarray(identity), np.asarray(same), originals, rewritten,
                    {"queries": len(qids), "groups": len(set(group_map.values())),
                     "eligible": int(eligible.sum()), "identity_text": int(sum(identity)),
                     "same_ranking": int(sum(same)), "max_feature_reconstruction_error": max_feature_error,
                     "folds": [{"fold": f, "queries": int((fold_array == f).sum()),
                                "eligible": int((eligible & (fold_array == f)).sum()),
                                "fixed20_adoptions": int((eligible & (fold_array == f)).sum()) // 5}
                               for f in range(5)]})


def load_outcomes(data):
    """Labels are separate from feature/action construction and decision functions."""
    positives = {q: set() for q in data.qids}
    seen = set()
    for row in readl(PARENT / "data/private/development_qrels.jsonl"):
        q, doc = row["query_id"], row["doc_id"]
        require(q in positives and (q, doc) not in seen, "Qrels outside D or duplicated")
        seen.add((q, doc))
        if row.get("relevance", 1) > 0:
            positives[q].add(doc)
    require(all(positives.values()), "Missing positive relevance")
    def rr(order, relevant):
        return next((1.0 / k for k, doc in enumerate(order, 1) if doc in relevant), 0.0)
    baseline = np.asarray([rr(data.originals[i], positives[q]) for i, q in enumerate(data.qids)])
    candidate = np.asarray([rr(data.candidates[i], positives[q]) for i, q in enumerate(data.qids)])
    y = candidate - baseline
    require(not np.any(np.abs(y[data.same_ranking]) > 0), "Same ranking must have zero target")
    return baseline, candidate, y
