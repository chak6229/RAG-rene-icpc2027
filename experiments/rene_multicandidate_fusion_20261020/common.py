from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / "results/rene_multicandidate_fusion_20261020"
PARENT = ROOT / "results/rene_cosqa_development_20261019"
PLAN = ROOT / 'ResearchVault/Handoffs/ RENE - Validation of core contribution priority after AB -20261020.md'
RETRIEVERS = ("CodeBERT", "BM25")
POLICIES = ("NoRewrite", "ENTRTop1", "RRF3", "RRF4")
OFFSET, DEPTH = 60, 100
DENOMINATOR = math.lcm(*range(OFFSET + 1, OFFSET + DEPTH + 1))
RANK_WEIGHT = tuple(DENOMINATOR // (OFFSET + r) for r in range(1, DEPTH + 1))
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
        for block in iter(lambda: handle.read(2**23), b""):
            value.update(block)
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
    p = read(OUT / "PROTOCOL.json")
    require(p["role"] == "consumed_development_post_hoc_extension" and not p["new_confirmation"], "Invalid role")
    for item in p["bound_files"] + p["protected_files"]:
        require(sha(ROOT / item["path"]) == item["sha256"], f'Frozen file changed: {item["path"]}')
    return p


def unique_text_lists(rows):
    """Collapse duplicate retrieval INPUT text, never duplicate outcomes or ranks."""
    seen, unique = {}, []
    for row in rows:
        text = row["text"]
        if text in seen:
            require(row["ranked_doc_ids"] == seen[text]["ranked_doc_ids"], "Identical inputs have inconsistent ranking")
            require(row["scores"] == seen[text]["scores"], "Identical inputs have inconsistent score")
        else:
            seen[text] = row
            unique.append(row)
    return unique


def fuse(rows):
    """Exact rational RRF via a shared integer denominator, doc-ID tie break."""
    unique = unique_text_lists(rows)
    require(bool(unique), "Cannot fuse empty inputs")
    totals = Counter()
    for row in unique:
        order = row["ranked_doc_ids"]
        require(len(order) == len(set(order)) == DEPTH, "Fusion input is not distinct Top100")
        for i, doc in enumerate(order):
            require(type(doc) is int, "Integer document identities required")
            totals[doc] += RANK_WEIGHT[i]
    ranked = sorted(totals, key=lambda doc: (-totals[doc], doc))[:DEPTH]
    counts = Counter(totals.values())
    cutoff = totals[ranked[-1]]
    return ranked, {"unique_lists": len(unique), "union_documents": len(totals),
                    "score_tie_groups": sum(n > 1 for n in counts.values()),
                    "documents_in_score_ties": sum(n for n in counts.values() if n > 1),
                    "cutoff_tie_documents": counts[cutoff]}


def load_public(retriever):
    """Only public query/candidate/selection/ranking data; no relevance labels."""
    qids = read(PARENT / "PROTOCOL.json")["query_ids"]
    queries = {r["query_id"]: r["query"] for r in readl(PARENT / "data/queries.jsonl")}
    groups = read(PARENT / "data/groups.json")
    roles = read(PARENT / "ROLE_MANIFEST.json")
    require(len(qids) == len(set(qids)) == 1490 and len(set(groups.values())) == 512, "Wrong population")
    require(set(qids) == set(queries) == set(groups) == set(roles["development_query_ids"]), "D population differs")
    require(not set(qids) & set(roles["confirmation_pool_query_ids"]), "H queries included")
    require(not set(groups.values()) & set(roles["confirmation_pool_group_ids"]), "H groups included")
    candidates = readl(PARENT / "candidates.jsonl")
    candidate_map = {(r["query_id"], r["position"]): r for r in candidates}
    require(len(candidate_map) == len(candidates) == 9915, "Candidate pool changed")
    selection = {r["query_id"]: [v["position"] for v in sorted(r["selected"], key=lambda v: v["selected_rank"])]
                 for r in readl(PARENT / "selections.jsonl") if r["policy"] == "ENTR"}
    require(set(selection) == set(qids), "Missing selection")
    rank_rows = readl(PARENT / f"rankings/{retriever}.jsonl")
    ranks = {(r["query_id"], r["position"]): r for r in rank_rows}
    require(len(ranks) == len(rank_rows), "Duplicate query-position ranking")
    pools, counts = {}, {}
    corpus_ids = {r["doc_id"] for r in readl(ROOT / "results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl")}
    for q in qids:
        all_candidates = [candidate_map[q, p] for p in range(1, len(queries[q].split())+1)]
        ordered = sorted(all_candidates, key=lambda r: (r["full_vocab_entropy"], r["position"]))[:3]
        require(selection[q] == [r["position"] for r in ordered] and len(ordered) == 3, "Not exact ENTR3")
        require(ranks[q, None]["text"] == queries[q], "Original input differs")
        pool = [ranks[q, r["position"]] for r in ordered]
        for candidate, row in zip(ordered, pool):
            require(row["text"] == candidate["rewrite"], "Candidate input text differs")
        for row in [ranks[q, None]] + pool:
            require(len(row["ranked_doc_ids"]) == len(set(row["ranked_doc_ids"])) == 100, "Invalid Top100")
            require(set(row["ranked_doc_ids"]) <= corpus_ids, "Outside corpus")
        pools[q] = [ranks[q, None]] + pool
        counts[q] = len(all_candidates)
    return qids, groups, selection, pools, counts
