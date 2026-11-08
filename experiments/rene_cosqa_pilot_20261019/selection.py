"""Selection is independent of retrieval outcomes and uses one position cache."""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict


def select(candidates, query_id, policy, k, seed=None):
    if len({row["position"] for row in candidates}) != len(candidates) or not candidates:
        raise ValueError("candidate positions must be nonempty and unique")
    if policy == "ENTR":
        rows = sorted(candidates, key=lambda row: (row["full_vocab_entropy"], row["position"]))
    elif policy == "PROB":
        rows = sorted(candidates, key=lambda row: (-row["label_mean_log_probability"], row["position"]))
    elif policy == "RAND":
        if type(seed) is not int:
            raise ValueError("RAND requires an explicit integer seed")
        digest = hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).digest()
        rows = random.Random(int.from_bytes(digest, "big")).sample(
            sorted(candidates, key=lambda row: row["position"]), len(candidates))
    else:
        raise ValueError("unknown selection policy")
    return [row["position"] for row in rows[:min(k, len(rows))]]


def selection_rows(candidates, query_ids, protocol):
    by_query = defaultdict(list)
    for row in candidates:
        by_query[row["query_id"]].append(row)
    rows = []
    for query_id in query_ids:
        for policy, seeds in (("ENTR", [None]), ("PROB", [None]), ("RAND", protocol["rand_seeds"])):
            for seed in seeds:
                positions = select(by_query[query_id], query_id, policy, protocol["candidate_k"], seed)
                rows.append({"query_id": query_id, "policy": policy, "repeat_seed": seed,
                             "selected": [{"position": position, "selected_rank": rank}
                                          for rank, position in enumerate(positions, 1)]})
    return rows
