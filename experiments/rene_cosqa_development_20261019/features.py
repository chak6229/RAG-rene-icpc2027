"""Deterministic features from the fixed ENTR pool and label-free rankings only."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from experiments.rene_cosqa_pilot_20261019.ranking import lexical_tokens
from experiments.rene_cosqa_pilot_20261019.evaluation import validate_inputs
from .common import ADAPTER, OUT, read, readl, require, sha, verify_frozen, write, writel

GEN_NAMES = (
    "full_vocab_entropy", "label_mean_log_probability", "label_sum_log_probability",
    "mean_max_vocab_probability", "label_token_count", "span_token_count",
    "span_full_vocab_entropy", "span_mean_log_probability", "span_empty",
    "position_fraction", "log_query_words", "log_span_words", "span_novel_fraction",
    "rewrite_word_ratio", "identity_text", "entr_rank_fraction", "prob_rank_fraction",
    "entropy_above_pool_min", "logp_below_pool_max",
)
SCORE_NAMES = ("top1", "top2", "top3", "top10", "top100", "margin12", "margin1_10",
               "mean10", "std10", "std100", "standardized_margin12", "entropy100")
REL_NAMES = (
    "same_ranking", "same_top1", "jaccard3", "jaccard10", "jaccard100",
    "original_top1_rr_in_candidate", "candidate_top1_rr_in_original",
    "original_top3_mean_rr_in_candidate", "rank_biased_overlap20",
    "original_query_coverage_original_top1", "original_query_coverage_candidate_top1",
    "original_query_coverage_candidate_mean3", "rewrite_coverage_candidate_top1",
    "span_coverage_candidate_top1", "span_coverage_candidate_max3", "log_candidate_top1_code_words",
)
FEATURE_NAMES = (GEN_NAMES + tuple(f"original_{name}" for name in SCORE_NAMES)
                 + tuple(f"candidate_{name}" for name in SCORE_NAMES)
                 + tuple(f"difference_{name}" for name in SCORE_NAMES) + REL_NAMES)


def score_features(scores):
    x = np.asarray(scores, dtype=np.float64)
    require(x.shape == (100,) and np.isfinite(x).all(), "100 finite retrieval scores required")
    sd = float(x.std())
    z = (x - x.max()) / max(sd, 1e-8)
    probability = np.exp(z - np.log(np.exp(z).sum()))
    return np.array([x[0], x[1], x[2], x[9], x[99], x[0] - x[1], x[0] - x[9],
                     x[:10].mean(), x[:10].std(), sd, (x[0] - x[1]) / max(sd, 1e-8),
                     -float(np.sum(probability * np.log(np.maximum(probability, 1e-300))))], dtype=float)


def coverage(words, code_words):
    return len(words & code_words) / max(len(words), 1)


def feature_vector(query, candidate, original, rewritten, pool, code_words):
    """No qrels, relevance, RR, repository identity, or target document argument."""
    ordered_e = sorted(pool, key=lambda r: (r["full_vocab_entropy"], r["position"]))
    ordered_p = sorted(pool, key=lambda r: (-r["label_mean_log_probability"], r["position"]))
    positions_e = [r["position"] for r in ordered_e]
    positions_p = [r["position"] for r in ordered_p]
    qwords, swords = set(lexical_tokens(query)), set(lexical_tokens(candidate["span"]))
    rwords = set(lexical_tokens(candidate["rewrite"]))
    qcount, scount = len(query.split()), len(candidate["span"].split())
    gen = [float(candidate[name]) for name in GEN_NAMES[:6]]
    gen += [float(candidate["span_full_vocab_entropy"] or 0), float(candidate["span_mean_log_probability"] or 0),
            float(not candidate["span"]), candidate["position"] / max(qcount, 1), math.log1p(qcount),
            math.log1p(scount), len(swords - qwords) / max(len(swords), 1),
            len(candidate["rewrite"].split()) / max(qcount, 1), float(candidate["is_identity"]),
            positions_e.index(candidate["position"]) / max(len(pool) - 1, 1),
            positions_p.index(candidate["position"]) / max(len(pool) - 1, 1),
            candidate["full_vocab_entropy"] - ordered_e[0]["full_vocab_entropy"],
            ordered_p[0]["label_mean_log_probability"] - candidate["label_mean_log_probability"]]
    original_scores, candidate_scores = score_features(original["scores"]), score_features(rewritten["scores"])
    old, new = original["ranked_doc_ids"], rewritten["ranked_doc_ids"]
    old_rank, new_rank = {doc: i for i, doc in enumerate(old, 1)}, {doc: i for i, doc in enumerate(new, 1)}
    reciprocal = lambda rank, doc: 1 / rank[doc] if doc in rank else 0.0
    relative = [float(old == new), float(old[0] == new[0])]
    relative += [len(set(old[:k]) & set(new[:k])) / len(set(old[:k]) | set(new[:k])) for k in (3, 10, 100)]
    rbo = sum(.1 * .9 ** (k - 1) * len(set(old[:k]) & set(new[:k])) / k for k in range(1, 21))
    relative += [reciprocal(new_rank, old[0]), reciprocal(old_rank, new[0]),
                 np.mean([reciprocal(new_rank, doc) for doc in old[:3]]), rbo,
                 coverage(qwords, code_words[old[0]]), coverage(qwords, code_words[new[0]]),
                 np.mean([coverage(qwords, code_words[doc]) for doc in new[:3]]),
                 coverage(rwords, code_words[new[0]]), coverage(swords, code_words[new[0]]),
                 max(coverage(swords, code_words[doc]) for doc in new[:3]), math.log1p(len(code_words[new[0]]))]
    vector = np.asarray(gen + original_scores.tolist() + candidate_scores.tolist()
                        + (candidate_scores - original_scores).tolist() + relative, dtype=np.float64)
    require(vector.shape == (len(FEATURE_NAMES),) and np.isfinite(vector).all(), "feature schema/nonfinite mismatch")
    return vector


def build(retriever):
    protocol = verify_frozen()
    destination = OUT / f"features/{retriever}.jsonl"
    manifest_path = destination.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = read(manifest_path)
        require(sha(destination) == manifest["sha256"], "feature cache changed")
        return
    queries = readl(OUT / "data/queries.jsonl")
    corpus = readl(ADAPTER / "public/corpus.jsonl")
    candidates = readl(OUT / "candidates.jsonl")
    rankings = readl(OUT / f"rankings/{retriever}.jsonl")
    selections = readl(OUT / "selections.jsonl")
    _, actions, selected = validate_inputs(queries, corpus, candidates, rankings, selections,
                                         read(OUT / "data/groups.json"), protocol)
    candidate_map = {(row["query_id"], row["position"]): row for row in candidates}
    code_words = {row["doc_id"]: set(lexical_tokens(row["code_text"])) for row in corpus}
    rows = []
    for query in queries:
        qid = query["query_id"]
        pool = [candidate_map[qid, p] for p in selected[qid, "ENTR", None]]
        for rank, candidate in enumerate(pool, 1):
            position = candidate["position"]
            row = {"query_id": qid, "position": position, "entr_rank": rank,
                   "values": feature_vector(query["query"], candidate, actions[qid, None], actions[qid, position], pool, code_words).tolist(),
                   "structural_zero": actions[qid, None]["ranked_doc_ids"] == actions[qid, position]["ranked_doc_ids"]}
            rows.append(row)
    writel(destination, rows)
    write(manifest_path, {"sha256": sha(destination), "rows": len(rows), "feature_names": FEATURE_NAMES,
                          "generation_features": len(GEN_NAMES), "qrels_read": False,
                          "candidate_pool": "exact min(3,N) ENTR selection", "other_position_retrieval_features": False}, immutable=True)
    print({"features": retriever, "rows": len(rows), "columns": len(FEATURE_NAMES)}, flush=True)
