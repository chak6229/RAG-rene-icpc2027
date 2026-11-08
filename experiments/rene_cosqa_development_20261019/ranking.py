"""Generate only missing D queries; preserve the pilot and reuse corpus encodings."""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.rene_cosqa_pilot_20261019.ranking import BM25, all_actions
from experiments.rene_cosqa_pilot_20261019.selection import selection_rows
from .common import (ADAPTER, CODEBERT, OUT, PILOT, T5, now, read, readl, require, sha,
                     verify_frozen, write, writel)


def generate():
    protocol = verify_frozen()
    if (OUT / "CANDIDATES_SEALED.json").exists():
        seal = read(OUT / "CANDIDATES_SEALED.json")
        require(sha(OUT / "candidates.jsonl") == seal["candidate_sha256"], "candidate seal changed")
        return
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import GenerationSettings, generate_candidates, load_tokenizer
    from experiments.rene_ssqr_alignment_20261019.pipeline import require_development_roles
    queries = readl(OUT / "data/queries.jsonl")
    require_development_roles(queries, OUT / "data/DEVELOPMENT_ROLE.json")
    new_ids = set(protocol["new_query_ids"])
    prior = readl(PILOT / "candidates.jsonl")
    require({r["query_id"] for r in prior} == set(protocol["pilot_query_ids"]), "pilot query cache mismatch")
    partial = OUT / "candidates.new.partial.jsonl"
    completed = readl(partial) if partial.exists() else []
    by_query = defaultdict(list)
    for row in completed:
        require(row["query_id"] in new_ids, "partial cache query outside remaining D")
        by_query[row["query_id"]].append(row)
    text_by_query = {q["query_id"]: q["query"] for q in queries}
    for qid, rows in by_query.items():
        require(sorted(r["position"] for r in rows) == list(range(1, len(text_by_query[qid].split()) + 1)),
                "incomplete partial query must be repaired before resuming; no silent regeneration")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = load_tokenizer(T5)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5, local_files_only=True).float().cuda().eval()
    settings = GenerationSettings(**protocol["generation"])
    started = time.perf_counter()
    generated_queries = 0
    with partial.open("a", encoding="utf-8") as handle:
        for query in queries:
            qid = query["query_id"]
            if qid not in new_ids or qid in by_query:
                continue
            rows = generate_candidates(model, tokenizer, qid, query["query"], settings)
            handle.write("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
            handle.flush()
            generated_queries += 1
            if generated_queries % 64 == 0 or generated_queries == len(new_ids) - len(by_query):
                print({"stage": "missing_generation", "new_queries_this_session": generated_queries,
                       "remaining_queries": len(new_ids), "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    all_new = readl(partial)
    require({r["query_id"] for r in all_new} == new_ids, "new candidate generation incomplete")
    pools = defaultdict(list)
    for row in prior + all_new:
        pools[row["query_id"]].append(row)
    combined = [row for q in protocol["query_ids"] for row in sorted(pools[q], key=lambda r: r["position"])]
    writel(OUT / "candidates.jsonl", combined)
    writel(OUT / "selections.jsonl", selection_rows(combined, protocol["query_ids"], protocol))
    write(OUT / "CANDIDATES_SEALED.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                                           "candidate_sha256": sha(OUT / "candidates.jsonl"),
                                           "selection_sha256": sha(OUT / "selections.jsonl"),
                                           "candidates": len(combined), "reused_candidates": len(prior), "new_candidates": len(all_new),
                                           "qrels_read": False, "new_queries_generated": len(new_ids),
                                           "generation_session_seconds": time.perf_counter() - started,
                                           "extra_seed_generations": 0}, immutable=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def rank(retriever):
    protocol = verify_frozen()
    seal = read(OUT / "CANDIDATES_SEALED.json")
    require(sha(OUT / "candidates.jsonl") == seal["candidate_sha256"], "candidate cache changed")
    destination = OUT / f"rankings/{retriever}.jsonl"
    manifest_path = destination.with_suffix(".manifest.json")
    if manifest_path.exists():
        require(sha(destination) == read(manifest_path)["rankings_sha256"], "ranking cache changed")
        return
    queries = readl(OUT / "data/queries.jsonl")
    corpus = readl(ADAPTER / "public/corpus.jsonl")
    actions = all_actions(queries, readl(OUT / "candidates.jsonl"))
    prior_cache = PILOT / f"retrieval_cache/{retriever}"
    old_texts = read(prior_cache / "texts.json")
    old_text_set = set(old_texts)
    missing = list(dict.fromkeys(r["text"] for r in actions if r["text"] not in old_text_set))
    texts = old_texts + missing
    index = {text: i for i, text in enumerate(texts)}
    require(len(index) == len(texts), "duplicate text-cache key")
    artifacts = OUT / f"retrieval_cache/{retriever}"
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if retriever == "CodeBERT":
        import torch
        from experiments.rene_ssqr_alignment_20261019.pipeline import load_trained_retriever
        from experiments.rene_ssqr_alignment_20261019.retriever import encode_texts
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, tokenizer = load_trained_retriever(CODEBERT, "cuda")
        code_vectors = np.load(prior_cache / "code_vectors.npy", allow_pickle=False)
        print({"stage": "CodeBERT_new_texts", "new_texts": len(missing), "reused_texts": len(old_texts),
               "new_code_encodings": 0}, flush=True)
        new_vectors = encode_texts(model, tokenizer, missing, batch_size=16, device="cuda")
        query_vectors = np.concatenate([np.load(prior_cache / "query_vectors.npy", allow_pickle=False), new_vectors])
        np.save(artifacts / "query_vectors.npy", query_vectors, allow_pickle=False)
        write(artifacts / "code_vectors_reference.json", {"path": str((prior_cache / "code_vectors.npy").relative_to(OUT.parents[1])).replace("\\", "/"),
                                                          "sha256": sha(prior_cache / "code_vectors.npy")}, immutable=True)
        with threadpool_limits(limits=4):
            new_scores = new_vectors @ code_vectors.T
        del model
        gc.collect()
        torch.cuda.empty_cache()
    elif retriever == "BM25":
        bm25 = BM25([r["code_text"] for r in corpus], **protocol["bm25"]["parameters"])
        new_scores = np.stack([bm25.score(text) for text in missing])
    else:
        raise ValueError("unknown retriever")
    scores = np.concatenate([np.load(prior_cache / "scores.npy", allow_pickle=False), new_scores])
    np.save(artifacts / "scores.npy", scores, allow_pickle=False)
    write(artifacts / "texts.json", texts, immutable=True)
    doc_ids = np.array([r["doc_id"] for r in corpus], dtype=np.int64)
    orders = [np.lexsort((doc_ids, -row))[:100] for row in scores]
    prior_rows = {(r["query_id"], r["position"]): r for r in readl(PILOT / f"rankings/{retriever}.jsonl")}
    records = []
    for action in actions:
        key = action["query_id"], action["position"]
        if key in prior_rows:
            row = prior_rows[key]
            require(row["text"] == action["text"], "reused action text mismatch")
            records.append(row)
        else:
            i = index[action["text"]]
            order = orders[i]
            records.append({**action, "ranked_doc_ids": doc_ids[order].tolist(), "scores": scores[i, order].astype(float).tolist()})
    writel(destination, records)
    write(manifest_path, {"retriever": retriever, "created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                          "rankings_sha256": sha(destination), "reused_actions": len(prior_rows),
                          "action_rows": len(records), "reused_texts": len(old_texts), "new_texts": len(missing),
                          "qrels_read": False, "metrics_computed": False, "new_code_encodings": 0,
                          "elapsed_seconds": time.perf_counter() - started,
                          "cache_files": [{"path": path.relative_to(OUT).as_posix(), "sha256": sha(path)}
                                          for path in sorted(artifacts.iterdir()) if path.is_file()]}, immutable=True)
    print({"ranked": retriever, "actions": len(records), "seconds": round(time.perf_counter() - started, 1)}, flush=True)


def seal():
    verify_frozen()
    require(not (OUT / "metrics").exists(), "cannot first seal new evidence after new effects opened")
    if (OUT / "DECISIONS_SEALED.json").exists():
        return
    paths = [OUT / name for name in ("candidates.jsonl", "selections.jsonl", "CANDIDATES_SEALED.json")]
    for name in ("CodeBERT", "BM25"):
        paths += [OUT / f"rankings/{name}.jsonl", OUT / f"rankings/{name}.manifest.json",
                  OUT / f"features/{name}.jsonl", OUT / f"features/{name}.manifest.json"]
        paths += [OUT / r["path"] for r in read(OUT / f"rankings/{name}.manifest.json")["cache_files"]]
    write(OUT / "DECISIONS_SEALED.json", {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                                         "new_effects_not_yet_opened": True, "pilot_effects_previously_known": True,
                                         "inference_read_qrels": False, "confirmation_opened": False,
                                         "files": [{"path": path.relative_to(OUT).as_posix(), "sha256": sha(path)} for path in paths]}, immutable=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "CodeBERT", "BM25", "features-CodeBERT", "features-BM25", "seal"))
    args = parser.parse_args()
    if args.mode == "generate":
        generate()
    elif args.mode == "seal":
        seal()
    elif args.mode.startswith("features-"):
        from .features import build
        build(args.mode.split("-", 1)[1])
    else:
        rank(args.mode)
