"""One label-free retrieval per unique text, shared by all position selections."""
from __future__ import annotations

import argparse
import gc
import math
import re
import time
from collections import Counter, defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from .common import ADAPTER, CODEBERT, OUT, T5, now, read, readl, sha, verify_frozen, write, writel
from .selection import selection_rows


def lexical_tokens(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z][a-z0-9]*", text.lower())


class BM25:
    def __init__(self, texts, k1=1.2, b=.75):
        self.n = len(texts)
        self.k1, self.b = k1, b
        counts = [Counter(lexical_tokens(text)) for text in texts]
        lengths = np.array([sum(count.values()) for count in counts], dtype=np.float64)
        self.average_length = max(float(lengths.mean()), 1.0)
        postings = defaultdict(list)
        for index, count in enumerate(counts):
            for word, frequency in count.items():
                postings[word].append((index, frequency))
        self.postings = {}
        for word, items in postings.items():
            indices = np.array([item[0] for item in items], dtype=int)
            tf = np.array([item[1] for item in items], dtype=np.float64)
            idf = math.log1p((self.n - len(indices) + .5) / (len(indices) + .5))
            weights = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[indices] / self.average_length))
            self.postings[word] = indices, weights

    def score(self, text):
        scores = np.zeros(self.n, dtype=np.float64)
        for word in sorted(set(lexical_tokens(text))):
            if word in self.postings:
                indices, weights = self.postings[word]
                scores[indices] += weights
        return scores


def generate():
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import GenerationSettings, generate_candidates, load_tokenizer
    from experiments.rene_ssqr_alignment_20261019.pipeline import require_development_roles

    protocol = verify_frozen()
    if (OUT / "CANDIDATES_SEALED.json").exists():
        seal = read(OUT / "CANDIDATES_SEALED.json")
        if sha(OUT / "candidates.jsonl") != seal["candidate_sha256"]:
            raise ValueError("candidate cache changed")
        return seal
    queries = readl(OUT / "data/queries.jsonl")
    require_development_roles(queries, OUT / "data/PILOT_ROLE.json")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = load_tokenizer(T5)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5, local_files_only=True).float().cuda().eval()
    settings = GenerationSettings(**protocol["generation"])
    partial = OUT / "candidates.partial.jsonl"
    completed = readl(partial) if partial.exists() else []
    completed_ids = {row["query_id"] for row in completed}
    if not completed_ids <= set(protocol["query_ids"]):
        raise ValueError("partial candidate cache contains other queries")
    started = time.perf_counter()
    with partial.open("a", encoding="utf-8") as handle:
        import json
        for index, query in enumerate(queries):
            if query["query_id"] in completed_ids:
                continue
            rows = generate_candidates(model, tokenizer, query["query_id"], query["query"], settings)
            handle.write("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
            handle.flush()
            if (index + 1) % 16 == 0 or index + 1 == len(queries):
                print({"stage": "generation", "queries_done": index + 1, "queries_total": len(queries),
                       "seconds_this_session": round(time.perf_counter() - started, 1)}, flush=True)
    if (OUT / "candidates.jsonl").exists():
        raise ValueError("refusing to replace an existing completed cache")
    partial.rename(OUT / "candidates.jsonl")
    candidates = readl(OUT / "candidates.jsonl")
    writel(OUT / "selections.jsonl", selection_rows(candidates, protocol["query_ids"], protocol))
    seal = {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
            "candidate_sha256": sha(OUT / "candidates.jsonl"), "selection_sha256": sha(OUT / "selections.jsonl"),
            "queries": len(queries), "candidates": len(candidates), "qrels_read": False,
            "generated_once_per_position": True, "generation_session_seconds": time.perf_counter() - started}
    write(OUT / "CANDIDATES_SEALED.json", seal, immutable=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return seal


def all_actions(queries, candidates):
    pools = defaultdict(list)
    for row in candidates:
        pools[row["query_id"]].append(row)
    actions = []
    for query in queries:
        actions.append({"query_id": query["query_id"], "policy": "NoRewrite", "position": None, "text": query["query"]})
        actions.extend({"query_id": query["query_id"], "policy": "Candidate", "position": row["position"], "text": row["rewrite"]}
                       for row in sorted(pools[query["query_id"]], key=lambda row: row["position"]))
    return actions


def rank(retriever):
    protocol = verify_frozen()
    candidate_seal = read(OUT / "CANDIDATES_SEALED.json")
    if sha(OUT / "candidates.jsonl") != candidate_seal["candidate_sha256"]:
        raise ValueError("candidate seal mismatch")
    destination = OUT / "rankings" / f"{retriever}.jsonl"
    seal_file = destination.with_suffix(".manifest.json")
    if seal_file.exists():
        if sha(destination) != read(seal_file)["rankings_sha256"]:
            raise ValueError("stored rankings changed")
        return read(seal_file)
    queries = readl(OUT / "data/queries.jsonl")
    corpus = readl(ADAPTER / "public/corpus.jsonl")
    actions = all_actions(queries, readl(OUT / "candidates.jsonl"))
    texts = list(dict.fromkeys(row["text"] for row in actions))
    text_index = {text: index for index, text in enumerate(texts)}
    doc_ids = np.array([row["doc_id"] for row in corpus], dtype=np.int64)
    artifacts = OUT / "retrieval_cache" / retriever
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
        print({"stage": "CodeBERT", "documents": len(corpus), "unique_query_texts": len(texts)}, flush=True)
        code_vectors = encode_texts(model, tokenizer, [row["code_text"] for row in corpus], batch_size=16, device="cuda")
        query_vectors = encode_texts(model, tokenizer, texts, batch_size=16, device="cuda")
        np.save(artifacts / "code_vectors.npy", code_vectors, allow_pickle=False)
        np.save(artifacts / "query_vectors.npy", query_vectors, allow_pickle=False)
        with threadpool_limits(limits=4):
            scores = query_vectors @ code_vectors.T
        del model
        gc.collect()
        torch.cuda.empty_cache()
    elif retriever == "BM25":
        index = BM25([row["code_text"] for row in corpus], **protocol["bm25"]["parameters"])
        scores = np.stack([index.score(text) for text in texts])
    else:
        raise ValueError("unknown retriever")
    np.save(artifacts / "scores.npy", scores, allow_pickle=False)
    write(artifacts / "texts.json", texts, immutable=True)
    orders = [np.lexsort((doc_ids, -row))[:min(protocol["ranking_depth"], len(corpus))] for row in scores]
    records = []
    for action in actions:
        index = text_index[action["text"]]
        order = orders[index]
        records.append({**action, "ranked_doc_ids": doc_ids[order].tolist(), "scores": scores[index, order].astype(float).tolist()})
    writel(destination, records)
    manifest = {"retriever": retriever, "created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
                "rankings_sha256": sha(destination), "candidate_sha256": sha(OUT / "candidates.jsonl"),
                "unique_query_texts": len(texts), "action_rows": len(actions), "corpus_documents": len(corpus),
                "qrels_read": False, "metrics_computed": False, "tie_rule": "score_desc_doc_id_asc",
                "elapsed_seconds": time.perf_counter() - started,
                "cache_files": [{"path": str(path.relative_to(OUT)).replace("\\", "/"), "sha256": sha(path)}
                                for path in sorted(artifacts.iterdir()) if path.is_file()]}
    write(seal_file, manifest, immutable=True)
    print({"stage": "ranked", "retriever": retriever, "seconds": round(manifest["elapsed_seconds"], 1), "rows": len(records)}, flush=True)
    return manifest


def seal_decisions():
    verify_frozen()
    if (OUT / "DECISIONS_SEALED.json").exists():
        return read(OUT / "DECISIONS_SEALED.json")
    paths = [OUT / "candidates.jsonl", OUT / "selections.jsonl", OUT / "CANDIDATES_SEALED.json"]
    for retriever in ("CodeBERT", "BM25"):
        manifest_path = OUT / f"rankings/{retriever}.manifest.json"
        manifest = read(manifest_path)
        paths.extend([manifest_path, OUT / f"rankings/{retriever}.jsonl"])
        paths.extend(OUT / row["path"] for row in manifest["cache_files"])
    seal = {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"), "role": "development",
            "effects_not_yet_opened": not (OUT / "metrics").exists(), "qrels_read_by_inference": False,
            "files": [{"path": path.relative_to(OUT).as_posix(), "sha256": sha(path)} for path in paths]}
    if not seal["effects_not_yet_opened"]:
        raise ValueError("effect artifacts exist before decisions seal")
    write(OUT / "DECISIONS_SEALED.json", seal, immutable=True)
    return seal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "CodeBERT", "BM25", "seal"))
    args = parser.parse_args()
    result = generate() if args.mode == "generate" else seal_decisions() if args.mode == "seal" else rank(args.mode)
    print({"mode": args.mode, "done": True, "created_utc": result["created_utc"]}, flush=True)


if __name__ == "__main__":
    main()
