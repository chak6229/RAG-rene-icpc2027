"""Chunked full-corpus ranking; development baseline rows are reused byte-for-byte."""
from __future__ import annotations

import gc
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .common import (
    ROOT, OUT, ADAPTER, DEVELOPMENT, CODEBERT, now, read, readl, write, writel, record, sha,
    require, verify_frozen, verify_seal, deny_private_reads, text_id,
)


def stable_order(scores, doc_ids, depth=100):
    return np.lexsort((doc_ids, -np.asarray(scores)))[:depth]


def candidate_paths(domain):
    files = sorted((OUT / f"candidates/{domain}").glob("*.jsonl"))
    if domain == "cosqa":
        files.append(DEVELOPMENT / "candidates.jsonl")
    return files


def domain_texts(domain):
    query_path = OUT / "data/source/queries.jsonl" if domain == "source" else DEVELOPMENT / "data/queries.jsonl"
    texts = [q["query"] for q in readl(query_path)]
    for path in candidate_paths(domain):
        texts.extend(r["rewrite"] for r in readl(path))
    return list(dict.fromkeys(texts))


def rank():
    protocol = verify_frozen()
    verify_seal("CANDIDATES_SEALED.json")
    if (OUT / "RANKINGS_SEALED.json").exists():
        return verify_seal("RANKINGS_SEALED.json")
    from experiments.rene_cosqa_pilot_20261019.ranking import BM25
    import torch
    from experiments.rene_ssqr_alignment_20261019.pipeline import load_trained_retriever
    from experiments.rene_ssqr_alignment_20261019.retriever import encode_texts
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths, manifests = [], []
    with deny_private_reads():
        for domain in ("source", "cosqa"):
            corpus_path = OUT / "data/source/corpus.jsonl" if domain == "source" else ADAPTER / "public/corpus.jsonl"
            corpus = readl(corpus_path)
            require(len(corpus) == protocol["samples"][domain]["corpus_documents"], "corpus scope mismatch")
            doc_ids = np.array([r["doc_id"] for r in corpus], dtype=np.int64)
            texts = domain_texts(domain)
            require(len({text_id(text) for text in texts}) == len(texts), "text hash collision")
            for retriever in protocol["retrievers"]:
                destination = OUT / f"rankings/{domain}/{retriever}.jsonl"
                manifest_path = destination.with_suffix(".manifest.json")
                paths += [destination, manifest_path]
                if manifest_path.exists():
                    manifest = read(manifest_path)
                    require(sha(destination) == manifest["ranking_sha256"], "completed ranking cache changed")
                    manifests.append(manifest)
                    continue
                require(not destination.exists(), "unsealed completed ranking output exists")
                reused = {}
                if domain == "cosqa":
                    for row in readl(DEVELOPMENT / f"rankings/{retriever}.jsonl"):
                        key = text_id(row["text"])
                        if key in reused:
                            require(reused[key]["ranked_doc_ids"] == row["ranked_doc_ids"], "parent duplicate text has inconsistent rankings")
                        reused[key] = {"text_id": key, "text": row["text"], "ranked_doc_ids": row["ranked_doc_ids"],
                                       "scores": row["scores"], "origin": "frozen_development_cache"}
                new_texts = [text for text in texts if text_id(text) not in reused]
                artifacts = OUT / f"retrieval_cache/{domain}/{retriever}"
                artifacts.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                print({"stage": "ranking", "domain": domain, "retriever": retriever,
                       "documents": len(corpus), "unique_texts": len(texts), "new_texts": len(new_texts)}, flush=True)
                if retriever == "CodeBERT":
                    model, tokenizer = load_trained_retriever(CODEBERT, "cuda")
                    code_path = artifacts / "code_vectors.npy"
                    if domain == "cosqa":
                        code_vectors = np.load(ROOT / "results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy")
                    elif code_path.exists():
                        code_vectors = np.load(code_path)
                    else:
                        code_vectors = encode_texts(model, tokenizer, [r["code_text"] for r in corpus], batch_size=16, device="cuda")
                        np.save(code_path, code_vectors, allow_pickle=False)
                    require(code_vectors.shape == (len(corpus), 768), "wrong code vector/corpus mapping")
                    vector_path = artifacts / "new_query_vectors.npy"
                    texts_path = artifacts / "new_texts.json"
                    if vector_path.exists() and texts_path.exists():
                        require(read(texts_path) == new_texts, "partial vector cache text order differs")
                        query_vectors = np.load(vector_path)
                    else:
                        query_vectors = encode_texts(model, tokenizer, new_texts, batch_size=16, device="cuda")
                        np.save(vector_path, query_vectors, allow_pickle=False)
                        write(texts_path, new_texts)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    require(query_vectors.shape == (len(new_texts), 768), "query vector coverage mismatch")
                else:
                    bm25 = BM25([r["code_text"] for r in corpus], k1=1.2, b=.75)
                new_records = {}
                with threadpool_limits(limits=4):
                    for offset in range(0, len(new_texts), 128):
                        current = new_texts[offset:offset+128]
                        scores = query_vectors[offset:offset+len(current)] @ code_vectors.T if retriever == "CodeBERT" else np.stack([bm25.score(text) for text in current])
                        for text, values in zip(current, scores):
                            order = stable_order(values, doc_ids)
                            key = text_id(text)
                            new_records[key] = {"text_id": key, "text": text, "ranked_doc_ids": doc_ids[order].tolist(),
                                                "scores": values[order].astype(float).tolist(), "origin": "new_label_free_ranking"}
                merged = {**reused, **new_records}
                writel(destination, [merged[text_id(text)] for text in texts])
                manifest = {"created_utc": now(), "domain": domain, "retriever": retriever,
                            "ranking_sha256": sha(destination), "corpus_sha256": sha(corpus_path),
                            "protocol_sha256": sha(OUT / "PROTOCOL.json"), "unique_texts": len(texts),
                            "reused_texts": len(texts)-len(new_texts), "new_texts": len(new_texts),
                            "corpus_documents": len(corpus), "seconds": time.perf_counter()-started,
                            "qrels_read": False, "whole_score_matrix_persisted": False,
                            "artifact_files": [record(p) for p in sorted(artifacts.iterdir()) if p.is_file()]}
                write(manifest_path, manifest)
                manifests.append(manifest)
                print({"stage": "ranked", "domain": domain, "retriever": retriever,
                       "seconds": round(manifest["seconds"],1)}, flush=True)
    seal = {"created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
            "files": [record(p) for p in paths], "qrels_read": False,
            "effects_not_opened": not (OUT / "metrics").exists(), "jobs": manifests}
    require(seal["effects_not_opened"], "new effects opened before ranking seal")
    write(OUT / "RANKINGS_SEALED.json", seal)
    return seal


if __name__ == "__main__":
    rank()
