"""Real single-generation retrieval/CE evidence, with no relevance dependency."""
from __future__ import annotations

import gc
import json
import time
from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.rene_cosqa_pilot_20261019.ranking import BM25, all_actions
from experiments.rene_cosqa_pilot_20261019.selection import selection_rows
from experiments.rene_cosqa_development_20261019.features import feature_vector, FEATURE_NAMES
from experiments.rene_cosqa_pilot_20261019.ranking import lexical_tokens
from .common import (ADAPTER, CE, CODEBERT, OUT, PARENT, ROOT, T5, check_seal, now, read, readl,
                     require, seal, sha, status, verify_design, write, writel)


def cuda_config():
    import torch
    require(torch.cuda.is_available(), "GPU runtime required; use .venv-models/Scripts/python.exe")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def score_ce(items_by_retriever, destination, *, all_eligible=False):
    """Original user query judges code; never score rewrite text against a gold target."""
    from .policy import allocation
    from transformers import AutoModel, AutoTokenizer
    import torch
    cuda_config()
    corpus = {r["doc_id"]: r["code_text"] for r in readl(ADAPTER / "public/corpus.jsonl")}
    queries = {r["query_id"]: r["query"] for r in readl(ADAPTER / "public/all_positive_queries.jsonl")}
    needed = set()
    for items in items_by_retriever.values():
        chosen = {q for q, v in items.items() if v["plan"] is not None} if all_eligible else allocation(items)
        for q in chosen:
            needed.update((q, int(d)) for d in items[q]["plan"]["doc_ids"])
    old_rows = readl(destination) if destination.exists() else []
    have = {(r["query_id"], r["doc_id"]): r for r in old_rows}
    require(len(have) == len(old_rows) and set(have) <= needed, "CE cache has duplicate/unplanned pairs")
    missing = sorted(needed - set(have))
    started = time.perf_counter()
    if missing:
        base = ROOT / "models/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)

        class CrossEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(base, local_files_only=True)
                self.drop = torch.nn.Dropout(.1)
                self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

            def forward(self, **tokens):
                return self.head(self.drop(self.encoder(**tokens).last_hidden_state[:, 0])).squeeze(-1)

        model = CrossEncoder().cuda().eval()
        model.load_state_dict(torch.load(CE, map_location="cpu", weights_only=True), strict=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle, torch.inference_mode():
            for offset in range(0, len(missing), 32):
                pairs = missing[offset:offset + 32]
                qtokens = tokenizer([queries[q] for q, d in pairs], add_special_tokens=False, truncation=True, max_length=64)["input_ids"]
                ctokens = tokenizer([corpus[d] for q, d in pairs], add_special_tokens=False, truncation=True, max_length=160)["input_ids"]
                ids = [tokenizer.build_inputs_with_special_tokens(q, c) for q, c in zip(qtokens, ctokens)]
                tokens = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**tokens).float().cpu().numpy()
                for (q, d), value, token_ids in zip(pairs, logits, ids):
                    row = {"query_id": q, "doc_id": d, "logit": float(value), "input_tokens": len(token_ids),
                           "query_conditioning": "original_user_query", "qrels_read": False}
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                if (offset // 32) % 64 == 0:
                    print({"stage": "real_CE", "pairs": min(offset + 32, len(missing)), "total": len(missing),
                           "seconds": round(time.perf_counter() - started, 1)}, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    elif not destination.exists():
        writel(destination, [])
    rows = readl(destination)
    values = {(r["query_id"], r["doc_id"]): r for r in rows}
    require(set(values) == needed and len(values) == len(rows), "real CE cache incomplete")
    by_query = defaultdict(dict)
    for (q, d), row in values.items():
        by_query[q][d] = row["logit"]
    for items in items_by_retriever.values():
        for q, item in items.items():
            item["ce"] = dict(by_query[q])
    return {"physical_unique_pairs": len(rows), "actual_input_tokens": sum(r["input_tokens"] for r in rows),
            "new_pairs_this_session": len(missing), "session_seconds": time.perf_counter() - started,
            "CE_weights_sha256": sha(CE), "pair_cache_sha256": sha(destination), "qrels_read": False}


def generate():
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import GenerationSettings, generate_candidates, load_tokenizer
    p = verify_design()
    check_seal("POLICY_SEALED.json")
    require(not (OUT / "OUTCOME_OPENING.json").exists(), "H already opened; no new generation allowed")
    if (OUT / "CANDIDATES_SEALED.json").exists():
        check_seal("CANDIDATES_SEALED.json")
        return
    queries = readl(OUT / "public/H_queries.jsonl")
    require({r["query_id"] for r in queries} == set(p["H_query_ids"]), "inference query role mismatch")
    partial = OUT / "candidates.partial.jsonl"
    cached = readl(partial) if partial.exists() else []
    pools = defaultdict(list)
    text = {r["query_id"]: r["query"] for r in queries}
    for r in cached:
        require(r["query_id"] in text, "foreign H candidate")
        pools[r["query_id"]].append(r)
    for q, rows in pools.items():
        require(sorted(r["position"] for r in rows) == list(range(1, len(text[q].split()) + 1)), "partial H query is incomplete; no silent regeneration")
    cuda_config()
    tokenizer = load_tokenizer(T5)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5, local_files_only=True).float().cuda().eval()
    settings = GenerationSettings(**p["generation"])
    started = time.perf_counter()
    with partial.open("a", encoding="utf-8") as handle:
        for index, query in enumerate(queries):
            q = query["query_id"]
            if q in pools:
                continue
            rows = generate_candidates(model, tokenizer, q, query["query"], settings)
            handle.write("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))
            handle.flush()
            if (index + 1) % 64 == 0 or index + 1 == len(queries):
                print({"stage": "H_generation_once", "queries": index + 1, "total": len(queries),
                       "seconds": round(time.perf_counter() - started, 1)}, flush=True)
                status("H_generation", done_queries=index + 1, all_queries=len(queries))
    partial.rename(OUT / "candidates.jsonl")
    candidates = readl(OUT / "candidates.jsonl")
    require({r["query_id"] for r in candidates} == set(p["H_query_ids"]), "H generation missing queries")
    writel(OUT / "selections.jsonl", selection_rows(candidates, p["H_query_ids"], p))
    seal("CANDIDATES_SEALED.json", [OUT / "candidates.jsonl", OUT / "selections.jsonl"],
         H_outcomes_opened=False, generated_once=True, extra_seed_generations=0,
         queries=len(queries), candidates=len(candidates), seconds_this_session=time.perf_counter() - started)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def rank(retriever):
    p = verify_design()
    check_seal("CANDIDATES_SEALED.json")
    destination = OUT / f"rankings/{retriever}.jsonl"
    if destination.with_suffix(".manifest.json").exists():
        require(sha(destination) == read(destination.with_suffix(".manifest.json"))["rankings_sha256"], "ranking drift")
        return
    require(not destination.exists(), "partial ranking file requires inspection")
    queries = readl(OUT / "public/H_queries.jsonl")
    actions = all_actions(queries, readl(OUT / "candidates.jsonl"))
    texts = list(dict.fromkeys(r["text"] for r in actions))
    corpus = readl(ADAPTER / "public/corpus.jsonl")
    docids = np.asarray([r["doc_id"] for r in corpus], dtype=np.int64)
    started = time.perf_counter()
    cache = OUT / f"retrieval_cache/{retriever}"
    cache.mkdir(parents=True, exist_ok=True)
    write(cache / "texts.json", texts, immutable=True)
    if retriever == "CodeBERT":
        import torch
        from experiments.rene_ssqr_alignment_20261019.pipeline import load_trained_retriever
        from experiments.rene_ssqr_alignment_20261019.retriever import encode_texts
        cuda_config()
        model, tokenizer = load_trained_retriever(CODEBERT, "cuda")
        vectors = encode_texts(model, tokenizer, texts, batch_size=16, device="cuda")
        np.save(cache / "query_vectors.npy", vectors, allow_pickle=False)
        code_vectors = np.load(ROOT / "results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy", allow_pickle=False)
        topids, topscores = [], []
        with threadpool_limits(limits=4):
            for offset in range(0, len(texts), 128):
                scores = vectors[offset:offset + 128] @ code_vectors.T
                for s in scores:
                    order = np.lexsort((docids, -s))[:100]
                    topids.append(docids[order].tolist())
                    topscores.append(s[order].astype(float).tolist())
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        bm25 = BM25([r["code_text"] for r in corpus], **p["bm25"]["parameters"])
        topids, topscores = [], []
        for text in texts:
            s = bm25.score(text)
            order = np.lexsort((docids, -s))[:100]
            topids.append(docids[order].tolist())
            topscores.append(s[order].tolist())
    index = {s: i for i, s in enumerate(texts)}
    writel(destination, ({**r, "ranked_doc_ids": topids[index[r["text"]]], "scores": topscores[index[r["text"]]]} for r in actions))
    write(destination.with_suffix(".manifest.json"), {"rankings_sha256": sha(destination), "actions": len(actions),
                                                     "texts": len(texts), "qrels_read": False,
                                                     "elapsed_seconds": time.perf_counter() - started}, immutable=True)
    print({"ranked_H": retriever, "actions": len(actions), "seconds": round(time.perf_counter() - started, 1)}, flush=True)


def features(retriever):
    p = verify_design()
    destination = OUT / f"features/{retriever}.jsonl"
    if destination.exists():
        require(sha(destination) == read(destination.with_suffix(".manifest.json"))["sha256"], "feature drift")
        return
    candidates = defaultdict(dict)
    for row in readl(OUT / "candidates.jsonl"):
        candidates[row["query_id"]][row["position"]] = row
    selections = {r["query_id"]: [v["position"] for v in sorted(r["selected"], key=lambda v: v["selected_rank"])]
                  for r in readl(OUT / "selections.jsonl") if r["policy"] == "ENTR"}
    actions = {(r["query_id"], r["position"]): r for r in readl(OUT / f"rankings/{retriever}.jsonl")}
    code_words = {r["doc_id"]: set(lexical_tokens(r["code_text"])) for r in readl(ADAPTER / "public/corpus.jsonl")}
    rows = []
    for query in readl(OUT / "public/H_queries.jsonl"):
        q = query["query_id"]
        pool = [candidates[q][pos] for pos in selections[q]]
        for rank, candidate in enumerate(pool, 1):
            pos = candidate["position"]
            rows.append({"query_id": q, "position": pos, "entr_rank": rank,
                         "values": feature_vector(query["query"], candidate, actions[q, None], actions[q, pos], pool, code_words).tolist(),
                         "structural_zero": actions[q, None]["ranked_doc_ids"] == actions[q, pos]["ranked_doc_ids"]})
    writel(destination, rows)
    write(destination.with_suffix(".manifest.json"), {"sha256": sha(destination), "rows": len(rows),
                                                      "feature_names": FEATURE_NAMES, "qrels_read": False}, immutable=True)
