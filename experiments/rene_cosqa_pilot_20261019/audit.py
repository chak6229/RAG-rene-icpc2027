"""Independent replay before and after opening development effects."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from fractions import Fraction

import numpy as np
from threadpoolctl import threadpool_limits

from .common import ADAPTER, INVENTORY, OUT, T5, now, read, readl, sha, verify_frozen, write
from .evaluation import require, validate_inputs


def preoutcome():
    protocol = verify_frozen()
    require(not (OUT / "metrics").exists(), "preoutcome audit cannot be first issued after effects exist")
    seal = read(OUT / "DECISIONS_SEALED.json")
    require(seal["protocol_sha256"] == sha(OUT / "PROTOCOL.json"), "protocol seal mismatch")
    for record in seal["files"]:
        require(sha(OUT / record["path"]) == record["sha256"], "action/cache seal mismatch")
    queries = readl(OUT / "data/queries.jsonl")
    corpus = readl(ADAPTER / "public/corpus.jsonl")
    candidates = readl(OUT / "candidates.jsonl")
    selections = readl(OUT / "selections.jsonl")
    groups = read(OUT / "data/groups.json")
    roles = read(OUT / "ROLE_MANIFEST.json")
    require(not set(roles["development_group_ids"]) & set(roles["confirmation_pool_group_ids"]), "D/H groups overlap")
    require(set(roles["official_test_group_ids"]) <= set(roles["confirmation_pool_group_ids"]), "official test groups entered development")
    inv = {row["idx"]: row for row in readl(INVENTORY)}
    repo_roles = {}
    code_roles = {}
    for query, role in roles["roles_by_query"].items():
        for repo in inv[query]["source_repositories"]:
            require(repo_roles.setdefault(repo, role) == role, "known repository crosses D/H")
        code = inv[query]["code_tokens_sha256"]
        require(code_roles.setdefault(code, role) == role, "known code crosses D/H")
    checks = {}
    doc_ids = np.array([row["doc_id"] for row in corpus], dtype=int)
    for retriever in protocol["retrievers"]:
        rankings = readl(OUT / f"rankings/{retriever}.jsonl")
        validate_inputs(queries, corpus, candidates, rankings, selections, groups, protocol)
        cache = OUT / f"retrieval_cache/{retriever}"
        texts = read(cache / "texts.json")
        index = {text: i for i, text in enumerate(texts)}
        scores = np.load(cache / "scores.npy", allow_pickle=False)
        require(scores.shape == (len(texts), len(corpus)), "retrieval score matrix has wrong shape")
        if retriever == "CodeBERT":
            query_vectors = np.load(cache / "query_vectors.npy", allow_pickle=False)
            code_vectors = np.load(cache / "code_vectors.npy", allow_pickle=False)
            with threadpool_limits(limits=4):
                independent_scores = np.matmul(query_vectors, np.transpose(code_vectors))
            difference = float(np.max(np.abs(independent_scores - scores)))
            require(difference < 1e-5, "independent dot product mismatch")
            checks["dense_score_max_difference"] = difference
        else:
            # Independent document-wise formula, not the production posting implementation.
            def words(text):
                expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
                return re.findall(r"[a-z][a-z0-9]*", expanded.lower())
            counts = [Counter(words(row["code_text"])) for row in corpus]
            lengths = np.array([sum(item.values()) for item in counts], dtype=np.float64)
            avg = max(float(lengths.mean()), 1.0)
            df = Counter()
            for count in counts:
                df.update(count.keys())
            sampled = sorted(range(len(texts)), key=lambda i: hashlib.sha256(f"bm25-audit-{texts[i]}".encode()).hexdigest())[:8]
            max_error = 0.0
            for text_id in sampled:
                query_terms = set(words(texts[text_id]))
                expected = np.zeros(len(corpus), dtype=np.float64)
                for j, count in enumerate(counts):
                    for word in sorted(query_terms):
                        if count[word]:
                            frequency = count[word]
                            idf = math.log(1 + (len(corpus) - df[word] + .5) / (df[word] + .5))
                            expected[j] += idf * frequency * 2.2 / (frequency + 1.2 * (.25 + .75 * lengths[j] / avg))
                max_error = max(max_error, float(np.max(np.abs(expected - scores[text_id]))))
            require(max_error < 1e-10, "independent BM25 formula mismatch")
            checks["bm25_independent_sampled_texts"] = len(sampled)
            checks["bm25_formula_max_difference"] = max_error
        for row in rankings:
            values = scores[index[row["text"]]]
            order = sorted(range(len(corpus)), key=lambda i: (-float(values[i]), int(doc_ids[i])))[:protocol["ranking_depth"]]
            require([int(doc_ids[i]) for i in order] == row["ranked_doc_ids"], "independent stable ranking differs")
            require(np.array_equal(np.array(row["scores"]), values[order]), "saved top scores differ from full retrieval cache")
        checks[f"{retriever}_ranking_actions_replayed"] = len(rankings)

    # Recompute real pilot candidate entropy using direct model logits and FP64 math.
    import torch
    from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = T5Tokenizer(vocab_file=str(T5.parents[1] / "t5-base/spiece.model"), extra_ids=100, legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5, local_files_only=True).float().cuda().eval()
    samples = sorted(candidates, key=lambda row: hashlib.sha256(f'{row["query_id"]}:{row["position"]}'.encode()).hexdigest())[:16]
    entropy_errors, logp_errors, detail = [], [], []
    with torch.inference_mode():
        for row in samples:
            inputs = tokenizer(row["masked_query"], return_tensors="pt").to("cuda")
            labels = tokenizer(f'<extra_id_0> {row["span"]} <extra_id_1>', return_tensors="pt").input_ids.to("cuda")
            logits = model(**inputs, labels=labels, decoder_attention_mask=torch.ones_like(labels)).logits[0].float().cpu().numpy().astype(np.float64)
            shifted = logits - logits.max(-1, keepdims=True)
            logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
            entropy = float(-(np.exp(logp) * logp).sum(-1).mean())
            label_logp = float(logp[np.arange(labels.shape[1]), labels[0].cpu().numpy()].mean())
            entropy_errors.append(abs(entropy - row["full_vocab_entropy"]))
            logp_errors.append(abs(label_logp - row["label_mean_log_probability"]))
            detail.append({"query_id": row["query_id"], "position": row["position"], "fp64_entropy": entropy,
                           "cached_entropy": row["full_vocab_entropy"], "fp64_mean_logp": label_logp})
    checks["pilot_candidates_fp64_replayed"] = len(samples)
    checks["pilot_fp64_entropy_max_error"] = max(entropy_errors)
    checks["pilot_fp64_logp_max_error"] = max(logp_errors)
    require(max(entropy_errors) < 1e-5 and max(logp_errors) < 1e-5, "pilot independent FP64 candidate score mismatch")
    report = {"passed": True, "created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
              "decision_seal_sha256": sha(OUT / "DECISIONS_SEALED.json"), "qrels_read": False,
              "method_effects_opened": False, "confirmation_effects_opened": False,
              "checks": checks, "fp64_samples": detail}
    write(OUT / "PREOUTCOME_AUDIT.json", report, immutable=True)
    print({"preoutcome_audit": "PASS", **checks}, flush=True)
    return report


def postoutcome():
    protocol = verify_frozen(include_private=True)
    qrels = readl(OUT / "data/private/pilot_qrels.jsonl")
    positives = {}
    for row in qrels:
        if row.get("relevance", 1) > 0:
            positives.setdefault(row["query_id"], set()).add(row["doc_id"])
    selections = readl(OUT / "selections.jsonl")
    expected_ids = set(protocol["query_ids"])
    max_error = 0.0
    checked = 0
    for retriever in protocol["retrievers"]:
        saved = {row["query_id"]: row for row in readl(OUT / f"metrics/{retriever}_per_query.jsonl")}
        rankings = readl(OUT / f"rankings/{retriever}.jsonl")
        require(set(saved) == expected_ids, "metric query denominator mismatch")
        action_rr = {}
        for row in rankings:
            rank = next((i for i, doc in enumerate(row["ranked_doc_ids"][:protocol["metric_k"]], 1)
                         if doc in positives[row["query_id"]]), None)
            action_rr[row["query_id"], row["position"]] = Fraction(1, rank) if rank is not None else Fraction(0)
        by_query = {query: {"ENTR": [], "PROB": [], "RAND": []} for query in expected_ids}
        for selection in selections:
            query, policy = selection["query_id"], selection["policy"]
            ordered = sorted(selection["selected"], key=lambda value: value["selected_rank"])
            rr_values = [action_rr[query, row["position"]] for row in ordered]
            by_query[query][policy].append((rr_values[0], max(rr_values)))
        for query in expected_ids:
            max_error = max(max_error, abs(saved[query]["NoRewrite"] - float(action_rr[query, None])))
            for policy, values in by_query[query].items():
                for column, index in (("top1_rr", 0), ("best3_rr", 1)):
                    exact = float(sum(value[index] for value in values) / len(values))
                    max_error = max(max_error, abs(saved[query]["policies"][policy][column] - exact))
                    checked += 1
        metric = read(OUT / f"metrics/{retriever}.json")
        base = float(sum(action_rr[query, None] for query in expected_ids) / len(expected_ids))
        max_error = max(max_error, abs(metric["NoRewrite_mrr"] - base))
        for policy in ("ENTR", "PROB", "RAND"):
            for part, index in (("automatic_top1", 0), ("candidate_set_quality", 1)):
                mrr = float(sum(sum(value[index] for value in by_query[query][policy]) / len(by_query[query][policy])
                                for query in expected_ids) / len(expected_ids))
                max_error = max(max_error, abs(metric[part][policy]["mrr"] - mrr),
                                abs(metric[part][policy]["delta_vs_NoRewrite"]["mean"] - (mrr - base)))
    require(max_error < 1e-12, "independent Fraction RR/MRR recomputation failed")
    verify_frozen()
    report = {"passed": True, "created_utc": now(), "protocol_sha256": sha(OUT / "PROTOCOL.json"),
              "independent_exact_rr_fields_checked": checked, "max_absolute_error": max_error,
              "development_batch_consumed": True, "confirmation_opened": False,
              "protected_pointers_unchanged": True}
    write(OUT / "POSTOUTCOME_AUDIT.json", report, immutable=True)
    print(report, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pre", "post"))
    args = parser.parse_args()
    preoutcome() if args.stage == "pre" else postoutcome()


if __name__ == "__main__":
    main()
