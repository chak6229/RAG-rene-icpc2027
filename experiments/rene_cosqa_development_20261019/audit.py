"""Independent formula/ranking replay and out-of-group prediction auditing."""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.rene_cosqa_pilot_20261019.evaluation import validate_inputs
from .common import (ADAPTER, INVENTORY, OUT, PILOT, T5, now, read, readl, require, sha,
                     verify_decisions, verify_frozen, write)
from .features import FEATURE_NAMES, GEN_NAMES


def words(text):
    return re.findall(r"[a-z][a-z0-9]*", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower())


def pre():
    protocol = verify_frozen(parent=True)
    require(not (OUT / "metrics").exists(), "first new preaudit must precede new effects")
    verify_decisions()
    queries, corpus = readl(OUT / "data/queries.jsonl"), readl(ADAPTER / "public/corpus.jsonl")
    candidates, selections = readl(OUT / "candidates.jsonl"), readl(OUT / "selections.jsonl")
    groups = read(OUT / "data/groups.json")
    roles, folds = read(OUT / "ROLE_MANIFEST.json"), read(OUT / "FOLDS.json")
    require(set(protocol["query_ids"]) == set(roles["development_query_ids"]), "not exactly the frozen D role")
    require(not set(protocol["query_ids"]) & set(roles["confirmation_pool_query_ids"]), "confirmation processed")
    require(not set(roles["development_group_ids"]) & set(roles["confirmation_pool_group_ids"]), "D/H group leak")
    require(set(roles["official_test_group_ids"]) <= set(roles["confirmation_pool_group_ids"]), "official test group in D")
    inventory = {r["idx"]: r for r in readl(INVENTORY)}
    repos, codes = {}, {}
    for qid, role in roles["roles_by_query"].items():
        for repo in inventory[qid]["source_repositories"]:
            require(repos.setdefault(repo, role) == role, "known repository crosses roles")
        code = inventory[qid]["code_tokens_sha256"]
        require(codes.setdefault(code, role) == role, "code crosses roles")
    for qid, fold in folds["query_to_fold"].items():
        require(folds["group_to_fold"][groups[qid]] == fold, "group crosses training folds")
    public = {r["query_id"]: r["query"] for r in readl(ADAPTER / "public/all_positive_queries.jsonl")}
    normalized = lambda q: " ".join(public[q].lower().split())
    require(not {normalized(q) for q in roles["development_query_ids"]} &
            {normalized(q) for q in roles["confirmation_pool_query_ids"]}, "normalized query text crosses D/H")
    prior_candidates = {(r["query_id"], r["position"]): r for r in readl(PILOT / "candidates.jsonl")}
    candidate_map = {(r["query_id"], r["position"]): r for r in candidates}
    require(all(candidate_map[key] == row for key, row in prior_candidates.items()), "pilot candidate altered")
    doc_ids = np.array([r["doc_id"] for r in corpus], dtype=int)
    checks = {"D_queries": len(queries), "D_groups": len(set(groups.values())), "H_effects_opened": False,
              "reused_candidates_exact": len(prior_candidates), "normalized_D_H_query_overlap": 0}
    for retriever in protocol["retrievers"]:
        rankings = readl(OUT / f"rankings/{retriever}.jsonl")
        _, actions, selected = validate_inputs(queries, corpus, candidates, rankings, selections, groups, protocol)
        old_actions = {(r["query_id"], r["position"]): r for r in readl(PILOT / f"rankings/{retriever}.jsonl")}
        require(all(actions[key] == value for key, value in old_actions.items()), "pilot action ranking altered")
        cache = OUT / f"retrieval_cache/{retriever}"
        texts = read(cache / "texts.json")
        text_index = {text: i for i, text in enumerate(texts)}
        scores = np.load(cache / "scores.npy", allow_pickle=False)
        require(scores.shape == (len(texts), len(corpus)), "score cache dimensions differ")
        old_cache = PILOT / f"retrieval_cache/{retriever}"
        old_texts, old_scores = read(old_cache / "texts.json"), np.load(old_cache / "scores.npy", allow_pickle=False)
        require(texts[:len(old_texts)] == old_texts and np.array_equal(scores[:len(old_texts)], old_scores), "pilot score cache altered")
        if retriever == "CodeBERT":
            vectors = np.load(cache / "query_vectors.npy", allow_pickle=False)
            code_vectors = np.load(old_cache / "code_vectors.npy", allow_pickle=False)
            difference = 0.0
            with threadpool_limits(limits=4):
                for start in range(0, len(texts), 256):
                    expected = vectors[start:start + 256] @ code_vectors.T
                    difference = max(difference, float(np.max(np.abs(expected - scores[start:start + 256]))))
            require(difference < 1e-5, "independent dense dot mismatch")
            checks["CodeBERT_max_dot_error"] = difference
        else:
            counts = [Counter(words(r["code_text"])) for r in corpus]
            lengths = np.asarray([sum(c.values()) for c in counts], dtype=float)
            average = max(float(lengths.mean()), 1.0)
            df = Counter()
            for count in counts:
                df.update(count.keys())
            sample = sorted(range(len(old_texts), len(texts)), key=lambda i: hashlib.sha256(texts[i].encode()).hexdigest())[:8]
            error = 0.0
            for index in sample:
                expected = np.zeros(len(corpus))
                for j, count in enumerate(counts):
                    for word in sorted(set(words(texts[index]))):
                        frequency = count[word]
                        if frequency:
                            expected[j] += math.log1p((len(corpus) - df[word] + .5) / (df[word] + .5)) * frequency * 2.2 / (
                                frequency + 1.2 * (.25 + .75 * lengths[j] / average))
                error = max(error, float(np.max(np.abs(expected - scores[index]))))
            require(error < 1e-10, "independent BM25 formula differs")
            checks["BM25_independent_new_texts"] = len(sample)
            checks["BM25_formula_max_error"] = error
        # Python stable sorting independently checks every action against its full score row.
        for row in rankings:
            x = scores[text_index[row["text"]]]
            order = sorted(range(len(corpus)), key=lambda i: (-float(x[i]), int(doc_ids[i])))[:100]
            require([int(doc_ids[i]) for i in order] == row["ranked_doc_ids"], "independent stable ranking mismatch")
            require(np.array_equal(np.asarray(row["scores"]), x[order]), "stored top scores differ")
        feature_rows = readl(OUT / f"features/{retriever}.jsonl")
        require([(r["query_id"], r["position"]) for r in feature_rows] ==
                [(q, p) for q in protocol["query_ids"] for p in selected[q, "ENTR", None]], "features outside common pool or wrong order")
        for feature in feature_rows:
            qid, position = feature["query_id"], feature["position"]
            f = dict(zip(FEATURE_NAMES, feature["values"]))
            candidate, original, new = candidate_map[qid, position], actions[qid, None], actions[qid, position]
            require(len(feature["values"]) == len(FEATURE_NAMES) and np.isfinite(feature["values"]).all(), "bad feature schema")
            independent = {
                "full_vocab_entropy": candidate["full_vocab_entropy"],
                "label_mean_log_probability": candidate["label_mean_log_probability"],
                "difference_top1": new["scores"][0] - original["scores"][0],
                "candidate_margin12": new["scores"][0] - new["scores"][1],
                "same_top1": float(original["ranked_doc_ids"][0] == new["ranked_doc_ids"][0]),
                "same_ranking": float(original["ranked_doc_ids"] == new["ranked_doc_ids"]),
                "jaccard10": len(set(original["ranked_doc_ids"][:10]) & set(new["ranked_doc_ids"][:10])) /
                             len(set(original["ranked_doc_ids"][:10]) | set(new["ranked_doc_ids"][:10])),
            }
            require(all(abs(f[k] - v) < 1e-12 for k, v in independent.items()), "independent decision evidence mismatch")
            require(feature["structural_zero"] == bool(independent["same_ranking"]), "wrong structural zero")
        checks[f"{retriever}_action_rankings_replayed"] = len(rankings)
        checks[f"{retriever}_same_pool_feature_rows"] = len(feature_rows)
    # Actual remaining-D candidates, not the earlier pilot's arithmetic check.
    import torch
    from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = T5Tokenizer(vocab_file=str(T5.parents[1] / "t5-base/spiece.model"), extra_ids=100, legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5, local_files_only=True).float().cuda().eval()
    new_ids = set(protocol["new_query_ids"])
    sample = sorted((r for r in candidates if r["query_id"] in new_ids),
                    key=lambda r: hashlib.sha256(f'{r["query_id"]}:{r["position"]}'.encode()).hexdigest())[:8]
    entropy_error, logp_error = 0.0, 0.0
    details = []
    with torch.inference_mode():
        for row in sample:
            inputs = tokenizer(row["masked_query"], return_tensors="pt").to("cuda")
            labels = tokenizer(f'<extra_id_0> {row["span"]} <extra_id_1>', return_tensors="pt").input_ids.to("cuda")
            raw = model(**inputs, labels=labels, decoder_attention_mask=torch.ones_like(labels)).logits[0].float().cpu().numpy().astype(np.float64)
            shifted = raw - raw.max(-1, keepdims=True)
            logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
            entropy = float(-(np.exp(logp) * logp).sum(-1).mean())
            label_logp = float(logp[np.arange(labels.shape[1]), labels[0].cpu().numpy()].mean())
            entropy_error = max(entropy_error, abs(entropy - row["full_vocab_entropy"]))
            logp_error = max(logp_error, abs(label_logp - row["label_mean_log_probability"]))
            details.append({"query_id": row["query_id"], "position": row["position"], "fp64_entropy": entropy})
    require(entropy_error < 1e-5 and logp_error < 1e-5, "remaining-D FP64 entropy/logp replay failed")
    checks.update({"new_candidate_fp64_replays": len(sample), "entropy_max_error": entropy_error, "logp_max_error": logp_error})
    write(OUT / "PREOUTCOME_AUDIT.json", {"passed": True, "created_utc": now(), "checks": checks,
                                         "protocol_sha256": sha(OUT / "PROTOCOL.json"), "new_method_effects_opened": False,
                                         "qrels_read_by_audit": False, "fp64_samples": details, "confirmation_opened": False}, immutable=True)
    print({"preoutcome_audit": "PASS", **checks}, flush=True)


def post():
    protocol = verify_frozen(private=True, parent=True)
    verify_decisions()
    groups = read(OUT / "data/groups.json")
    folds = read(OUT / "FOLDS.json")
    positives = defaultdict(set)
    for row in readl(OUT / "data/private/development_qrels.jsonl"):
        if row.get("relevance", 1) > 0:
            positives[row["query_id"]].add(row["doc_id"])
    selections = readl(OUT / "selections.jsonl")
    checks = {}
    max_error, max_prediction_error = 0.0, 0.0
    total_rr_fields = 0
    for retriever in protocol["retrievers"]:
        seal = read(OUT / f"OOF_SEALED_{retriever}.json")
        for row in seal["files"]:
            require(sha(OUT / row["path"]) == row["sha256"], "OOF sealed file changed")
        ranking_rows = readl(OUT / f"rankings/{retriever}.jsonl")
        actions = {(r["query_id"], r["position"]): r for r in ranking_rows}
        exact_rr = {}
        for row in ranking_rows:
            qid = row["query_id"]
            rank = next((i for i, doc in enumerate(row["ranked_doc_ids"], 1) if doc in positives[qid]), None)
            exact_rr[qid, row["position"]] = Fraction(1, rank) if rank else Fraction(0)
        rq1 = {r["query_id"]: r for r in readl(OUT / f"metrics/{retriever}_per_query.jsonl")}
        original_pilot = {r["query_id"]: r for r in readl(PILOT / f"metrics/{retriever}_per_query.jsonl")}
        require(all(rq1[q] == r for q, r in original_pilot.items()), "original pilot RR/selection changed")
        expected_policy = defaultdict(lambda: defaultdict(list))
        for row in selections:
            qid = row["query_id"]
            ordered = sorted(row["selected"], key=lambda r: r["selected_rank"])
            values = [exact_rr[qid, r["position"]] for r in ordered]
            expected_policy[qid][row["policy"]].append((values[0], max(values)))
        for qid in protocol["query_ids"]:
            max_error = max(max_error, abs(rq1[qid]["NoRewrite"] - float(exact_rr[qid, None])))
            for policy, repeats in expected_policy[qid].items():
                for field, index in (("top1_rr", 0), ("best3_rr", 1)):
                    value = float(sum(r[index] for r in repeats) / len(repeats))
                    max_error = max(max_error, abs(rq1[qid]["policies"][policy][field] - value))
                    total_rr_fields += 1
        rq1_summary = read(OUT / f"metrics/{retriever}_RQ1.json")["subsets"]
        rq2_summary = read(OUT / f"metrics/{retriever}_RQ2.json")["subsets"]
        rq2 = {r["query_id"]: r for r in readl(OUT / f"metrics/{retriever}_RQ2_per_query.jsonl")}
        decisions = {(r["query_id"], r["policy"]): r for r in readl(OUT / f"decisions/{retriever}.jsonl")}
        for label, ids in (("all_development", protocol["query_ids"]), ("original_pilot", protocol["pilot_query_ids"]),
                           ("remaining_development", protocol["new_query_ids"])):
            exact_base = sum(exact_rr[q, None] for q in ids) / len(ids)
            max_error = max(max_error, abs(rq1_summary[label]["NoRewrite_mrr"] - float(exact_base)))
            for policy in ("ENTR", "PROB", "RAND"):
                for part, index in (("automatic_top1", 0), ("candidate_set_quality", 1)):
                    exact_mean = sum(sum(v[index] for v in expected_policy[q][policy]) / len(expected_policy[q][policy]) for q in ids) / len(ids)
                    saved = rq1_summary[label][part][policy]
                    max_error = max(max_error, abs(saved["mrr"] - float(exact_mean)),
                                    abs(saved["delta_vs_NoRewrite"]["mean"] - float(exact_mean - exact_base)))
            for policy, saved in rq2_summary[label]["policies"].items():
                exact_values = [exact_rr[q, decisions[q, policy]["selected_position"]] - exact_rr[q, None] for q in ids]
                exact_mean = sum(exact_values) / len(ids)
                adopted = sum(actions[q, decisions[q, policy]["selected_position"]]["ranked_doc_ids"] != actions[q, None]["ranked_doc_ids"] for q in ids)
                harmful = sum(v < 0 for v in exact_values)
                require(saved["adopted"] == adopted and saved["harmful"] == harmful, "adoption/harm independent replay differs")
                max_error = max(max_error, abs(saved["delta"]["mean"] - float(exact_mean)), abs(saved["mrr"] - float(exact_base + exact_mean)))
                total_rr_fields += len(ids)
        feature_rows = readl(OUT / f"features/{retriever}.jsonl")
        oof_rows = readl(OUT / f"oof_predictions/{retriever}.jsonl")
        require([(r["query_id"], r["position"]) for r in feature_rows] == [(r["query_id"], r["position"]) for r in oof_rows], "OOF row ordering mismatch")
        x = np.asarray([r["values"] for r in feature_rows])
        for fit in read(OUT / f"TRAINING_LEDGER_{retriever}.json")["fits"]:
            if not isinstance(fit["fold"], int):
                require(fit["used_for_development_metrics"] is False and fit["test_queries"] == 0, "full-D model used for D outcomes")
                continue
            fold, model_name = fit["fold"], fit["model"]
            expected_test = {g for g, f in folds["group_to_fold"].items() if f == fold}
            expected_train = set(folds["group_to_fold"]) - expected_test
            require(set(fit["train_groups"]) == expected_train and set(fit["test_groups"]) == expected_test, "OOF group provenance differs")
            indices = [i for i, r in enumerate(oof_rows) if r["fold"] == fold]
            require(all(groups[oof_rows[i]["query_id"]] in expected_test for i in indices), "held prediction group in training")
            model = joblib.load(OUT / fit["model_path"])
            with threadpool_limits(limits=4):
                pred = np.clip(model.predict(x[indices, :fit["feature_columns"]]), -1, 1)
            for i, value in zip(indices, pred):
                saved = oof_rows[i]["predictions"][model_name]
                max_prediction_error = max(max_prediction_error, abs(saved["raw_clipped"] - float(value)))
                effective = 0.0 if oof_rows[i]["structural_zero"] else float(value)
                max_prediction_error = max(max_prediction_error, abs(saved["effective"] - effective))
        pooled = defaultdict(list)
        for row in oof_rows:
            pooled[row["query_id"]].append(row)
        for qid, pool in pooled.items():
            for name in protocol["direct_gain"]["models"]:
                best = sorted(pool, key=lambda r: (-r["predictions"][name]["effective"], r["entr_rank"], r["position"]))[0]
                for suffix in ("_forced", "_selective"):
                    position = None if suffix == "_selective" and best["predictions"][name]["effective"] <= 0 else best["position"]
                    saved = decisions[qid, name + suffix]
                    require(saved["selected_position"] == position, "independent policy replay differs")
                    wanted = actions[qid, position]["ranked_doc_ids"]
                    require(saved["ranked_doc_ids"] == wanted, "policy rank/fallback replay differs")
                    actual = float(exact_rr[qid, position])
                    max_error = max(max_error, abs(rq2[qid]["policies"][name + suffix]["rr"] - actual))
        checks[f"{retriever}_OOF_rows"] = len(oof_rows)
        checks[f"{retriever}_fold_models_reloaded"] = 15
    require(max_error < 1e-12 and max_prediction_error < 1e-12, "independent postoutcome replay failed")
    verify_frozen()
    write(OUT / "POSTOUTCOME_AUDIT.json", {"passed": True, "created_utc": now(), "checks": checks,
                                          "exact_rr_fields_checked": total_rr_fields, "metric_max_error": max_error,
                                          "prediction_max_error": max_prediction_error,
                                          "protocol_sha256": sha(OUT / "PROTOCOL.json"), "protected_pointers_unchanged": True,
                                          "confirmation_opened": False}, immutable=True)
    print({"postoutcome_audit": "PASS", "metric_error": max_error, "prediction_error": max_prediction_error}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pre", "post"))
    (pre if parser.parse_args().stage == "pre" else post)()
