from __future__ import annotations

import argparse
import gc
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.rene_cosqa_pilot_20261019.evaluation import validate_inputs
from .common import (ADAPTER, OUT, PARENT, RETRIEVERS, ROOT, SOURCE, T5, check_seal, deny_relevance_reads,
                     digest, now, read, readl, require, sha, verify_design, write)
from .inference import load_items
from .policy import all_decisions


def precision_audit():
    plan = read(OUT / "PRECISION_PLAN.json")["selected"]
    primary = read(OUT / "PRIMARY_SAMPLE.json")
    require(primary["cap_per_group"] == 16 and len(primary["query_ids"]) == 7718, "unexpected primary cap")
    groups = read(OUT / "public/H_groups.json")
    counts = Counter(groups[q] for q in primary["query_ids"])
    require(len(counts) == 1963 and max(counts.values()) <= 16, "invalid primary group size")
    # Independent known-size and normal-power arithmetic, with no H labels/effects.
    from scipy.stats import norm, t
    dfactor = len(counts) * sum(n * n for n in counts.values()) / sum(counts.values()) ** 2
    variance = plan["variance_bootstrap_upper95"] * max(1., dfactor / plan["D_design_factor"])
    critical = float(t.isf(.025, len(counts) - 1))
    required = math.ceil((critical + norm.ppf(.8)) ** 2 * variance / .005 ** 2)
    require(abs(dfactor - plan["H_design_factor"]) < 1e-14 and required == plan["required_groups"], "independent precision arithmetic mismatch")
    require(required <= .95 * 1963 and plan["planned_power_at_capacity"] >= .8, "precision gate fails")
    return {"independent_required_groups": required, "available_groups": 1963, "alpha": .05,
            "effect_target_unchanged": .005, "all_groups_retained": True, "H_outcomes_used": False}


def audit_predictions():
    p = verify_design()
    require(not (OUT / "OUTCOME_OPENING.json").exists(), "pre-audit must precede H opening")
    check_seal("POLICY_SEALED.json")
    check_seal("DECISIONS_SEALED.json")
    checks = {"precision": precision_audit()}
    roles = read(OUT / "STAGE_ROLES.json")
    sets = [set(v) for v in roles["groups"].values()]
    require(all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:]), "stage roles overlap")
    require(not set.union(*sets) & set(p["H_group_ids"]), "H overlaps training group")
    queries = readl(OUT / "public/H_queries.jsonl")
    corpus = readl(ADAPTER / "public/corpus.jsonl")
    candidates = readl(OUT / "candidates.jsonl")
    selections = readl(OUT / "selections.jsonl")
    groups = read(OUT / "public/H_groups.json")
    docs = np.array([r["doc_id"] for r in corpus], dtype=np.int64)
    max_score_error, replayed = 0., 0
    with deny_relevance_reads(), threadpool_limits(limits=4):
        for ret in RETRIEVERS:
            rankings = readl(OUT / f"rankings/{ret}.jsonl")
            validate_inputs(queries, corpus, candidates, rankings, selections, groups,
                            {**p, "query_ids": p["H_query_ids"]})
            checks[ret + "_all_ranking_contract_checks"] = len(rankings)
            if ret == "CodeBERT":
                texts = read(OUT / "retrieval_cache/CodeBERT/texts.json")
                vectors = np.load(OUT / "retrieval_cache/CodeBERT/query_vectors.npy", allow_pickle=False)
                codes = np.load(ROOT / "results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy", allow_pickle=False)
                saved = {r["text"]: r for r in rankings}
                for offset in range(0, len(texts), 128):
                    scores = vectors[offset:offset + 128] @ codes.T
                    for text, values in zip(texts[offset:offset + 128], scores):
                        order = np.lexsort((docs, -values))[:100]
                        require(docs[order].tolist() == saved[text]["ranked_doc_ids"], "dense ranking replay differs")
                        max_score_error = max(max_score_error, float(np.max(np.abs(values[order] - saved[text]["scores"]))))
                require(max_score_error < 1e-5, "dense score replay differs")
                checks["dense_unique_texts_recomputed"] = len(texts)
                del vectors, codes, saved
            else:
                def words(text):
                    return re.findall(r"[a-z][a-z0-9]*", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower())
                counts = [Counter(words(r["code_text"])) for r in corpus]
                lengths = np.array([sum(c.values()) for c in counts], float)
                avg = max(float(lengths.mean()), 1.)
                df = Counter()
                for c in counts:
                    df.update(c.keys())
                saved = {r["text"]: r for r in rankings}
                chosen = sorted(saved, key=lambda s: digest(s, "rene-confirm-bm25-audit"))[:16]
                err = 0.
                for text in chosen:
                    terms = sorted(set(words(text)))
                    expected = np.array([sum(math.log1p((len(counts) - df[w] + .5) / (df[w] + .5)) * c[w] * 2.2
                                               / (c[w] + 1.2 * (.25 + .75 * length / avg)) for w in terms if c[w])
                                         for c, length in zip(counts, lengths)])
                    ordering = sorted(range(len(docs)), key=lambda i: (-float(expected[i]), int(docs[i])))[:100]
                    require(docs[ordering].tolist() == saved[text]["ranked_doc_ids"], "independent BM25 ranking differs")
                    err = max(err, float(np.max(np.abs(expected[ordering] - saved[text]["scores"]))))
                require(err < 1e-10, "independent BM25 formula mismatch")
                checks["BM25_independent_formula_samples"] = len(chosen)
                checks["BM25_independent_formula_max_error"] = err
                del counts, saved
            model, items = load_items(ret, p["H_query_ids"])
            require(set(model["gain"]["fit_query_ids"]) <= set(roles["queries"]["gain_fit"]), "gain role leak")
            require(set(model["verification"]["fit_query_ids"]) <= set(roles["queries"]["verification_fit"]), "verification role leak")
            require(set(model["risk"]["fit_query_ids"]) <= set(roles["queries"]["risk_fit"]), "risk role leak")
            ce = defaultdict(dict)
            token_lengths = {}
            for row in readl(OUT / "ce_scores.jsonl"):
                ce[row["query_id"]][row["doc_id"]] = row["logit"]
                token_lengths[row["query_id"], row["doc_id"]] = row["input_tokens"]
            for q, item in items.items():
                item["ce"] = dict(ce[q])
            outputs, allocated = all_decisions(items, model["verification"], model["risk"])
            saved_decisions = {(r["query_id"], r["policy"]): r for r in readl(OUT / f"decisions/{ret}.jsonl")}
            for row in outputs:
                old = saved_decisions[row["query_id"], row["policy"]]
                for key in ("ranking", "position", "accepted", "verified", "actual_pairs"):
                    require(row[key] == old[key], "reloaded model decision mismatch: " + key)
                for key in ("predicted_gain", "class_probabilities", "final_scores"):
                    if key in row and row[key] is not None:
                        require(np.allclose(row[key], old[key], atol=1e-12, rtol=0), "reloaded numeric prediction mismatch")
                tokens = sum(token_lengths[row["query_id"], d] for d in row.get("verification_doc_ids", []))
                require(tokens == old["actual_input_tokens"], "actual CE token accounting differs")
                replayed += 1
            checks[ret + "_verification_queries"] = len(allocated)
            del rankings, items, outputs, saved_decisions
            gc.collect()
    # Genuine full-vocabulary FP64 recalculation on newly generated H candidates.
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import load_tokenizer
    from .cache import cuda_config
    cuda_config()
    tokenizer = load_tokenizer(T5)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5, local_files_only=True).float().cuda().eval()
    entropy_error = logp_error = 0.
    chosen = sorted(candidates, key=lambda r: digest(f"{r['query_id']}:{r['position']}", "rene-confirm-FP64"))[:16]
    with deny_relevance_reads(), torch.inference_mode():
        for row in chosen:
            inputs = tokenizer(row["masked_query"], return_tensors="pt").to("cuda")
            labels = tokenizer(f'<extra_id_0> {row["span"]} <extra_id_1>', return_tensors="pt").input_ids.to("cuda")
            logits = model(**inputs, labels=labels, decoder_attention_mask=torch.ones_like(labels)).logits[0].float().cpu().numpy().astype(np.float64)
            shifted = logits - logits.max(-1, keepdims=True)
            lp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
            entropy = float(-(np.exp(lp) * lp).sum(-1).mean())
            value = float(lp[np.arange(labels.shape[1]), labels[0].cpu().numpy()].mean())
            entropy_error = max(entropy_error, abs(entropy - row["full_vocab_entropy"]))
            logp_error = max(logp_error, abs(value - row["label_mean_log_probability"]))
    require(max(entropy_error, logp_error) < 1e-5, "H FP64 entropy/logprob mismatch")
    del model
    torch.cuda.empty_cache()
    checks.update(dense_score_max_error=max_score_error, model_decisions_reloaded=replayed,
                  inference_replayed_under_relevance_read_denial=True,
                  H_candidates_FP64_checked=len(chosen), entropy_FP64_max_error=entropy_error, logp_FP64_max_error=logp_error)
    write(OUT / "PREOUTCOME_AUDIT.json", {"passed": True, "created_utc": now(), "checks": checks,
                                          "H_effects_opened": False, "decision_seal_sha256": sha(OUT / "DECISIONS_SEALED.json")}, immutable=True)
    print({"preoutcome_audit": "PASS", "model_decisions": replayed, "FP64_entropy_error": entropy_error}, flush=True)


def postoutcome():
    verify_design()
    check_seal("POLICY_SEALED.json")
    check_seal("DECISIONS_SEALED.json")
    require((OUT / "PRIMARY_RESULT.json").exists(), "outcomes not evaluated")
    clean = OUT / "clean_replay"
    require(not clean.exists(), "clean replay already exists; inspect retained artifacts")
    clean.mkdir(parents=True)
    pairs = [(SOURCE / "replay_worker.py", "replay_worker.py"), (OUT / "PRIMARY_SAMPLE.json", "PRIMARY_SAMPLE.json"),
             (OUT / "public/H_groups.json", "H_groups.json"), (OUT / "private/H_qrels.jsonl", "H_qrels.jsonl"),
             (OUT / "selections.jsonl", "selections.jsonl")]
    for ret in RETRIEVERS:
        pairs += [(OUT / f"rankings/{ret}.jsonl", f"{ret}_rankings.jsonl"), (OUT / f"decisions/{ret}.jsonl", f"{ret}_decisions.jsonl")]
    for source, name in pairs:
        shutil.copyfile(source, clean / name)
        require(sha(source) == sha(clean / name), "clean replay copy drift")
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    result = subprocess.run([sys.executable, "-X", "utf8", str(clean / "replay_worker.py"), str(clean)], cwd=clean,
                            env=env, capture_output=True, text=True, encoding="utf-8")
    require(result.returncode == 0, "standalone clean replay failed: " + result.stderr)
    print(result.stdout, flush=True)
    replay = read(clean / "REPLAY.json")
    primary = read(OUT / "PRIMARY_RESULT.json")
    max_error = 0.
    for key in ("mean", "standard_error_cluster", "ci95_cluster_t"):
        max_error = max(max_error, float(np.max(np.abs(np.asarray(primary[key]) - replay["primary"][key]))))
    for ret in RETRIEVERS:
        summary = read(OUT / f"metrics/{ret}_full_H.json")
        for name, m in summary["policy_results"].items():
            check = replay["retrievers"][ret]["stages"][name]
            for k, actual in (("delta", m["delta_MRR"]["mean"]), ("positive_gain", m["positive_gain"]), ("loss_magnitude", m["loss_magnitude"])):
                max_error = max(max_error, abs(actual - check[k]))
            for k, actual in (("adopted", m["adopted"]), ("harmed", m["harmed"]), ("pairs", m["CE_pairs_charged"]), ("tokens", m["CE_input_tokens_charged"])):
                require(actual == check[k], "independent stage count mismatch")
        for key, worker_key in (("missed_positive_gain", "gate_missed_positive"), ("avoided_loss", "gate_avoided_loss")):
            max_error = max(max_error, abs(summary["gate_accounting"][key] - replay["retrievers"][ret]["stages"][worker_key]))
        saved = {r["query_id"]: r for r in readl(OUT / f"metrics/{ret}_per_query.jsonl")}
        require(set(saved) == set(replay["retrievers"][ret]["per_query"]), "independent RR denominator differs")
        for q, values in replay["retrievers"][ret]["per_query"].items():
            for key, value in values.items():
                max_error = max(max_error, abs(saved[q]["candidate_best3"][key] - value))
    require(max_error < 1e-13, "independent RR/metric replay mismatch")
    design = read(OUT / "DESIGN.json")
    policy = read(OUT / "POLICY_SEALED.json")
    decisions = read(OUT / "DECISIONS_SEALED.json")
    pre = read(OUT / "PREOUTCOME_AUDIT.json")
    opening = read(OUT / "OUTCOME_OPENING.json")
    require(design["created_utc"] < policy["created_utc"] < decisions["created_utc"] < pre["created_utc"] < opening["opened_utc"], "freeze/decide/audit/open order invalid")
    final = {"passed": True, "created_utc": now(), "standalone_clean_replay": True,
             "clean_replay_mode": "cached rankings and decisions -> independent Fraction RR and clustered precision; not a neural-training reproduction",
             "Fraction_RR_checks": replay["fraction_RR_checks"], "max_metric_error": max_error,
             "model_decisions_reloaded": pre["checks"]["model_decisions_reloaded"],
             "stage_roles_disjoint": True, "precision_check": pre["checks"]["precision"],
             "freeze_before_H_decisions_before_outcome_opening": True, "original_five_pointers_unchanged": True,
             "risk_certified": False, "H_consumed": True, "H_queries": 8530, "H_groups": 1963,
             "source_frozen_no_post_H_tuning": True, "primary_result_sha256": sha(OUT / "PRIMARY_RESULT.json")}
    write(OUT / "FINAL_ACCEPTANCE.json", final, immutable=True)
    print({"final_acceptance": "PASS", "independent_RR_checks": replay["fraction_RR_checks"], "max_error": max_error}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pre", "post"))
    args = parser.parse_args()
    (audit_predictions if args.stage == "pre" else postoutcome)()
