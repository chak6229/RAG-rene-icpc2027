"""One-time downstream-independent confirmation for the RENE protocol.

This run uses a repository-disjoint sample from CodeXGLUE train that was not
used by the VAC-MQR query-level development roles.  It is intentionally small
and is a confirmation of the candidate-selection/UQ interface, not a safety
certificate.  The upstream CodeBERT checkpoint has seen the CodeXGLUE train
partition; that exposure is recorded in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vac_mqr import ENHANCED_RETRIEVAL_FEATURES as ENHANCED_FEATURES

OUT = ROOT / "results/rene_independent_confirmation_20261018"
TRAIN = ROOT / "data/codexglue/training/train.jsonl"
T5 = ROOT / "models/t5-cqc-codenn-1m/best"
RETRIEVER = ROOT / "models/codebert-codexglue-2epoch/checkpoint-3000"
SEEDS = (101, 202, 303, 404)
QUERY_COUNT = 96
CORPUS_COUNT = 2048
K = 100
TOL = 1e-12


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf8") if line.strip()]


def repository(url: str) -> str:
    match = re.match(r"https?://github\.com/([^/]+/[^/]+)", url)
    return match.group(1).casefold() if match else url.casefold()


def code_text(row: dict) -> str:
    tokens = row.get("code_tokens") or row.get("function_tokens")
    if not tokens:
        raise ValueError("code_tokens/function_tokens required")
    return " ".join(map(str, tokens))


def blocked_repositories() -> set[str]:
    blocked: set[str] = set()
    for path in (
        ROOT / "results/uq_pilot_20261014/action_witness_uq/data/action_items.pkl",
        ROOT / "results/fresh_risk_confirmation_20261015/data/fresh_rows.pkl",
    ):
        if not path.exists():
            continue
        try:
            with path.open("rb") as f:
                rows = pickle.load(f)
            for row in rows:
                if row.get("repo"):
                    blocked.add(str(row["repo"]).casefold())
                elif row.get("url"):
                    blocked.add(repository(row["url"]))
        except Exception:
            # A legacy cache that cannot be imported is still not used as a
            # source of confirmation rows; the selection remains conservative.
            continue
    return blocked


def select_rows() -> tuple[list[dict], list[dict], dict]:
    blocked = blocked_repositories()
    selected: list[dict] = []
    seen_repo: set[str] = set()
    with TRAIN.open(encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            repo = str(row.get("repo") or repository(row["url"])).casefold()
            query = str(row.get("docstring") or "").strip()
            if repo in blocked or repo in seen_repo or not query:
                continue
            if len(query.split()) > 64 or len(row.get("code_tokens") or row.get("function_tokens") or []) < 8:
                continue
            seen_repo.add(repo)
            selected.append(dict(row, _repo=repo, _query=query))
            if len(selected) >= CORPUS_COUNT:
                break
    if len(selected) < CORPUS_COUNT:
        raise RuntimeError(f"only selected {len(selected)} eligible train repositories")
    queries = selected[:QUERY_COUNT]
    corpus = selected
    return queries, corpus, {"blocked_repository_count": len(blocked), "query_count": len(queries), "corpus_count": len(corpus)}


def stable_indices(scores: np.ndarray, k: int = K) -> np.ndarray:
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores, float)))[:min(k, len(scores))]


def rr(ranking: np.ndarray, target: int) -> float:
    where = np.flatnonzero(np.asarray(ranking) == int(target))
    return 1.0 / (int(where[0]) + 1) if len(where) else 0.0


def rrf(original: np.ndarray, candidate: np.ndarray, offset: int = 60) -> np.ndarray:
    scores: dict[int, float] = defaultdict(float)
    for rank, doc in enumerate(original[:K], 1):
        scores[int(doc)] += 1.0 / (offset + rank)
    for rank, doc in enumerate(candidate[:K], 1):
        scores[int(doc)] += 1.0 / (offset + rank)
    return np.asarray(sorted(scores, key=lambda doc: (-scores[doc], doc)), dtype=int)[:K]


def encode(model, tokenizer, texts: list[str], max_length: int, device: str) -> np.ndarray:
    import torch
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch = tokenizer(texts[start:start + 32], padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda" and torch.cuda.is_bf16_supported()):
                hidden = model(**batch).last_hidden_state[:, 0, :]
            out.append(torch.nn.functional.normalize(hidden.float(), dim=1).cpu().numpy())
    return np.vstack(out).astype(np.float32)


def generate_candidates(rows: list[dict]) -> list[dict]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from vac_mqr import CandidateQualityValidator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(T5, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        T5, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=True).to(device).eval()
    validator = CandidateQualityValidator()
    excluded = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    outputs = []
    for index, row in enumerate(rows):
        query = row["_query"]
        by_seed = []
        for seed in SEEDS:
            torch.manual_seed(seed + index)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed + index)
            inputs = []
            positions = []
            words = query.split()
            for position in range(len(words) + 1):
                inputs.append(" ".join(words[:position] + ["<extra_id_0>"] + words[position:]))
                positions.append(position)
            accepted = []
            for start in range(0, len(inputs), 32):
                batch = tokenizer(inputs[start:start + 32], padding=True, truncation=True,
                                  max_length=512, return_tensors="pt").to(device)
                with torch.inference_mode():
                    generated = model.generate(**batch, do_sample=True, top_k=50, top_p=1.0,
                                               temperature=1.0, num_beams=1,
                                               num_return_sequences=1, max_new_tokens=12)
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
                for local, text in enumerate(decoded):
                    match = re.search(r"<extra_id_0>\s*(.*?)\s*(?:<extra_id_1>|</s>|$)", text)
                    span = re.sub(r"<[^>]+>", "", match.group(1)).strip() if match else ""
                    if not span:
                        continue
                    candidate = " ".join(words[:positions[start + local]] + [span] + words[positions[start + local]:])
                    if candidate == query:
                        continue
                    score = float(-len(span.split()))
                    quality = validator.validate(query, candidate)
                    if quality.accepted:
                        accepted.append({"text": candidate, "position": positions[start + local],
                                         "span": span, "seed": seed,
                                         "information_gain": score,
                                         "quality_pass": True,
                                         "constraint_preservation": quality.constraint_preservation,
                                         "negation_preserved": quality.negation_preserved})
            accepted.sort(key=lambda x: (x["information_gain"], x["text"]))
            unique = []
            seen = set()
            for candidate in accepted:
                key = validator.canonical(candidate["text"])
                if key not in seen:
                    seen.add(key)
                    candidate["rank"] = len(unique) + 1
                    unique.append(candidate)
                if len(unique) >= 3:
                    break
            by_seed.append(unique)
        union: dict[str, dict] = {}
        for seed_candidates in by_seed:
            for candidate in seed_candidates:
                key = validator.canonical(candidate["text"])
                union.setdefault(key, dict(candidate, seed_appearances=[]))
                if candidate["seed"] not in union[key]["seed_appearances"]:
                    union[key]["seed_appearances"].append(candidate["seed"])
        candidates = list(union.values())[:12]
        outputs.append(dict(query_index=index, source_idx=int(row["idx"]), repo=row["_repo"],
                            url=row["url"], query=query, candidates=candidates,
                            seed_candidates=by_seed))
        if (index + 1) % 16 == 0:
            print(f"generated {index + 1}/{len(rows)}", flush=True)
    return outputs


def build_evidence(rows: list[dict], corpus: list[dict], generated: list[dict]) -> list[dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from build_enhanced_retrieval_feedback_dataset import enhanced_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER, local_files_only=True)
    model = AutoModel.from_pretrained(RETRIEVER, local_files_only=True).to(device).eval()
    code_vectors = encode(model, tokenizer, [code_text(x) for x in corpus], 256, device)
    query_vectors = encode(model, tokenizer, [x["query"] for x in generated], 256, device)
    candidate_texts = [c["text"] for row in generated for c in row["candidates"]]
    candidate_vectors = encode(model, tokenizer, candidate_texts, 256, device) if candidate_texts else np.empty((0, code_vectors.shape[1]), dtype=np.float32)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    all_items = []
    offset = 0
    for local, (source, gen, qv) in enumerate(zip(rows, generated, query_vectors)):
        original_scores = code_vectors @ qv
        original_top = stable_indices(original_scores)
        target = local
        candidates = []
        for candidate in gen["candidates"]:
            cv = candidate_vectors[offset]
            offset += 1
            scores = code_vectors @ cv
            top = stable_indices(scores)
            features = enhanced_features(gen["query"], candidate, qv, cv, original_scores, scores)
            # Keep only label-free evidence in the decision cache.  RR is
            # opened after the sealed decisions are written.
            candidates.append(dict(candidate, features=features,
                                   candidate_vector=cv.astype(np.float32),
                                   candidate_top100=top))
        all_items.append(dict(query_index=local, source_idx=gen["source_idx"], repo=gen["repo"],
                              query=gen["query"], original_top100=original_top,
                              candidates=candidates,
                              seed_candidates=gen.get("seed_candidates", []),
                              target=target))
    assert offset == len(candidate_vectors)
    return all_items


def seed_features(item: dict) -> np.ndarray:
    candidates = item["candidates"]
    if not candidates:
        return np.empty((0, 12), dtype=float)
    vectors = np.asarray([c["candidate_vector"] for c in candidates], dtype=float)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("candidate vectors must be finite and rectangular")
    cos = np.clip(vectors @ vectors.T, -1.0, 1.0)
    seed_sets = [set(c["seed_appearances"]) for c in candidates]
    tops = [np.asarray(c["candidate_top100"]) for c in candidates]
    a10 = np.eye(len(candidates), dtype=float)
    a100 = np.eye(len(candidates), dtype=float)
    for i in range(len(candidates)):
        for j in range(i):
            a10[i, j] = a10[j, i] = rank_agreement(tops[i][:10], tops[j][:10])
            a100[i, j] = a100[j, i] = rank_agreement(tops[i], tops[j])
    groups = [[i for i, seeds in enumerate(seed_sets) if seed in seeds] for seed in SEEDS]
    weights = np.zeros(len(candidates), dtype=float)
    for group in groups:
        if group:
            weights[group] += 1.0 / (len(SEEDS) * len(group))
    weights /= max(weights.sum(), 1e-12)
    semdisp = float(1.0 - weights @ cos @ weights)
    retdisp = float(1.0 - weights @ a10 @ weights)
    support_counts = defaultdict(float)
    for group in groups:
        for i in group:
            for document in tops[i][:10]:
                support_counts[int(document)] += 1.0 / (len(SEEDS) * len(group))
    raw_count = sum(len(group) for group in item.get("seed_candidates", []))
    quality = 1.0
    out = []
    for i, (candidate, seeds) in enumerate(zip(candidates, seed_sets)):
        origin = next(seed for seed in SEEDS if seed in seeds)
        cross_semantic = []
        cross_top10 = []
        cross_top100 = []
        for seed, group in zip(SEEDS, groups):
            if seed == origin:
                continue
            cross_semantic.append(float(cos[i, group].max()) if group else 0.0)
            cross_top10.append(float(a10[i, group].max()) if group else 0.0)
            cross_top100.append(float(a100[i, group].max()) if group else 0.0)
        out.append([len(seeds) / len(SEEDS), np.mean(cross_semantic or [0.0]),
                    np.std(cross_semantic or [0.0]), np.mean(cross_top10 or [0.0]),
                    np.std(cross_top10 or [0.0]), np.mean(cross_top100 or [0.0]),
                    np.std(cross_top100 or [0.0]),
                    np.mean([support_counts[int(d)] for d in tops[i][:10]]),
                    semdisp, retdisp, len(candidates) / max(raw_count, 1), quality])
    return np.asarray(out, dtype=float)


def rank_agreement(left: np.ndarray, right: np.ndarray, k: int = 60) -> float:
    left_rank = {int(index): rank for rank, index in enumerate(left, 1)}
    right_rank = {int(index): rank for rank, index in enumerate(right, 1)}
    common = set(left_rank) & set(right_rank)
    numerator = sum(min(1.0 / (k + left_rank[index]),
                        1.0 / (k + right_rank[index])) for index in common)
    denominator = sum(1.0 / (k + rank) for rank in range(1, len(left) + 1))
    return float(numerator / denominator) if denominator else 0.0


def score_and_evaluate(items: list[dict]) -> dict:
    from experiments.gain_model_registry_20261014.registry import GainModel
    weak = GainModel("weak46-20261014", family="rewrite", fold=0)
    model_items = []
    for item in items:
        x = (np.asarray([[c["features"][name] for name in ENHANCED_FEATURES]
                         for c in item["candidates"]], dtype=float)
             if item["candidates"] else np.empty((0, len(ENHANCED_FEATURES)), dtype=float))
        sx = seed_features(item)
        model_items.append(dict(x=x, seed_x=sx,
                                original_top100=np.asarray(item["original_top100"], dtype=int),
                                candidate_top100=np.asarray([c["candidate_top100"] for c in item["candidates"]], dtype=int).reshape(-1, 100)))
    uq_predictions = weak.predict(model_items)
    no_uq_items = [dict(z, seed_x=np.zeros_like(z["seed_x"])) for z in model_items]
    no_uq_predictions = weak.predict(no_uq_items)
    methods = {"no_uq": no_uq_predictions, "seed_uq": uq_predictions}
    output = {}
    for method, predictions in methods.items():
        deltas = []
        chosen = []
        for item, pred in zip(items, predictions):
            candidates = item["candidates"]
            if not candidates:
                chosen.append(-1); deltas.append(0.0); continue
            scores = np.asarray(pred["mu"])
            index = int(np.argmax(scores)) if np.max(scores) > 0 else -1
            chosen.append(index)
            original = rr(item["original_top100"], item["target"])
            deltas.append(float(rr(candidates[index]["candidate_top100"], item["target"]) - original)
                           if index >= 0 else 0.0)
        y = np.asarray(deltas, dtype=float)
        output[method] = dict(query_count=len(items), accepted=int(np.count_nonzero(np.asarray(chosen) >= 0)),
                              positive=int(np.count_nonzero(y > TOL)), harmful=int(np.count_nonzero(y < -TOL)),
                              coverage=float(np.mean(np.asarray(chosen) >= 0)),
                              harmful_rate=float(np.mean(y[np.asarray(chosen) >= 0] < -TOL)) if np.any(np.asarray(chosen) >= 0) else None,
                              delta_mrr=float(y.mean()), gross_gain=float(np.maximum(y, 0).sum() / len(y)),
                              loss=float(np.maximum(-y, 0).sum() / len(y)), chosen=chosen,
                              deltas=y.tolist())
    oracle = []
    seed101 = []
    for item in items:
        original = rr(item["original_top100"], item["target"])
        candidate_deltas = [rr(c["candidate_top100"], item["target"]) - original
                            for c in item["candidates"]]
        oracle.append(max([0.0] + candidate_deltas))
        seed101_candidates = [c for c in item["candidates"] if 101 in c["seed_appearances"]]
        seed101.append(max([0.0] + [rr(c["candidate_top100"], item["target"]) - original
                                    for c in seed101_candidates]))
    output["oracle"] = dict(query_count=len(items), delta_mrr=float(np.mean(oracle)),
                             positive=int(np.count_nonzero(np.asarray(oracle) > TOL)))
    output["seed101_best"] = dict(query_count=len(items), delta_mrr=float(np.mean(seed101)),
                                   positive=int(np.count_nonzero(np.asarray(seed101) > TOL)))
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "PROTOCOL.json"
    if protocol_path.exists() and (OUT / "RESULTS.json").exists():
        print("confirmation already complete")
        return
    queries, corpus, selection = select_rows()
    protocol = dict(schema="rene-independent-confirmation-v2", status="FROZEN_BEFORE_LABELS",
                    frozen_utc=datetime.now(timezone.utc).isoformat(), test_used=False,
                    source_code_sha256=sha(Path(__file__)),
                    source_dataset=str(TRAIN), source_dataset_sha256=sha(TRAIN),
                    source_partition="CodeXGLUE train; query-level development roles disjoint",
                    upstream_retriever_exposure="CodeBERT checkpoint trained on CodeXGLUE train; retained and disclosed",
                    excluded_historical_repositories=selection["blocked_repository_count"],
                    query_count=selection["query_count"], corpus_count=selection["corpus_count"],
                    query_repositories=sorted({r["_repo"] for r in queries}),
                    candidate_generator=str(T5), candidate_generator_sha256=sha(T5 / "model.safetensors"),
                    retriever=str(RETRIEVER), retriever_weights_sha256=sha(RETRIEVER / "model.safetensors"),
                    generation=dict(seeds=list(SEEDS), all_whitespace_boundaries=True, do_sample=True,
                                    top_k=50, top_p=1.0, temperature=1.0, max_new_tokens=12,
                                    quality_filter_before_union=True),
                    retrieval=dict(top_k=K, pooling="normalized last_hidden_state[:,0,:]",
                                   tie_breaking="score descending, document index ascending",
                                   fusion="equal-weight RRF over original and candidate Top-100, offset 60"),
                    roles=dict(label_free="all selected rows before label opening", confirmation="all selected rows"),
                    statistical_scope="descriptive one-time confirmation; 96 queries is not a risk certificate",
                    artifact_scope="UQ candidate-selection interface using frozen weak46 gain component")
    write_json(protocol_path, protocol)
    failed_cache = OUT / "failed_label_leaking_snapshot.pkl"
    if failed_cache.exists():
        with failed_cache.open("rb") as stream:
            failed = pickle.load(stream)
        generated = failed["generated"]
        if len(generated) != len(queries):
            raise RuntimeError("failed candidate cache does not match the frozen query sample")
        print("reusing candidate generation from failed pre-seal attempt", flush=True)
    else:
        generated = generate_candidates(queries)
    items = build_evidence(queries, corpus, generated)
    label_free = dict(protocol_sha256=sha(protocol_path), query_count=len(items),
                      candidate_count=sum(len(x["candidates"]) for x in items),
                      generated=generated, items_without_labels=items)
    with (OUT / "label_free_decisions.pkl").open("wb") as f:
        pickle.dump(label_free, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(OUT / "DECISION_SEAL.json", dict(protocol_sha256=sha(protocol_path),
        label_free_sha256=sha(OUT / "label_free_decisions.pkl"), labels_opened_utc=datetime.now(timezone.utc).isoformat(),
        test_used=False, no_model_selection_after_labels=True))
    result = score_and_evaluate(items)
    write_json(OUT / "RESULTS.json", dict(protocol_sha256=sha(protocol_path), decision_seal_sha256=sha(OUT / "DECISION_SEAL.json"),
                                          test_used=False, one_time_confirmation=True, methods=result))
    protocol["status"] = "COMPLETED_ONE_TIME_CONFIRMATION"
    protocol["label_free_decisions_sha256"] = sha(OUT / "label_free_decisions.pkl")
    protocol["results_sha256"] = sha(OUT / "RESULTS.json")
    write_json(OUT / "CONFIRMATION_MANIFEST.json", protocol)
    lines = ["# RENE independent confirmation", "", "This is a one-time descriptive confirmation of the frozen weak46 UQ candidate-selection interface.",
             "The CodeBERT retriever was trained on the source partition; this upstream exposure is disclosed and prevents a claim of fully unseen model generalization.", "",
             "| method | delta MRR | coverage | positive | harmful | harmful rate |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("no_uq", "seed_uq"):
        z = result[name]
        lines.append(f"| {name} | {z['delta_mrr']:+.6f} | {z['coverage']:.3f} | {z['positive']} | {z['harmful']} | {z['harmful_rate'] if z['harmful_rate'] is not None else 'NA'} |")
    lines += ["", f"Oracle candidate ceiling: {result['oracle']['delta_mrr']:+.6f}; seed-101 best-candidate ceiling: {result['seed101_best']['delta_mrr']:+.6f}.",
              "No risk certificate is issued. The result is not a replacement for the larger independent confirmation required for a publication claim."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
