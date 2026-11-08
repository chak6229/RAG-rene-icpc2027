"""Small, resumable valid-only T5 sampling-consistency pilot.

Eight decoding seeds, frozen T5/CodeBERT, separate repository roles. Historical
valid has been viewed: results are exploratory, never a new test conclusion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import time

import joblib
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import torch

from build_crsu_pilot import sha256, stable_indices
from build_enhanced_retrieval_feedback_dataset import weighted_rank_agreement
from build_rewrite_utility_dataset import encode, feature_row
from crsu.core import rrf
from prepare_vac_mqr_cv_folds import repository
from run_uq_pilot import booster, clustered_ci, dump, finite_quantile, grouped_folds
from vac_mqr import CandidateQualityValidator, ENHANCED_RETRIEVAL_FEATURES


OUT = Path("results/uq_pilot_20261011/t5_seed_consistency")
SOURCE = Path("results/vac_mqr_v5/enhanced_cv_folds/fold_0.jsonl")
VALID = Path("data/codexglue/advtest/valid.jsonl")
T5 = Path("models/t5-cqc-codenn-1m/best")
RETRIEVER = Path("models/codebert-codexglue-2epoch/checkpoint-3000")
CACHE = Path("results/crsu_development/valid_vector_cache")
STABLE = Path("results/vac_mqr_v5/STABLE.json")
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808)
COUNTS = dict(utility_train=160, lcb_calibration=48, selection_valid=48,
              risk_calibration=96, new_holdout=128)
BASE_FEATURES = list(ENHANCED_RETRIEVAL_FEATURES)
SEED_FEATURES = [
    "exact_seed_support", "cross_seed_semantic_max_mean", "cross_seed_semantic_max_std",
    "cross_seed_top10_agreement_mean", "cross_seed_top10_agreement_std",
    "cross_seed_top100_agreement_mean", "cross_seed_top100_agreement_std",
    "candidate_top10_document_support", "query_semantic_dispersion",
    "query_top10_dispersion", "query_unique_fraction", "seed_quality_pass_fraction",
]
TOL = 1e-12
DEPENDENCIES = [Path(p) for p in (
    'build_crsu_pilot.py', 'build_enhanced_retrieval_feedback_dataset.py',
    'build_rewrite_utility_dataset.py', 'crsu/core.py',
    'prepare_vac_mqr_cv_folds.py', 'run_uq_pilot.py', 'vac_mqr.py')]


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).open(encoding="utf-8") if s.strip()]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"PROTOCOL.json").exists():
        protocol = json.loads((OUT/"PROTOCOL.json").read_text(encoding="utf-8"))
        assert protocol["source_sha256"] == sha256(SOURCE)
        assert protocol["stable_sha256"] == sha256(STABLE)
        assert protocol["source_code_sha256"] == sha256(Path(__file__))
        assert protocol["valid_sha256"] == sha256(VALID)
        assert all(sha256(Path(p)) == h for p, h in protocol["weights"].items())
        assert all(sha256(Path(p)) == h for p, h in protocol["model_assets"].items())
        assert protocol["corpus_vectors_sha256"] == sha256(CACHE/"code_vectors.npy")
        assert all(sha256(Path(p)) == h for p, h in protocol["dependencies"].items())
        return protocol
    rows = read_jsonl(SOURCE)
    assert all(x["split_role"] != "test" for x in rows)
    rng = np.random.default_rng(11120)
    sampled = []
    for role, count in COUNTS.items():
        pool = [r for r in rows if r["split_role"] == role and len(r["query"].split()) <= 64]
        assert len(pool) >= count, (role, len(pool), count)
        ix = sorted(rng.choice(len(pool), count, replace=False))
        sampled.extend(pool[i] for i in ix)
    sampled.sort(key=lambda r: r["query_index"])
    repos = {role: {repository(r["url"]) for r in sampled if r["split_role"] == role}
             for role in COUNTS}
    assert all(not repos[a]&repos[b] for a in repos for b in repos if a != b)
    weights = {str(p): sha256(p) for model in (T5, RETRIEVER)
               for p in sorted(model.glob("*.safetensors"))}
    assert len(weights) == 2
    protocol = dict(
        development_only=True, exploratory=True, test_used=False,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS), role_counts=COUNTS, max_query_words=64,
        sampling_seed=11120, source_sha256=sha256(SOURCE), valid_sha256=sha256(VALID),
        source_code_sha256=sha256(Path(__file__)), stable_sha256=sha256(STABLE),
        dependencies={str(p): sha256(p) for p in DEPENDENCIES},
        weights=weights, roles={role: [r["query_index"] for r in sampled if r["split_role"] == role]
                               for role in COUNTS},
        model_assets={str(p): sha256(p) for model in (T5, RETRIEVER)
                      for p in sorted(model.iterdir()) if p.suffix in (".json", ".txt", ".model")},
        corpus_vectors_sha256=sha256(CACHE/"code_vectors.npy"),
        generation=dict(all_whitespace_boundaries=True, chunk_queries=8, batch_size=32,
                        do_sample=True, top_k=50, top_p=1., temperature=1., num_beams=1,
                        max_input_tokens=512, max_new_tokens=12, precision="bfloat16",
                        seed_schedule="decoding_seed + local_sorted_sample_chunk_start",
                        quality_before_top3=True),
        baseline="new pilot seed101; different batching from historical generation, not byte-identical replay",
        selector="query-weighted signed LightGBM leaves7/min30/120 rounds; four inner repo OOF folds",
        model_features=dict(base=BASE_FEATURES, seed=BASE_FEATURES+SEED_FEATURES),
        arms=["base", "seed_features", "base_mean_seed_scale"],
        pool_controls=["seed101", "multi8", "multi8_entropy_top3"],
        training="all unique multi8 utility-train actions only; same model compared across action pools",
        lower_bound="first-mean-action normalized residual, alpha .1; recalibrate each action pool on LCB role",
        risk="select one mean>0/harm threshold on selection, check once on risk; strict LCB alternative separately",
        limitation="historical valid roles; no independent confirmation; query-binomial bound ignores repo dependence",
    )
    dump(OUT/"PROTOCOL.json", protocol)
    snapshot = OUT/"frozen_source"
    for source in [Path(__file__)] + DEPENDENCIES:
        relative = Path(source.name) if source == Path(__file__) else source
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    with (OUT/"sampled_rows.pkl").open("wb") as stream:
        pickle.dump(sampled, stream)
    return protocol


def generation(protocol):
    path = OUT/"generation.jsonl"
    with (OUT/"sampled_rows.pkl").open("rb") as stream:
        sampled = pickle.load(stream)
    completed = read_jsonl(path) if path.exists() else []
    assert [r["query_index"] for r in completed] == [r["query_index"] for r in sampled[:len(completed)]]
    if len(completed) == len(sampled):
        print("Generation already complete", flush=True)
        return
    if len(completed)%8:
        raise RuntimeError("generation incomplete within chunk; preserve output and repair before resuming")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this sampling pilot")
    tokenizer = AutoTokenizer.from_pretrained(T5)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5, torch_dtype=torch.bfloat16).to("cuda").eval()
    validator = CandidateQualityValidator()
    excluded = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    started = time.perf_counter()
    for chunk in range(len(completed), len(sampled), 8):
        current = sampled[chunk:chunk+8]
        records = []
        for local, row in enumerate(current):
            tokens = row["query"].split()
            for position in range(len(tokens)+1):
                records.append((local, position, " ".join(tokens[:position]+["<extra_id_0>"]+tokens[position:])))
        output = [dict(query_index=r["query_index"], query=r["query"], url=r["url"],
                       role=r["split_role"], draws=[]) for r in current]
        for seed in SEEDS:
            seed_started = time.perf_counter()
            torch.manual_seed(seed+chunk)
            torch.cuda.manual_seed_all(seed+chunk)
            candidates = [[] for _ in current]
            for start in range(0, len(records), 32):
                batch = records[start:start+32]
                encoded = tokenizer([r[2] for r in batch], padding=True, truncation=True,
                                    max_length=512, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    g = model.generate(**encoded, do_sample=True, top_k=50, top_p=1.,
                                       temperature=1., num_beams=1, num_return_sequences=1,
                                       max_new_tokens=12, return_dict_in_generate=True, output_scores=True)
                decoded = tokenizer.batch_decode(g.sequences, skip_special_tokens=False)
                ids = g.sequences[:, 1:]
                terms = [[] for _ in batch]
                for step, logits in enumerate(g.scores):
                    p = torch.softmax(logits.float(), -1)
                    ent = (p*torch.log(p.clamp_min(1e-12))).sum(-1).cpu().tolist()
                    for j, token in enumerate(ids[:, step].tolist()):
                        if token not in excluded:
                            terms[j].append(ent[j])
                for j, (local, position, _) in enumerate(batch):
                    match = re.search(r"<extra_id_0>\s*(.*?)\s*(?:<extra_id_1>|</s>|$)", decoded[j])
                    span = re.sub(r"<[^>]+>", "", match.group(1)).strip() if match else ""
                    if not span:
                        continue
                    toks = current[local]["query"].split()
                    text = " ".join(toks[:position]+[span]+toks[position:])
                    if text == current[local]["query"] or not terms[j]:
                        continue
                    ig = float(np.mean(terms[j]))
                    if not math.isfinite(ig):
                        raise FloatingPointError("nonfinite generation entropy")
                    candidates[local].append(dict(text=text, position=position, span=span,
                                                  information_gain=ig, seed=seed))
            for local, raw in enumerate(candidates):
                raw.sort(key=lambda c: -c["information_gain"])
                seen = set()
                accepted = []
                reasons = Counter()
                for c in raw:
                    quality = validator.validate(current[local]["query"], c["text"], seen)
                    if not quality.accepted:
                        reasons.update(quality.reasons)
                        continue
                    accepted.append(dict(c, quality_pass=True, constraint_preservation=quality.constraint_preservation,
                                         rank=len(accepted)+1))
                output[local]["draws"].append(dict(seed=seed, candidates=accepted[:3],
                    raw_count=len(raw), quality_pass_count=len(accepted), quality_reasons=dict(reasons),
                    evaluated_positions=len(current[local]["query"].split())+1))
            print(f"Chunk {chunk}: seed {seed} done in {time.perf_counter()-seed_started:.1f}s", flush=True)
        with path.open("a", encoding="utf-8") as stream:
            for row in output:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            stream.flush()
        elapsed = time.perf_counter()-started
        print(f"Generated {chunk+len(current)}/{len(sampled)} queries x8 seeds; elapsed {elapsed:.1f}s", flush=True)
        dump(OUT/"generation_status.json", dict(completed=chunk+len(current), total=len(sampled),
             elapsed_session_seconds=elapsed, test_used=False,
             generated_rows_sha256=sha256(path) if chunk+len(current)==len(sampled) else None))
    del model
    torch.cuda.empty_cache()


def base_features(row, c, qv, cv, original, scores, otop, ctop):
    f = feature_row(row["query"], c, qv, cv, original, scores)
    # Scalar Top-10 statistics are invariant to equal-score tie order. The
    # document overlap must use the same score/index order as actual labels.
    f["top10_overlap"] = len(set(otop[:10]) & set(ctop[:10]))/10.
    ov, av = original[otop], scores[ctop]
    corr = float(np.corrcoef(original, scores)[0, 1])
    f.update(top100_overlap=len(set(otop)&set(ctop))/100.,
             top100_rrf_agreement=weighted_rank_agreement(otop, ctop),
             top10_rank_stability=weighted_rank_agreement(otop[:10], ctop[:10]),
             score_distribution_correlation=corr if np.isfinite(corr) else 0.,
             original_top100_mean=float(ov.mean()), original_top100_std=float(ov.std()),
             candidate_top100_mean=float(av.mean()), candidate_top100_std=float(av.std()),
             top100_mean_delta=float(av.mean()-ov.mean()), top100_std_delta=float(av.std()-ov.std()),
             top1_document_stable=float(otop[0] == ctop[0]))
    result = np.array([f[k] for k in BASE_FEATURES])
    assert np.isfinite(result).all()
    return result


def sampling_features(row, vectors, tops, appearances):
    n = len(vectors)
    if not n:
        return np.zeros((0, len(SEED_FEATURES)))
    cos = np.clip(vectors@vectors.T, -1, 1)
    a10 = np.eye(n)
    a100 = np.eye(n)
    for i in range(n):
        for j in range(i):
            a10[i, j] = a10[j, i] = weighted_rank_agreement(tops[i][:10], tops[j][:10])
            a100[i, j] = a100[j, i] = weighted_rank_agreement(tops[i], tops[j])
    groups = [[i for i, seeds in enumerate(appearances) if s in seeds] for s in SEEDS]
    # Each draw has equal weight; duplicate samples preserve seed frequency.
    weights = np.zeros(n)
    for group in groups:
        if group:
            weights[group] += 1/(len(SEEDS)*len(group))
    weights /= max(weights.sum(), 1e-12)
    semdisp = float(1-weights@cos@weights)
    retdisp = float(1-weights@a10@weights)
    counts = Counter()
    for group in groups:
        for i in group:
            for d in tops[i][:10]:
                counts[int(d)] += 1/(len(SEEDS)*len(group))
    raw = sum(len(d["candidates"]) for d in row["draws"])
    quality = np.mean([d["quality_pass_count"]/max(d["raw_count"], 1) for d in row["draws"]])
    out = []
    for i in range(n):
        origin = next(s for s in SEEDS if s in appearances[i])
        # Exclude one originating seed; there is no comparison with the exact
        # same draw alone. Missing other seed candidates get support zero.
        c = []; t10 = []; t100 = []
        for seed, group in zip(SEEDS, groups):
            if seed == origin:
                continue
            c.append(float(cos[i, group].max()) if group else 0.)
            t10.append(float(a10[i, group].max()) if group else 0.)
            t100.append(float(a100[i, group].max()) if group else 0.)
        out.append([len(appearances[i])/len(SEEDS), np.mean(c), np.std(c),
                    np.mean(t10), np.std(t10), np.mean(t100), np.std(t100),
                    np.mean([counts[int(d)] for d in tops[i][:10]]), semdisp, retdisp,
                    n/max(raw, 1), quality])
    result = np.asarray(out)
    assert result.shape == (n, len(SEED_FEATURES)) and np.isfinite(result).all()
    return result


def retrieval(protocol):
    path = OUT/"evidence.pkl"
    if path.exists():
        print("Retrieval evidence already complete", flush=True)
        return
    rows = read_jsonl(OUT/"generation.jsonl")
    assert len(rows) == sum(COUNTS.values())
    assert all(len(r["draws"]) == len(SEEDS) for r in rows)
    assert all(tuple(d["seed"] for d in r["draws"]) == SEEDS for r in rows)
    cachemeta = json.loads((CACHE/"manifest.json").read_text())
    assert cachemeta["valid_sha256"] == protocol["valid_sha256"]
    assert all(sha256(Path(p)) == h for p, h in cachemeta["weights"].items())
    assert all(sha256(Path(p)) == h for p, h in cachemeta["tokenizer"].items())
    codes = np.load(CACHE/"code_vectors.npy")
    valid = read_jsonl(VALID)
    assert len(codes) == len(valid) and np.isfinite(codes).all()
    candidates = []; appearances = []; texts = []
    for row in rows:
        lookup = {}; unique = []; seeds = []
        for draw in row["draws"]:
            for c in draw["candidates"]:
                key = CandidateQualityValidator.canonical(c["text"])
                if key not in lookup:
                    lookup[key] = len(unique); unique.append(dict(c)); seeds.append(set())
                i = lookup[key]; seeds[i].add(draw["seed"])
        candidates.append(unique); appearances.append(seeds)
        texts.extend(c["text"] for c in unique)
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(RETRIEVER).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER)
    qvectors = encode(model, tokenizer, [r["query"] for r in rows], 48, 256, "cuda")
    cvectors = encode(model, tokenizer, texts, 48, 256, "cuda")
    del model
    torch.cuda.empty_cache()
    np.save(OUT/"query_vectors.npy", qvectors)
    np.save(OUT/"candidate_vectors.npy", cvectors)
    evidence = []; offset = 0; start = time.perf_counter(); label_checks = 0
    from crsu.core import RankingStates
    for i, (row, cs, apps, qv) in enumerate(zip(rows, candidates, appearances, qvectors)):
        qi = row["query_index"]
        assert valid[qi]["url"] == row["url"]
        original = codes@qv
        otop = stable_indices(original, 100)
        op = np.flatnonzero(otop == qi)
        rr0 = 1/(int(op[0])+1) if len(op) else 0.
        n = len(cs); vectors = cvectors[offset:offset+n]; offset += n
        scores = vectors@codes.T
        tops = [stable_indices(s, 100) for s in scores]
        base = np.asarray([base_features(row, c, qv, cv, original, s, otop, top)
                           for c, cv, s, top in zip(cs, vectors, scores, tops)]).reshape(n, len(BASE_FEATURES))
        extra = sampling_features(row, vectors, tops, apps)
        yy = {"rewrite": [], "fusion": []}
        for top in tops:
            for family, ranking in (("rewrite", top), ("fusion", np.asarray(rrf(otop, top)))):
                pos = np.flatnonzero(ranking == qi)
                rr = 1/(int(pos[0])+1) if len(pos) else 0.
                yy[family].append(rr-rr0)
        for family in yy:
            action_lists = {str(j): top if family == "rewrite" else rrf(otop, top)
                            for j, top in enumerate(tops)}
            state = RankingStates.from_rankings(otop, action_lists, cutoff=100)
            gold_row = state.document_ids.index(qi) if qi in state.document_ids else len(state.document_ids)
            induced = state.gains[gold_row]
            np.testing.assert_allclose(induced, yy[family], rtol=0, atol=TOL)
            label_checks += len(induced)
        pool101 = np.array([101 in s for s in apps], bool)
        shortlist = np.zeros(n, bool)
        if n:
            bestig = []
            for c in cs:
                key = CandidateQualityValidator.canonical(c["text"])
                bestig.append(max(z["information_gain"] for d in row["draws"] for z in d["candidates"]
                                  if CandidateQualityValidator.canonical(z["text"]) == key))
            shortlist[np.argsort(-np.asarray(bestig), kind="stable")[:3]] = True
        evidence.append(dict(query_index=qi, repo=repository(row["url"]), role=row["role"],
                             x=base, seed_x=extra, original_rr=rr0,
                             y={f: np.asarray(v) for f, v in yy.items()},
                             pools=dict(seed101=pool101, multi8=np.ones(n, bool), multi8_entropy_top3=shortlist),
                             candidates=cs, seed_appearances=[sorted(s) for s in apps],
                             original_top100=otop, candidate_top100=tops))
        if (i+1)%40 == 0:
            print(f"Retrieved/audited {i+1}/{len(rows)} queries; {time.perf_counter()-start:.1f}s", flush=True)
    assert offset == len(cvectors)
    with path.open("wb") as stream:
        pickle.dump(evidence, stream)
    dump(OUT/"retrieval_status.json", dict(test_used=False, queries=len(rows), unique_candidates=len(texts),
        metric_policy="actual_list_v2", elapsed_seconds=time.perf_counter()-start,
        state_induced_labels_checked=label_checks, label_mismatches=0))


def flatten(items, family):
    n = [len(x["x"]) for x in items]
    return dict(x=np.concatenate([x["x"] for x in items]),
                seed_x=np.concatenate([x["seed_x"] for x in items]),
                y=np.concatenate([x["y"][family] for x in items]),
                q=np.concatenate([np.full(k, x["query_index"]) for k, x in zip(n, items)]),
                repo=np.concatenate([np.full(k, x["repo"], object) for k, x in zip(n, items)]),
                pools={p: np.concatenate([x["pools"][p] for x in items]) for p in items[0]["pools"]},
                query_ids=[x["query_index"] for x in items], query_repos=[x["repo"] for x in items],
                query_count=len(items), weights=np.concatenate([np.full(k, 1/k) for k in n if k]))


def first(data, pred, pool):
    return np.array([ix[np.argmax(pred["mu"][ix])] for q in data["query_ids"]
                     for ix in [np.flatnonzero((data["q"] == q)&data["pools"][pool])]
                     if len(ix)], dtype=int)


def decision(data, ix):
    ix = np.asarray(ix, int); y = data["y"][ix]
    assert len(np.unique(data["q"][ix])) == len(ix)
    values = {int(data["q"][j]): float(data["y"][j]) for j in ix}
    return dict(accepted=len(ix), coverage=len(ix)/data["query_count"],
                delta_mrr=float(y.sum()/data["query_count"]),
                repo_ci=clustered_ci(data, values, 11120, 1000),
                positive=int((y>TOL).sum()), zero=int((abs(y)<=TOL).sum()), harmful=int((y<-TOL).sum()),
                harmful_rate=float(np.mean(y<-TOL)) if len(y) else None,
                gross_gain_per_all_query=float(np.maximum(y, 0).sum()/data["query_count"]),
                loss_per_all_query=float(np.maximum(-y, 0).sum()/data["query_count"]),
                query_values=values)


def train_models(data):
    xs = {"base": data["x"], "seed_features": np.column_stack((data["x"], data["seed_x"]))}
    y = data["y"]; w = data["weights"]; models = {}; base_error = None
    assert len(np.unique(data["repo"])) >= 4
    for name, x in xs.items():
        print(f"Fit {name}, {len(y)} train actions, {x.shape[1]} features", flush=True)
        oof = np.zeros(len(y)); pieces = []
        for held in grouped_folds(data["repo"], 4, 11120):
            tr = np.setdiff1d(np.arange(len(y)), held)
            assert not set(data["repo"][held])&set(data["repo"][tr])
            m = booster(x[tr], y[tr], 11120, weight=w[tr])
            oof[held] = m.predict(x[held]); pieces.append(dict(fit_repos=sorted(set(data["repo"][tr])),
                                                            held_repos=sorted(set(data["repo"][held]))))
        error = np.log(abs(y-oof)+.001)
        if name == "base":
            base_error = error
        models[name] = dict(mean=booster(x, y, 11120, weight=w),
                            scale=booster(x, error, 11120, weight=w),
                            harm=booster(x, (y<-TOL).astype(int), 11120, "binary", w),
                            constant=float(np.average(abs(y-oof)+.001, weights=w)), oof_roles=pieces)
    extra = xs["seed_features"]
    models["base_mean_seed_scale"] = dict(models["base"])
    models["base_mean_seed_scale"]["scale"] = booster(extra, base_error, 11120, weight=w)
    return models


def predict(model, data, name):
    base = data["x"]; full = np.column_stack((base, data["seed_x"]))
    if not len(base):
        return {k: np.zeros(0) for k in ("mu", "s", "harm")}
    mx = full if name == "seed_features" else base
    sx = base if name == "base" else full
    return dict(mu=model["mean"].predict(mx), s=np.maximum(np.exp(model["scale"].predict(sx)), .001),
                harm=np.clip(model["harm"].predict(mx), 0, 1))


def curve(data, pred, pool, score, coverage=.1):
    ix = first(data, pred, pool)
    rank = ix[np.argsort(-score[ix], kind="stable")]
    return decision(data, rank[:min(len(rank), round(coverage*data["query_count"]))])


def risk_policy(data, preds, pool, gate):
    sel = data["selection_valid"]; ps = preds["selection_valid"]; si = first(sel, ps, pool)
    score = ps["mu"] if gate == "mean" else ps["lower"]
    choices = []
    for t in (.05, .1, .2, .4, 1.):
        ix = si[(score[si]>0)&(ps["harm"][si]<=t)]
        y = sel["y"][ix]
        if len(ix) and np.mean(y<-TOL)<=.1 and y.sum()>0:
            choices.append((float(y.sum()), len(ix), t))
    threshold = max(choices)[2] if choices else -1.
    risk = data["risk_calibration"]; pr = preds["risk_calibration"]; ri = first(risk, pr, pool)
    score = pr["mu"] if gate == "mean" else pr["lower"]
    ra = ri[(score[ri]>0)&(pr["harm"][ri]<=threshold)]
    n = len(ra); h = int((risk["y"][ra]<-TOL).sum())
    upper = float(stats.beta.ppf(.95, h+1, n-h)) if n>h else 1.
    passed = n>0 and upper<=.1
    ev = data["new_holdout"]; pe = preds["new_holdout"]; ei = first(ev, pe, pool)
    score = pe["mu"] if gate == "mean" else pe["lower"]
    ea = ei[(score[ei]>0)&(pe["harm"][ei]<=threshold)]
    return dict(threshold=threshold, risk_accepted=n, risk_harmful=h, query_iid_cp95_upper=upper,
                passes_descriptive_check=passed, evaluation_before_risk=decision(ev, ea),
                evaluation_after_risk=decision(ev, ea if passed else []),
                caveat="query-binomial diagnostic; repo dependence and multiple policy comparisons not certified")


def evaluate(protocol):
    if (OUT/"summary.json").exists():
        print("Evaluation already complete", flush=True)
        return
    with (OUT/"evidence.pkl").open("rb") as stream:
        items = pickle.load(stream)
    roles = {r: [x for x in items if x["role"] == r] for r in COUNTS}
    repos = {r: {x["repo"] for x in v} for r, v in roles.items()}
    assert all(not repos[a]&repos[b] for a in repos for b in repos if a != b)
    result = dict(test_used=False, development_only=True, exploratory=True, protocol_sha256=sha256(OUT/"PROTOCOL.json"),
                  roles={r: dict(queries=len(v), repos=len(repos[r])) for r, v in roles.items()},
                  actions={}, sampling={})
    for r, vv in roles.items():
        result["sampling"][r] = dict(mean_unique_candidates=float(np.mean([len(x["x"]) for x in vv])),
            mean_seed101_candidates=float(np.mean([x["pools"]["seed101"].sum() for x in vv])),
            mean_cross_seed_retrieval_dispersion=float(np.mean([x["seed_x"][0, 9] for x in vv if len(x["x"])])))
    saved_predictions = {}
    for family in ("rewrite", "fusion"):
        data = {r: flatten(v, family) for r, v in roles.items()}
        models = train_models(data["utility_train"])
        joblib.dump(models, OUT/f"{family}_models.joblib")
        # Verify serialization before using external-role labels for summaries.
        restored = joblib.load(OUT/f"{family}_models.joblib")
        preds = {name: {r: predict(model, d, name) for r, d in data.items() if r != "utility_train"}
                 for name, model in models.items()}
        for r in preds["base"]:
            for k in ("mu", "harm"):
                np.testing.assert_array_equal(preds["base"][r][k], preds["base_mean_seed_scale"][r][k])
        for name in models:
            check = predict(restored[name], data["lcb_calibration"], name)
            for k in check:
                np.testing.assert_allclose(check[k], preds[name]["lcb_calibration"][k], atol=TOL, rtol=0)
        out = dict(pools={}, methods={}, paired={})
        ev = data["new_holdout"]
        for pool in ("seed101", "multi8", "multi8_entropy_top3"):
            available = ev["pools"][pool]; y = ev["y"][available]
            oracle = []
            for q in ev["query_ids"]:
                ix = np.flatnonzero((ev["q"] == q)&available)
                if len(ix) and ev["y"][ix].max()>TOL:
                    oracle.append(ix[int(np.argmax(ev["y"][ix]))])
            out["pools"][pool] = dict(actions=int(available.sum()),
                positive_fraction=float(np.mean(y>TOL)), zero_fraction=float(np.mean(abs(y)<=TOL)),
                harmful_fraction=float(np.mean(y<-TOL)), candidate_mean=float(y.mean()),
                queries_with_positive=len(oracle), oracle_with_fallback=decision(ev, oracle))
            for name, pred_roles in preds.items():
                cal = data["lcb_calibration"]; pc = pred_roles["lcb_calibration"]
                ci = first(cal, pc, pool)
                correction = max(finite_quantile((pc["mu"][ci]-cal["y"][ci])/pc["s"][ci]), 0.)
                constant_correction = max(finite_quantile(pc["mu"][ci]-cal["y"][ci]), 0.)
                pp = {r: dict(p, lower=p["mu"]-correction*p["s"], constant_lower=p["mu"]-constant_correction)
                      for r, p in pred_roles.items()}
                pe = pp["new_holdout"]; ix = first(ev, pe, pool)
                available = ev["pools"][pool]; y = ev["y"][available]; mu = pe["mu"][available]
                rho = stats.spearmanr(pe["s"][available], abs(y-mu)).statistic
                key = f"{pool}__{name}"
                metrics = dict(mean_mae=float(np.mean(abs(y-mu))),
                    harmful_brier=float(np.mean((pe["harm"][available]-(y<-TOL))**2)),
                    positive_auprc=float(average_precision_score(y>TOL, mu)) if np.any(y>TOL) else None,
                    error_scale_spearman=float(rho) if np.isfinite(rho) else None,
                    mean_selected_queries=len(ix),
                    lcb_correction=correction, first_choice_lower_violation=float(np.mean(ev["y"][ix]<pe["lower"][ix])),
                    positive_lcb=decision(ev, ix[pe["lower"][ix]>0]),
                    first_choice_oracle_adoption_ceiling=decision(ev, ix[ev["y"][ix]>TOL]),
                    curves={score: {str(c): curve(ev, pe, pool, value, c) for c in (.05, .1, .2)}
                            for score, value in (("mean", pe["mu"]), ("lcb", pe["lower"]),
                                                 ("constant_lcb", pe["constant_lower"]), ("negative_harm", -pe["harm"]))},
                    risk_mean=risk_policy(data, pp, pool, "mean"), risk_lcb=risk_policy(data, pp, pool, "lcb"))
                out["methods"][key] = metrics
                saved_predictions[f"{family}__{key}"] = pe
        comparisons = [
            ("pool_expansion", "multi8__base", "seed101__base", "mean"),
            ("seed_features_same_pool", "multi8__seed_features", "multi8__base", "mean"),
            ("seed_scale_same_mean", "multi8__base_mean_seed_scale", "multi8__base", "lcb"),
        ]
        for label, a, b, score in comparisons:
            va = out["methods"][a]["curves"][score]["0.1"]["query_values"]
            vb = out["methods"][b]["curves"][score]["0.1"]["query_values"]
            delta = {q: va.get(q, 0)-vb.get(q, 0) for q in ev["query_ids"]}
            out["paired"][label] = dict(delta_difference=float(sum(delta.values())/ev["query_count"]),
                repo_ci=clustered_ci(ev, delta, 11120, 1000),
                caveat="fixed fitted models and accepted sets; exploratory, no multiple comparison correction")
        result["actions"][family] = out
    with (OUT/"predictions.pkl").open("wb") as stream:
        pickle.dump(saved_predictions, stream)
    assert sha256(STABLE) == protocol["stable_sha256"]
    dump(OUT/"summary.json", result)
    dump(OUT/"reproducibility.json", dict(test_used=False, stable_unchanged=True, model_roundtrip=True,
        fixed_mean_scale_ablation_exact=True,
        repository_roles_disjoint=True, source_code_sha256=sha256(Path(__file__)),
        inputs={str(p): sha256(p) for p in (OUT/"PROTOCOL.json", OUT/"generation.jsonl", OUT/"evidence.pkl")},
        summary_sha256=sha256(OUT/"summary.json"), library_versions=dict(torch=torch.__version__)))
    lines = ["# T5 多 seed 一致性小样本验证", "", "探索性 valid-only；冻结 T5/CodeBERT，8 个解码 seed，480 queries，原查询≤64词。",
             "单 seed101 在本轮批次下重新生成，不是历史候选逐字重放。训练与角色按 repository 隔离；不是新 test。", "",
             "## 候选池与选择", "", "| 动作 | 候选池 | 动作数 | 有正候选query | Oracle ΔMRR | 候选harm |",
             "|---|---|---:|---:|---:|---:|"]
    for family, out in result["actions"].items():
        for pool, p in out["pools"].items():
            lines.append(f"| {family} | {pool} | {p['actions']} | {p['queries_with_positive']} | {p['oracle_with_fallback']['delta_mrr']:+.6f} | {p['harmful_fraction']:.3f} |")
    lines += ["", "## 固定采用10%（128-query评估，采用13）", "", "| 动作 | 候选池/模型 | mean ΔMRR | mean harm | LCB ΔMRR | LCB harm | LCB正采用 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for family, out in result["actions"].items():
        for name, p in out["methods"].items():
            m = p["curves"]["mean"]["0.1"]; l = p["curves"]["lcb"]["0.1"]
            mh = f"{m['harmful_rate']:.3f}" if m['harmful_rate'] is not None else "N/A"
            lh = f"{l['harmful_rate']:.3f}" if l['harmful_rate'] is not None else "N/A"
            lines.append(f"| {family} | {name} | {m['delta_mrr']:+.6f} | {mh} | {l['delta_mrr']:+.6f} | {lh} | {p['positive_lcb']['accepted']} |")
    lines += ["", "## 配对收益差（10%采用）", "", "| 动作 | 比较 | ΔMRR差 | repo CI |", "|---|---|---:|---|"]
    for family, out in result["actions"].items():
        for name, p in out["paired"].items():
            lines.append(f"| {family} | {name} | {p['delta_difference']:+.6f} | {p['repo_ci']} |")
    lines += ["", "限制：生成重复采样分歧不是严格认知不确定性；候选池扩增与特征增量分开报告。",
              "同一mean的seed尺度消融使用训练内repository OOF残差；不能把全回退写成方法正向。",
              "风险上界只是query-iid描述性量，未处理repo依赖或多策略比较，不是正式认证。",
              "新seed不会使历史valid成为独立确认集。详见summary.json与PROTOCOL.json。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({f: dict(pools=o["pools"], paired=o["paired"]) for f, o in result["actions"].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "generation", "retrieval", "evaluate", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        protocol = prepare()
        if args.stage in ("all", "generation"):
            generation(protocol)
        if args.stage in ("all", "retrieval"):
            retrieval(protocol)
        if args.stage in ("all", "evaluate"):
            evaluate(protocol)


if __name__ == "__main__":
    main()
