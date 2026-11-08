"""Definitions, training-only preparation, sample planning and inference guard."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from scipy import stats

from experiments.rene_fixed_action_uq_20261019 import model as heads
from experiments.rene_fixed_action_uq_20261019.run import fixed_traces, train_data
from experiments.rene_strict_confirmation_20261019.common import (
    DATA, assert_disjoint, canonical_bytes, check_hashes, load, observable,
    protected_hashes, read_json, require_runtime, save, sha, utc, write_json,
)

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/rene_uq_independent_proof_20261019"
FRESH = OUT / "fresh"
PRIOR = ROOT / "results/rene_strict_confirmation_20261019"
DEV = PRIOR / "development"
ARMS = ("no_seed_uq", "seed_uq")
POLICIES = ("threshold075", "fixed50", "all")
SEED, TOL = 26101927, 1e-10


def event(stage, **data):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "EVENTS.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(dict(utc=utc(), stage=stage, **data), ensure_ascii=False) + "\n")


def backend():
    """Private module instance: redirects new writes; original files stay intact."""
    path = ROOT / "experiments/rene_strict_confirmation_20261019/fresh_data.py"
    spec = importlib.util.spec_from_file_location("_uq_proof_data_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FRESH, module.OUT = FRESH, OUT
    module.freeze = check_protocol
    module.inventory = lambda: (_ for _ in ()).throw(RuntimeError("Legacy inventory disabled"))
    assert module.FRESH.is_relative_to(OUT) and module.OUT == OUT
    return module


def check_protocol():
    p = read_json(FRESH / "PROTOCOL.json")
    check_hashes(p["binding_hashes"])
    check_hashes(p["protected_hashes"])
    return p


def deny_outcome_reads():
    def hook(name, args):
        if name != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        path = str(args[0]).replace("\\", "/").casefold()
        forbidden = (
            "private_outcomes", "/data/codexglue/", "/development/items.pkl",
            "/development/scores.pkl", "/per_query_outcomes.pkl",
            "/rene_fixed_action_uq_20261019/per_query.pkl",
            "/rene_uq_independent_proof_20261019/training/",
        )
        if any(x in path for x in forbidden) or path.endswith("/results.json"):
            raise PermissionError("Label/source/result read forbidden during inference: " + path)
    sys.addaudithook(hook)


def prepare_heads():
    manifest_path = OUT / "HEADS_FROZEN.json"
    if manifest_path.exists():
        m = read_json(manifest_path)
        check_hashes(m["outputs"])
        return m
    items, scores = load(DEV / "items.pkl"), load(DEV / "scores.pkl")
    byid = {int(e["query_index"]): e for e in items}
    scoreby = {int(e["query_index"]): e for e in scores}
    roles = read_json(DEV / "DESIGN.json")["final_roles"]
    assert_disjoint({r: [byid[q]["repo"] for q in ids] for r, ids in roles.items()})
    fits, outputs = {}, {}
    for arm in ARMS:
        pipeline = joblib.load(PRIOR / f"models/fusion_{arm}.joblib")
        assert pipeline["roles"] == roles
        for role in ("gain", "update"):
            assert set(pipeline[role]["fit_queries"]) <= set(roles[role])
            assert set(pipeline[role]["fit_repos"]) <= {byid[q]["repo"] for q in roles[role]}
        risk_items = [byid[q] for q in roles["risk"]]
        traces, arrays = fixed_traces(pipeline, risk_items, [scoreby[q] for q in roles["risk"]])
        data = train_data(risk_items, arrays, traces, "fusion")
        models = {h: heads.fit(data, h) for h in heads.HEADS}
        common = models["common"]
        for k in ("fit_x_sha256", "fit_labels_sha256"):
            assert common[k] == pipeline["risk"][k]
        a = heads.predict(common, arrays)["classes"]
        b = heads.upstream.risk.probabilities(pipeline["risk"], dict(x=arrays["common"], native=arrays["native"]))[1]
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0.)
        training_path = OUT / f"training/{arm}.pkl"
        save(training_path, dict(arrays=arrays, training=data, fixed=traces, roles=roles))
        path = OUT / f"models/{arm}.pkl"
        save(path, dict(models=models, roles=roles,
                        common_probability_replay_max_error=float(abs(a-b).max()),
                        source_pipeline_sha256=sha(PRIOR / f"models/fusion_{arm}.joblib"),
                        training_provenance_path=training_path.relative_to(ROOT).as_posix(),
                        training_provenance_sha256=sha(training_path)))
        outputs[path.relative_to(ROOT).as_posix()] = sha(path)
        fits[arm] = dict(risk_queries=len(risk_items), adopted_training_actions=len(data["q"]),
                         counts=common["counts"], fit_repos=data["repo"],
                         active_features={h: int(models[h]["mask"].sum()) for h in heads.HEADS})
        print(f"Training-only heads {arm}: {len(data['q'])} actions, three real refits", flush=True)
    m = dict(completed_at_utc=utc(), roles=roles, fits=fits, outputs=outputs,
             new_query_inference_started=False, risk_heads=6)
    write_json(manifest_path, m, immutable=True)
    event("risk_heads_frozen", manifest_sha256=sha(manifest_path))
    return read_json(manifest_path)


def plan_power():
    path = OUT / "POWER.json"
    if path.exists():
        return read_json(path)
    vectors = load(ROOT / "results/rene_fixed_action_uq_20261019/PER_QUERY.pkl")["fusion"]["no_seed_uq"]
    generation = [json.loads(line) for line in (DATA / "generation.jsonl").open(encoding="utf8") if line.strip()]
    generation_blocks = {int(r["query_index"]): i // 8 for i, r in enumerate(generation)}
    observations = []
    for before in ("support", "common"):
        a, b = vectors[before]["threshold075"], vectors["full"]["threshold075"]
        assert [r["q"] for r in a] == [r["q"] for r in b]
        differences = np.array([y["delta"] - x["delta"] for x, y in zip(a, b)])
        ids = sorted({generation_blocks[r["q"]] for r in a})
        block_means = np.array([
            np.mean([d for r, d in zip(a, differences) if generation_blocks[r["q"]] == g])
            for g in ids
        ])
        assert len(block_means) == 64 and len(a) == 512
        sd = float(block_means.std(ddof=1))
        upper = sd * math.sqrt(63 / stats.chi2.ppf(.2, 63))
        observations.append(dict(contrast=f"full_minus_{before}", block_sd=sd, block_sd_upper80=upper,
                                 queries=512, generation_blocks=64, development_increment=float(differences.mean())))
    planning_sd = max(.05 / math.sqrt(8), max(z["block_sd_upper80"] for z in observations))
    zsum = stats.norm.ppf(1-.05/(2*2)) + stats.norm.ppf(.8)
    blocks = int(math.ceil((zsum * planning_sd / .005) ** 2))
    n = int(math.ceil(max(1024, blocks*8) / 128) * 128)
    result = dict(
        planned_queries=n, planned_generation_blocks=n//8, target_effect=.005,
        equivalence_margin=.005, target_power=.8, familywise_alpha=.05, key_contrasts=2,
        block_size=8, planning_block_sd=planning_sd, z_sum=float(zsum),
        observations=observations, fresh_outcomes_opened=False,
        approximation="normal planning using cluster-SD upper80; floor query SD .05 / sqrt(8)",
        inputs={str((ROOT/"results/rene_fixed_action_uq_20261019/PER_QUERY.pkl").relative_to(ROOT)):sha(ROOT/"results/rene_fixed_action_uq_20261019/PER_QUERY.pkl"),
                str((DATA/"generation.jsonl").relative_to(ROOT)):sha(DATA/"generation.jsonl")},
        no_risk_certificate_power_claim=True,
    )
    write_json(path, result, immutable=True)
    event("power_fixed", planned_queries=n, power_sha256=sha(path))
    return result


def prepare_roster(n):
    path = FRESH / "ROSTER_METADATA.json"
    if path.exists():
        return read_json(path)
    from transformers import AutoTokenizer
    from experiments.rene_strict_confirmation_20261019.fresh_data import repository, code_key, T5
    inv = read_json(PRIOR / "fresh/INVENTORY.json")
    check_hashes(inv["inputs"])
    eligible = load(PRIOR / "fresh/eligible_repositories.pkl")
    strict = read_json(PRIOR / "fresh/queries.json")
    prior_repos = {r["repo"] for r in strict}
    corpus_path = PRIOR / "fresh/corpus/block0.pkl"
    corpus = load(corpus_path)
    docby = {code_key(r["code_text"]): int(r["doc_id"]) for r in corpus}
    assert len(docby) == len(corpus) == 4096
    # Existing test is consumed: repository metadata only, no metric/target use.
    test_path = ROOT / "data/codexglue/advtest/test.jsonl"
    test_repos = set()
    for line in test_path.open(encoding="utf8"):
        row = json.loads(line)
        test_repos.add(str(row.get("repo") or repository(row["url"])).casefold())
    tokenizer = AutoTokenizer.from_pretrained(T5, local_files_only=True)
    sentinel = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    frame, rejected = [], dict(test_repository=0, t5_truncation=0)
    for row in eligible:
        if row["repo"] in prior_repos or row["code_hash"] not in docby:
            continue
        if row["repo"] in test_repos:
            rejected["test_repository"] += 1
            continue
        words = row["query"].split()
        texts = [" ".join(words[:i]+["<extra_id_0>"]+words[i:]) for i in range(len(words)+1)]
        enc = tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
        if any(len(ids) > 512 or ids.count(sentinel) != 1 for ids in enc):
            rejected["t5_truncation"] += 1
            continue
        frame.append(dict(row, t5_max_input_tokens=max(map(len, enc))))
    frame.sort(key=lambda r: (hashlib.sha256(f"{SEED}:uq-risk-proof:{r['repo']}".encode()).hexdigest(), r["repo"]))
    if len(frame) < n:
        raise RuntimeError(f"Insufficient clean frame: {len(frame)} < {n}; no underpowered substitute")
    roster = frame[:n]
    assert len({r["repo"] for r in roster}) == n
    assert len({r["query_hash"] for r in roster}) == n
    assert len({r["code_hash"] for r in roster}) == n
    queries, outcomes = [], []
    for i, row in enumerate(roster):
        qid = 2000000+i
        queries.append(dict(query_index=qid, query=row["query"], repo=row["repo"],
                            block=0, url=f"opaque-query:{qid}"))
        outcomes.append(dict(query_index=qid, target_doc_id=docby[row["code_hash"]],
                             repo=row["repo"], source_index=row["source_index"],
                             query_hash=row["query_hash"], code_hash=row["code_hash"]))
    write_json(FRESH/"queries.json", queries, immutable=True)
    write_json(FRESH/"private_outcomes.json", outcomes, immutable=True)
    meta = dict(
        prepared_at_utc=utc(), sample=n, original_clean_unused_fixed_corpus_frame=2560,
        eligible_after_metadata_filters=len(frame), rejected=rejected,
        sample_query_repositories=[r["repo"] for r in roster],
        maximum_t5_input_tokens=max(r["t5_max_input_tokens"] for r in roster),
        t5_sentinel_survives_all_positions=True, strict_query_overlap=0, old_test_query_overlap=0,
        corpus_documents=4096, corpus_path=corpus_path.relative_to(ROOT).as_posix(),
        corpus_sha256=sha(corpus_path), query_sha256=sha(FRESH/"queries.json"),
        private_mapping_sha256=sha(FRESH/"private_outcomes.json"),
        note="Target mapping construction is preparer-only; no retrieval outcome has been computed.",
        upstream_source_train_exposure=True, target_codes_previously_unlabeled_corpus_distractors=True,
        input_hashes={
            str((PRIOR/"fresh/INVENTORY.json").relative_to(ROOT)):sha(PRIOR/"fresh/INVENTORY.json"),
            str((PRIOR/"fresh/eligible_repositories.pkl").relative_to(ROOT)):sha(PRIOR/"fresh/eligible_repositories.pkl"),
            str((PRIOR/"fresh/queries.json").relative_to(ROOT)):sha(PRIOR/"fresh/queries.json"),
            str(test_path.relative_to(ROOT)):sha(test_path),
        },
    )
    write_json(path, meta, immutable=True)
    event("roster_fixed", queries=n, eligible_frame=len(frame), roster_sha256=sha(path))
    return read_json(path)


def prepare():
    path = FRESH/"PROTOCOL.json"
    if path.exists():
        return check_protocol()
    runtime = require_runtime(gpu=True)
    tests = read_json(OUT/"TESTS.json")
    assert tests["passed"]
    check_hashes(tests["test_sources"])
    import importlib.metadata
    runtime["packages"] = {name:importlib.metadata.version(name) for name in (
        "transformers", "scipy", "scikit-learn", "joblib", "lightgbm", "tokenizers")}
    head_manifest = prepare_heads()
    power = plan_power()
    meta = prepare_roster(power["planned_queries"])
    from experiments.rene_strict_confirmation_20261019.fresh_data import model_assets
    source_paths = sorted(set(
        list(ROOT.glob("*.py")) + list((ROOT/"crsu").rglob("*.py")) +
        list((ROOT/"experiments").rglob("*.py"))
    ))
    binding = {p.relative_to(ROOT).as_posix():sha(p) for p in source_paths}
    binding.update({str(Path(p)).replace("\\","/"):h for p,h in model_assets().items()})
    binding.update(head_manifest["outputs"])
    for p in [OUT/"HEADS_FROZEN.json", OUT/"POWER.json", OUT/"TESTS.json", HERE/"DESIGN.md",
              FRESH/"queries.json", FRESH/"ROSTER_METADATA.json",
              PRIOR/"models/fusion_no_seed_uq.joblib", PRIOR/"models/fusion_seed_uq.joblib"]:
        binding[p.relative_to(ROOT).as_posix()] = sha(p)
    binding[meta["corpus_path"]] = meta["corpus_sha256"]
    p = dict(
        frozen_at_utc=utc(), format="uq-risk-conditional-independent-confirmation-v1",
        query_count=power["planned_queries"], planned_queries=power["planned_queries"],
        generation_blocks=power["planned_generation_blocks"], seeds=[101,202,303,404],
        arms=ARMS, heads=heads.HEADS, policies=POLICIES, primary_upstream="no_seed_uq",
        key_contrasts=["full_minus_support","full_minus_common"], primary_policy="threshold075",
        family="fusion", target_effect=.005, equivalence_margin=.005, familywise_alpha=.05,
        simultaneous_interval=.975, bootstrap_repeats=10000, bootstrap_seed=SEED,
        generation_block_size=8, equivalence_min_nonzero_queries=20,
        equivalence_min_nonzero_blocks=10, risk_certified=False,
        statistics="8-query paired cluster bootstrap; cluster-t sensitivity; envelope for primary decisions",
        fixed50_analysis="secondary batch ranking diagnostic; adoption recomputed in each bootstrap draw",
        corpus=[dict(block=0,documents=4096,path=meta["corpus_path"],sha256=meta["corpus_sha256"],
                     query_count=power["planned_queries"])],
        generation=dict(chunk_queries=8,do_sample=True,top_k=50,top_p=1.,temperature=1.,
                        max_new_tokens=12,seed_schedule="seed+global chunk start",
                        accepted_top3="quality filter then information_gain top3"),
        downstream_new_query_repositories=True, corpus_reused=True,
        target_codes_previously_unlabeled_corpus_distractors=True,
        upstream_retriever_train_exposure=True, old_test_metrics_used=False,
        old_test_repository_metadata_only=True, new_t5_training=False,
        private_outcomes_sha256=meta["private_mapping_sha256"],
        power_sha256=sha(OUT/"POWER.json"), runtime=runtime,
        binding_hashes=binding, protected_hashes=protected_hashes(),
        preparation_input_hashes={
            str((DEV/"items.pkl").relative_to(ROOT)):sha(DEV/"items.pkl"),
            str((DEV/"scores.pkl").relative_to(ROOT)):sha(DEV/"scores.pkl"),
            **meta["input_hashes"], **power["inputs"],
        },
    )
    write_json(path,p,immutable=True)
    event("confirmation_protocol_frozen",protocol_sha256=sha(path),queries=p["query_count"])
    print(f"Protocol frozen, {p['query_count']} new query repositories; no new outcomes opened",flush=True)
    return read_json(path)
