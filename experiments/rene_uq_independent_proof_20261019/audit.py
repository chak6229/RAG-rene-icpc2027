"""Separate pre-opening contract audit and independent outcome recomputation."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from crsu.core import rrf
from experiments.rene_fixed_action_uq_20261019.audit import independent_probabilities, independent_rr
from experiments.rene_strict_confirmation_20261019 import model as upstream
from experiments.rene_strict_confirmation_20261019.common import (
    check_hashes, load, observable, read_json, reject_gold, require_runtime,
    sha, stable_top, utc, write_json,
)
from experiments.rene_strict_confirmation_20261019.development import compact
from experiments.rene_strict_confirmation_20261019.features import sampling_features, union_draws
from experiments.strong_flow_uq_20261014.build_evidence import unions
from experiments.rene_uq_independent_proof_20261019.study import (
    ARMS, FRESH, OUT, POLICIES, PRIOR, SEED, TOL, check_protocol, deny_outcome_reads, event, heads,
)


def inspect_inference():
    p = check_protocol()
    for file in ("GENERATION_COMPLETE.json", "RETRIEVAL_COMPLETE.json", "EVIDENCE_COMPLETE.json"):
        check_hashes(read_json(FRESH/file)["outputs"])
    for file,key in (
        ("sessions.pkl","sessions_sha256"), ("ce_scores.pkl","scores_sha256"),
        ("label_free_decisions.pkl","decisions_sha256"),
    ):
        seal = {"sessions.pkl":"PLAN_SEAL.json","ce_scores.pkl":"CE_COMPLETE.json",
                "label_free_decisions.pkl":"DECISION_SEAL.json"}[file]
        assert sha(FRESH/file) == read_json(FRESH/seal)[key]
    seal = read_json(FRESH/"DECISION_SEAL.json")
    assert seal["actual_decisions"] and not seal["outcomes_opened"]
    assert seal["protocol_sha256"] == sha(FRESH/"PROTOCOL.json")
    assert seal["observations_sha256"] == sha(FRESH/"observable_items.pkl")
    assert seal["scores_sha256"] == sha(FRESH/"ce_scores.pkl")
    assert seal["plan_seal_sha256"] == sha(FRESH/"PLAN_SEAL.json")
    assert read_json(OUT/"INFERENCE_COMPLETE.json")["actual_decisions_sha256"] == seal["decisions_sha256"]
    assert read_json(OUT/"TESTS.json")["passed"]
    queries = read_json(FRESH/"queries.json")
    items = load(FRESH/"observable_items.pkl")
    scores = load(FRESH/"ce_scores.pkl")
    saved = load(FRESH/"label_free_decisions.pkl")
    sessions = load(FRESH/"sessions.pkl")
    for obj in (items, scores, saved, sessions):
        reject_gold(obj)
    ids = [q["query_index"] for q in queries]
    assert ids == [e["query_index"] for e in items] == [e["query_index"] for e in scores]
    assert len(ids) == len(set(ids)) == p["query_count"]
    assert ids == list(range(2000000,2000000+len(ids))) and len(ids)%8 == 0
    repos = {q["repo"] for q in queries}
    assert len(repos) == len(ids)
    assert not repos & {q["repo"] for q in read_json(PRIOR/"fresh/queries.json")}
    assert not repos & set(read_json(PRIOR/"fresh/EXPOSURE.json")["excluded_repositories"])
    byid = {q["query_index"]:q for q in queries}
    expected_unions = unions(items)
    generated = [r for path in sorted((FRESH/"generation").glob("*.json")) for r in read_json(path)["rows"]]
    assert [r["query_index"] for r in generated] == ids
    assert read_json(FRESH/"GENERATOR_REPLAY.json")["exact"]
    candidate_total, vector_ranking_checks = 0,0
    cv = np.load(FRESH/"observations/block0_code_vectors.npy")
    qv = np.load(FRESH/"observations/block0_query_vectors.npy")
    av = np.load(FRESH/"observations/block0_candidate_vectors.npy")
    assert cv.shape == (4096,768) and qv.shape == (len(ids),768)
    offset = 0
    for i,(e,g,docs) in enumerate(zip(items,generated,expected_unions)):
        assert g["query"] == byid[e["query_index"]]["query"] == e["query"]
        assert tuple(d["seed"] for d in g["draws"]) == tuple(p["seeds"])
        assert all(d["evaluated_positions"] == len(g["query"].split())+1 for d in g["draws"])
        rebuilt, cs, appearances = union_draws(g)
        n = len(cs)
        candidate_total += n
        assert len(e["candidates"]) == n <= 12
        assert e["x"].shape == (n,34) and e["seed_x"].shape == (n,12)
        assert [c["text"] for c in cs] == [c["text"] for c in e["candidates"]]
        assert all(c["text"] != e["query"] and c["quality_pass"] for c in cs)
        assert [set(s) for s in appearances] == [set(s) for s in e["seed_appearances"]]
        np.testing.assert_array_equal(e["strong_evidence"]["doc_ids"],docs)
        np.testing.assert_array_equal(e["original_top100"],stable_top(cv@qv[i]))
        vector_ranking_checks += 1
        vv = av[offset:offset+n]
        offset += n
        for v,top in zip(vv,e["candidate_top100"]):
            np.testing.assert_array_equal(top,stable_top(cv@v))
            assert len(set(map(int,top))) == 100 and all(0 <= int(d) < 4096 for d in top)
            vector_ranking_checks += 1
        np.testing.assert_allclose(
            e["seed_x"], sampling_features(rebuilt,vv,e["candidate_top100"],appearances),atol=1e-12,rtol=0)
    assert len(av) == offset
    maximum_probability_error, probability_rows, action_rows = 0.,0,0
    requested = [set() for _ in items]
    physical_tokens = 0
    for arm in ARMS:
        pipeline = joblib.load(PRIOR/f"models/fusion_{arm}.joblib")
        fitted_bundle = load(OUT/f"models/{arm}.pkl")
        assert fitted_bundle["source_pipeline_sha256"] == sha(PRIOR/f"models/fusion_{arm}.joblib")
        assert not repos & set(r for rs in pipeline["role_repositories"].values() for r in rs)
        views = [observable(e) for e in items]
        predictions = upstream.predict_gain(pipeline["gain"],views)
        fresh_sessions = [upstream.prepare(e,pr,"fusion",pipeline["cut"]) for e,pr in zip(views,predictions)]
        fixed = [upstream.choose_trace(e,s,pipeline["update"],score)
                 for e,s,score in zip(views,fresh_sessions,scores)]
        assert [compact(t) for t in fixed] == [compact(t) for t in saved["fixed"][arm]]
        assert [compact(t) for t in fixed] == saved["outputs"][f"{arm}/pre_gate"]
        arrays = heads.inputs(views,fixed)
        assert arrays["q"] == saved["inputs"][arm]["q"]
        for key in ("common","seed","native"):
            np.testing.assert_array_equal(arrays[key],saved["inputs"][arm][key])
        for i,s in enumerate(sessions[arm]):
            if s["verified"]:
                assert set(s["plan"]["doc_ids"]) <= set(map(int,expected_unions[i]))
                requested[i].update(map(int,s["plan"]["doc_ids"]))
        for h in heads.HEADS:
            model = fitted_bundle["models"][h]
            assert set(model["fit_repos"]) <= set(pipeline["role_repositories"]["risk"])
            assert set(model["fit_query_ids"]) <= set(pipeline["roles"]["risk"])
            np.testing.assert_array_equal(model["mask"],heads.feature_mask(h))
            independent = independent_probabilities(model,arrays)
            recorded = saved["probabilities"][arm][h]["classes"]
            np.testing.assert_allclose(independent,recorded,atol=1e-12,rtol=0)
            np.testing.assert_allclose(independent[:,0],saved["probabilities"][arm][h]["harm"],atol=1e-12,rtol=0)
            maximum_probability_error = max(maximum_probability_error,float(abs(independent-recorded).max()) if len(independent) else 0.)
            probability_rows += len(independent)
            for policy in POLICIES:
                if policy == "threshold075":
                    mask = independent[:,0] <= .075
                else:
                    count = len(independent) if policy == "all" else len(independent)//2
                    chosen = sorted(range(len(independent)),key=lambda k:(independent[k,0],arrays["q"][k]))[:count]
                    mask = np.zeros(len(independent),bool)
                    mask[chosen] = True
                np.testing.assert_array_equal(mask,saved["masks"][arm][h][policy])
                selected = {q for q,use in zip(arrays["q"],mask) if use}
                for e,t,gated in zip(items,fixed,saved["outputs"][f"{arm}/{h}/{policy}"]):
                    q = e["query_index"]
                    expected = tuple(map(int,t["ranking"] if q in selected else e["original_top100"]))
                    assert tuple(gated["ranking"]) == expected
                    assert gated["accepted"] == (q in selected)
                    assert gated["chosen"] == (t["chosen"] if q in selected else -1)
                    for k in (*heads.COSTS,"verified","mu","harm","priority"):
                        assert gated[k] == t[k], k
                    action_rows += 1
    for e,score,req in zip(items,scores,requested):
        assert set(map(int,score["doc_ids"])) == req
        assert len(score["logits"]) == len(req) and score["cls_projection"].shape == (len(req),32)
        assert np.isfinite(score["logits"]).all() and np.isfinite(score["cls_projection"]).all()
        physical_tokens += sum(4+e["ce_query_length"]+e["ce_code_lengths"][d] for d in req)
    ce = read_json(FRESH/"CE_COMPLETE.json")
    assert ce["physical_unique_pairs"] == sum(map(len,requested))
    assert ce["physical_tokens"] == physical_tokens
    return dict(
        queries=len(ids), repositories=len(repos), candidates=candidate_total,
        independently_replayed_action_rows=action_rows,
        independently_recomputed_probability_rows=probability_rows,
        maximum_probability_error=maximum_probability_error,
        vector_ranking_checks=vector_ranking_checks, ce_physical_pairs=sum(map(len,requested)),
        ce_physical_tokens=physical_tokens, risk_refits=6,
        original_query_repository_overlap=0, risk_reselection=False, costs_preserved=True,
    )


def preopening():
    assert not (OUT/"OUTCOME_OPENING.json").exists()
    p = check_protocol()
    # Hash-only reads of development/source provenance; no fresh target decode.
    check_hashes(p["preparation_input_hashes"])
    # Test repository names only; never evaluate an old test outcome.
    from experiments.rene_strict_confirmation_20261019.fresh_data import repository
    old_test_repos = set()
    import json
    for line in (ROOT/"data/codexglue/advtest/test.jsonl").open(encoding="utf8"):
        row = json.loads(line)
        old_test_repos.add(str(row.get("repo") or repository(row["url"])).casefold())
    assert not old_test_repos & {q["repo"] for q in read_json(FRESH/"queries.json")}
    deny_outcome_reads()
    counts = inspect_inference()
    report = dict(
        passed=True,audited_at_utc=utc(),fresh_outcomes_opened=False,
        protocol_sha256=sha(FRESH/"PROTOCOL.json"),
        decision_seal_sha256=sha(FRESH/"DECISION_SEAL.json"),
        metadata_only_old_test_repository_exclusion=True,
        outcome_mapping_not_decoded=True, **counts,
    )
    write_json(OUT/"PREOUTCOME_AUDIT.json",report,immutable=True)
    event("preopening_audit_passed",queries=counts["queries"])
    print("Pre-opening independent audit PASS: " + str(counts),flush=True)


def final():
    p = check_protocol()
    check_hashes(p["preparation_input_hashes"])
    check_hashes(read_json(OUT/"EVALUATION_SEAL.json")["outputs"])
    counts = inspect_inference()
    result = read_json(OUT/"RESULTS.json")
    saved = load(FRESH/"label_free_decisions.pkl")
    items = load(FRESH/"observable_items.pkl")
    private = read_json(FRESH/"private_outcomes.json")
    assert sha(FRESH/"private_outcomes.json") == p["private_outcomes_sha256"]
    assert result["decision_seal_sha256"] == sha(FRESH/"DECISION_SEAL.json")
    seal_time = read_json(FRESH/"DECISION_SEAL.json")["sealed_at_utc"]
    audit_time = read_json(OUT/"PREOUTCOME_AUDIT.json")["audited_at_utc"]
    opening_time = read_json(OUT/"OUTCOME_OPENING.json")["opened_at_utc"]
    assert p["frozen_at_utc"] <= seal_time <= audit_time <= opening_time
    assert read_json(OUT/"CONSUMED.json")["consumed"]
    targets = {q["query_index"]:q["target_doc_id"] for q in private}
    byid = {e["query_index"]:e for e in items}
    ids = [e["query_index"] for e in items]
    corpus = load(ROOT/p["corpus"][0]["path"])
    source = load(PRIOR/"fresh/eligible_repositories.pkl")
    sourceby = {e["source_index"]:e for e in source}
    from experiments.rene_strict_confirmation_20261019.fresh_data import code_key,query_key
    for e,q in zip(items,private):
        src = sourceby[q["source_index"]]
        assert q["query_index"] == e["query_index"]
        assert q["repo"] == src["repo"] and e["query"] == src["query"]
        assert q["query_hash"] == query_key(e["query"]) == src["query_hash"]
        assert q["code_hash"] == code_key(corpus[q["target_doc_id"]]["code_text"]) == src["code_hash"]
    calculated = {}
    maximum_metric_error = 0.
    candidate_label_checks = 0
    for name,ts in saved["outputs"].items():
        delta = np.array([
            independent_rr(t["ranking"],targets[t["query_index"]])-
            independent_rr(byid[t["query_index"]]["original_top100"],targets[t["query_index"]]) for t in ts])
        calculated[name] = delta
        m = result["methods"][name]
        counts_expected = (sum(t["accepted"] for t in ts),int((delta>TOL).sum()),int((delta < -TOL).sum()))
        assert (m["adopted"],m["positive"],m["harmful"]) == counts_expected
        assert m["adopted_zero"] == counts_expected[0]-counts_expected[1]-counts_expected[2]
        error = abs(m["delta_mrr"]-float(delta.mean()))
        maximum_metric_error = max(maximum_metric_error,error)
        assert error < 1e-12
        for k in heads.COSTS:
            assert m[k] == sum(int(t[k]) for t in ts)
    draws = np.random.default_rng(SEED).integers(len(ids)//8,size=(10000,len(ids)//8),dtype=np.int32)
    maximum_interval_error = 0.
    for before in ("support","common"):
        x = calculated["no_seed_uq/full/threshold075"]-calculated[f"no_seed_uq/{before}/threshold075"]
        blocks = np.array([sum(x[i:i+8])/8 for i in range(0,len(x),8)])
        bs = blocks[draws].mean(1)
        quant = np.quantile(bs,[.0125,.9875])
        half = stats.t.ppf(.9875,len(blocks)-1)*np.std(blocks,ddof=1)/np.sqrt(len(blocks))
        env = [min(quant[0],x.mean()-half),max(quant[1],x.mean()+half)]
        recorded = result["primary_contrasts"][f"full_minus_{before}"]
        error = float(np.max(abs(np.array(env)-recorded["simultaneous_envelope"])))
        maximum_interval_error = max(maximum_interval_error,error)
        assert error < 1e-12
        bound = saved["geometry"]["no_seed_uq"][f"full_minus_{before}"]["threshold075"]
        assert np.all(x >= bound["per_query_lower"]-1e-12) and np.all(x <= bound["per_query_upper"]+1e-12)
    for e in items:
        target = targets[e["query_index"]]
        base = independent_rr(e["original_top100"],target)
        for top in e["candidate_top100"]:
            value = independent_rr(rrf(e["original_top100"],top),target)-base
            assert -1 <= value <= 1
            candidate_label_checks += 1
    for arm in ARMS:
        arr = saved["inputs"][arm]
        pre = calculated[f"{arm}/pre_gate"]
        ix = np.array([ids.index(q) for q in arr["q"]],int)
        labels = np.where(pre[ix] < -TOL,0,np.where(pre[ix]>TOL,2,1))
        y = (labels == 0).astype(float)
        for h in heads.HEADS:
            prob = saved["probabilities"][arm][h]["classes"]
            hp = np.minimum(1-1e-12,np.maximum(1e-12,prob[:,0]))
            expected = dict(
                harm_brier=float(np.mean((prob[:,0]-y)**2)),
                harm_nll=float(np.mean(-y*np.log(hp)-(1-y)*np.log1p(-hp))),
                class_nll=float(np.mean([-np.log(max(row[label],1e-12)) for row,label in zip(prob,labels)])),
                class_brier=float(np.mean(np.sum((prob-np.eye(3)[labels])**2,axis=1))),
            )
            for k,value in expected.items():
                assert abs(value-result["proper_scores"][arm]["methods"][h][k]) < 1e-12
        training = load(OUT/f"training/{arm}.pkl")
        bundle = load(OUT/f"models/{arm}.pkl")
        assert sha(OUT/f"training/{arm}.pkl") == bundle["training_provenance_sha256"]
        for model in bundle["models"].values():
            assert model["fit_labels_sha256"] == hashlib.sha256(training["training"]["delta"].tobytes()).hexdigest()
    report = dict(
        passed=True,audited_at_utc=utc(),protocol_sha256=sha(FRESH/"PROTOCOL.json"),
        results_sha256=sha(OUT/"RESULTS.json"), candidate_label_checks=candidate_label_checks,
        maximum_metric_error=maximum_metric_error,maximum_interval_error=maximum_interval_error,
        target_mapping_independently_verified=True,protected_unchanged=True,
        risk_certified=False,old_test_metrics_used=False,**counts,
    )
    write_json(OUT/"AUDIT.json",report,immutable=True)
    event("final_independent_audit_passed",audit_sha256=sha(OUT/"AUDIT.json"))
    print("Final independent audit PASS: " + str(report),flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase",choices=("pre","final"),required=True)
    args = ap.parse_args()
    require_runtime()
    with threadpool_limits(limits=2):
        {"pre":preopening,"final":final}[args.phase]()
