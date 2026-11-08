"""Observable-only inference. Preparation and label opening run separately."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.rene_uq_independent_proof_20261019.study import (
    ARMS, DATA, FRESH, OUT, POLICIES, PRIOR, backend, check_protocol,
    deny_outcome_reads, event, heads, prepare,
)
from experiments.rene_strict_confirmation_20261019 import model as upstream
from experiments.rene_strict_confirmation_20261019.common import (
    check_hashes, load, observable, read_json, reject_gold, require_runtime,
    save, sha, utc, write_json,
)
from experiments.rene_strict_confirmation_20261019.development import compact


def plan():
    check_protocol()
    check_hashes(read_json(FRESH/"EVIDENCE_COMPLETE.json")["outputs"])
    destination = FRESH/"sessions.pkl"
    if destination.exists():
        assert sha(destination) == read_json(FRESH/"PLAN_SEAL.json")["sessions_sha256"]
        return load(destination)
    items = [observable(e) for e in load(FRESH/"observable_items.pkl")]
    sessions = {}
    for arm in ARMS:
        model = joblib.load(PRIOR/f"models/fusion_{arm}.joblib")
        predictions = upstream.predict_gain(model["gain"], items)
        sessions[arm] = [upstream.prepare(e,p,"fusion",model["cut"]) for e,p in zip(items,predictions)]
    reject_gold(sessions)
    save(destination,sessions)
    write_json(FRESH/"PLAN_SEAL.json",dict(
        sealed_at_utc=utc(), protocol_sha256=sha(FRESH/"PROTOCOL.json"),
        observations_sha256=sha(FRESH/"observable_items.pkl"), sessions_sha256=sha(destination),
        actual_plans=True, outcomes_opened=False,
    ),immutable=True)
    event("observable_ce_plans_sealed")
    print("CE plans sealed without labels",flush=True)
    return sessions


def score():
    path = FRESH/"ce_scores.pkl"
    if path.exists():
        assert sha(path) == read_json(FRESH/"CE_COMPLETE.json")["scores_sha256"]
        return
    import torch
    from transformers import AutoTokenizer
    from experiments.action_witness_uq_20261014.gpu_data import CE, ce_tokens
    sessions = plan()
    p = check_protocol()
    queries = read_json(FRESH/"queries.json")
    queryby = {q["query_index"]:q for q in queries}
    corpus = load(ROOT/p["corpus"][0]["path"])
    requested = {q["query_index"]:set() for q in queries}
    for values in sessions.values():
        for q,s in zip(queries,values):
            if s["verified"]:
                requested[q["query_index"]].update(map(int,s["plan"]["doc_ids"]))
    pairs = [(q["query_index"],d) for q in queries for d in sorted(requested[q["query_index"]])]
    tokenizer = AutoTokenizer.from_pretrained(ROOT/"models/codebert-base",local_files_only=True)
    model = CE().to("cuda").eval()
    model.load_state_dict(torch.load(DATA/"ce_large_dense_4096.pt",map_location="cpu",weights_only=True))
    projection = torch.tensor(np.load(DATA/"cls_projection.npy"),device="cuda")
    logits, projected, tokens_executed = [], [], 0
    began = time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(pairs),32):
            current = pairs[i:i+32]
            tokens = ce_tokens(tokenizer,[queryby[q]["query"] for q,d in current],
                               [corpus[d]["code_text"] for q,d in current])
            tokens_executed += int(tokens["attention_mask"].sum())
            with torch.autocast("cuda",dtype=torch.bfloat16):
                value,cls = model(return_cls=True,**tokens)
            logits.extend(value.float().cpu().tolist())
            projected.extend((cls.float()@projection).cpu().numpy())
    lookup = {pair:i for i,pair in enumerate(pairs)}
    output = []
    for q in queries:
        docs = sorted(requested[q["query_index"]])
        ix = [lookup[q["query_index"],d] for d in docs]
        output.append(dict(query_index=q["query_index"],doc_ids=np.array(docs,int),
                           logits=np.array([logits[i] for i in ix]),
                           cls_projection=np.array([projected[i] for i in ix]).reshape(-1,32)))
    reject_gold(output)
    save(path,output)
    write_json(FRESH/"CE_COMPLETE.json",dict(
        completed=True, physical_unique_pairs=len(pairs), physical_tokens=tokens_executed,
        elapsed_seconds=time.perf_counter()-began, scores_sha256=sha(path),
        plan_seal_sha256=sha(FRESH/"PLAN_SEAL.json"), outcomes_opened=False,
    ),immutable=True)
    event("ce_scoring_completed",physical_unique_pairs=len(pairs))
    print(f"CE completed: {len(pairs)} actual shared pairs",flush=True)
    del model
    torch.cuda.empty_cache()


def bounds_for_trace(original,action,sign):
    """Exact possible RR-difference range over every single target identity."""
    if sign == 0:
        return 0.,0.
    old = {int(d):1/(i+1) for i,d in enumerate(original[:100])}
    new = {int(d):1/(i+1) for i,d in enumerate(action[:100])}
    shifts = [0.] + [sign*(new.get(d,0.)-old.get(d,0.)) for d in old.keys()|new.keys()]
    return float(min(shifts)),float(max(shifts))


def decide():
    path = FRESH/"label_free_decisions.pkl"
    if path.exists():
        assert sha(path) == read_json(FRESH/"DECISION_SEAL.json")["decisions_sha256"]
        return
    p = check_protocol()
    sessions = plan()
    scores = load(FRESH/"ce_scores.pkl")
    assert sha(FRESH/"ce_scores.pkl") == read_json(FRESH/"CE_COMPLETE.json")["scores_sha256"]
    items = [observable(e) for e in load(FRESH/"observable_items.pkl")]
    outputs, fixed, arrays, probabilities, masks, geometry = {},{},{},{},{},{}
    for arm in ARMS:
        pipeline = joblib.load(PRIOR/f"models/fusion_{arm}.joblib")
        ts = [upstream.choose_trace(e,s,pipeline["update"],score)
              for e,s,score in zip(items,sessions[arm],scores)]
        fixed[arm] = ts
        arrays[arm] = heads.inputs(items,ts)
        fitted = load(OUT/f"models/{arm}.pkl")["models"]
        probabilities[arm] = {h:heads.predict(fitted[h],arrays[arm]) for h in heads.HEADS}
        old = upstream.risk.probabilities(pipeline["risk"],dict(
            x=arrays[arm]["common"],native=arrays[arm]["native"]))[1]
        np.testing.assert_allclose(old,probabilities[arm]["common"]["classes"],atol=1e-12,rtol=0)
        outputs[f"{arm}/pre_gate"] = [compact(t) for t in ts]
        for stage,tau in (("mean",1.),("base",.1)):
            outputs[f"{arm}/{stage}"] = [
                compact(upstream.choose_trace(e,s,tau=tau,use_ce=False))
                for e,s in zip(items,sessions[arm])]
        masks[arm],geometry[arm] = {},{}
        for h in heads.HEADS:
            masks[arm][h] = {}
            for policy in POLICIES:
                mask = heads.adoption(arrays[arm]["q"],probabilities[arm][h]["harm"],policy)
                masks[arm][h][policy] = mask
                selected = {q for q,use in zip(arrays[arm]["q"],mask) if use}
                pr = dict(zip(arrays[arm]["q"],probabilities[arm][h]["harm"]))
                traces = []
                for e,t in zip(items,ts):
                    out = compact(t)
                    out["post_harm"] = None if t["query_index"] not in pr else float(pr[t["query_index"]])
                    out["post_risk_rejected"] = bool(t["accepted"] and t["query_index"] not in selected)
                    if out["post_risk_rejected"]:
                        out.update(accepted=False,chosen=-1,ranking=tuple(map(int,e["original_top100"])))
                    traces.append(out)
                outputs[f"{arm}/{h}/{policy}"] = traces
        for before in ("support","common"):
            geometry[arm][f"full_minus_{before}"] = {}
            for policy in ("threshold075","fixed50"):
                aq = dict(zip(arrays[arm]["q"],masks[arm][before][policy]))
                bq = dict(zip(arrays[arm]["q"],masks[arm]["full"][policy]))
                lower,upper = [],[]
                for e,t in zip(items,ts):
                    q = t["query_index"]
                    lo,hi = bounds_for_trace(e["original_top100"],t["ranking"],
                                            int(bq.get(q,False))-int(aq.get(q,False)))
                    lower.append(lo); upper.append(hi)
                geometry[arm][f"full_minus_{before}"][policy] = dict(
                    per_query_lower=np.array(lower),per_query_upper=np.array(upper),
                    mean_lower=float(np.mean(lower)),mean_upper=float(np.mean(upper)),
                    requires_labels=False, scope="this fixed action cohort only",
                )
    result = dict(outputs=outputs,fixed=fixed,inputs=arrays,probabilities=probabilities,
                  masks=masks,geometry=geometry,protocol_sha256=sha(FRESH/"PROTOCOL.json"))
    reject_gold(result)
    save(path,result)
    write_json(FRESH/"DECISION_SEAL.json",dict(
        sealed_at_utc=utc(),actual_decisions=True,outcomes_opened=False,
        protocol_sha256=sha(FRESH/"PROTOCOL.json"),decisions_sha256=sha(path),
        observations_sha256=sha(FRESH/"observable_items.pkl"),
        scores_sha256=sha(FRESH/"ce_scores.pkl"),plan_seal_sha256=sha(FRESH/"PLAN_SEAL.json"),
        outputs=len(outputs),accepted={k:sum(t["accepted"] for t in v) for k,v in outputs.items()},
    ),immutable=True)
    event("all_actual_decisions_sealed",queries=p["query_count"],outputs=len(outputs))
    print("All actual rankings and three-head decisions sealed; no outcome access",flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("prepare","generate","retrieve","relevance","plan","score","decide","all"),required=True)
    args = ap.parse_args()
    with threadpool_limits(limits=2):
        if args.stage == "prepare":
            prepare()
            return
        deny_outcome_reads()
        require_runtime(gpu=args.stage in ("generate","retrieve","relevance","score","all"))
        engine = backend()
        work = dict(generate=engine.generate,retrieve=engine.retrieval,relevance=engine.relevance,
                    plan=plan,score=score,decide=decide)
        if args.stage == "all":
            began = time.perf_counter()
            for stage in ("generate","retrieve","relevance","plan","score","decide"):
                event("inference_stage_started",stage_name=stage)
                work[stage]()
                event("inference_stage_completed",stage_name=stage)
            write_json(OUT/"INFERENCE_COMPLETE.json",dict(
                completed=True,elapsed_seconds=time.perf_counter()-began,
                actual_decisions_sha256=sha(FRESH/"label_free_decisions.pkl"),
                outcomes_opened=False),immutable=True)
        else:
            work[args.stage]()


if __name__ == "__main__":
    main()
