"""Independent Fraction sorting, exact relevance, cost and scalar CI verification.

Does not import any production data, ranking, dedup, metric or evaluation code.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/rene_multicandidate_fusion_20261020"
PARENT=ROOT/"results/rene_cosqa_development_20261019"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value:
        raise AssertionError(message)


def independent_merge(rows):
    unique={}
    for row in rows:
        if row["text"] in unique:
            require(unique[row["text"]]["ranked_doc_ids"]==row["ranked_doc_ids"],"Same text rankings differ")
            require(unique[row["text"]]["scores"]==row["scores"],"Same text scores differ")
        else:
            unique[row["text"]]=row
    scores=defaultdict(Fraction)
    for row in unique.values():
        require(len(row["ranked_doc_ids"])==len(set(row["ranked_doc_ids"]))==100,"Invalid input Top100")
        for rank,doc in enumerate(row["ranked_doc_ids"],1):
            scores[doc]+=Fraction(1,60+rank)
    order=sorted(scores,key=lambda doc:(-scores[doc],doc))[:100]
    counts=Counter(scores.values())
    return order,{"unique_lists":len(unique),"union_documents":len(scores),
                  "score_tie_groups":sum(n>1 for n in counts.values()),
                  "documents_in_score_ties":sum(n for n in counts.values() if n>1),
                  "cutoff_tie_documents":counts[scores[order[-1]]]}


def exact_rr(order,relevant):
    for i,doc in enumerate(order,1):
        if doc in relevant:
            return Fraction(1,i)
    return Fraction(0)


def interval(vector,groups,family=False):
    unique=sorted(set(groups))
    index={g:i for i,g in enumerate(unique)}
    codes=np.asarray([index[g] for g in groups])
    count=np.bincount(codes)
    total=np.bincount(codes,weights=vector)
    mean=math.fsum(map(float,vector))/len(vector)
    se=math.sqrt(len(unique)/(len(unique)-1)*sum((total-count*mean)**2)/len(vector)**2)
    rng=np.random.default_rng(20261020)
    sample=np.empty(10000)
    for start in range(0,10000,100):
        draw=rng.integers(0,len(unique),size=(100,len(unique)))
        sample[start:start+100]=np.sum(total[draw],axis=1)/np.sum(count[draw],axis=1)
    c=t.ppf(.975,len(unique)-1)
    result={"mean":mean,"standard_error_cluster":se,"ci95_cluster_t":[mean-c*se,mean+c*se],
            "ci95_group_bootstrap":np.quantile(sample,[.025,.975]).tolist(),
            "changed_queries":sum(abs(float(v))>1e-12 for v in vector),
            "groups_with_nonzero_queries":len({g for g,v in zip(groups,vector) if abs(v)>1e-12})}
    if family:
        critical=t.ppf(1-.05/8,len(unique)-1)
        result["four_effect_family95_bonferroni_cluster_t"]=[mean-critical*se,mean+critical*se]
        result["four_effect_family95_bonferroni_group_bootstrap"]=np.quantile(sample,[.05/8,1-.05/8]).tolist()
    return result


def main():
    protocol=read(OUT/"PROTOCOL.json")
    seal=read(OUT/"ACTIONS_SEALED.json")
    result=read(OUT/"RESULTS.json")
    require(protocol["role"]==result["role"]=="consumed_development_post_hoc_extension","Role differs")
    require(not result["new_confirmation"] and not result["risk_certified"],"Unsupported status")
    require(seal["protocol_sha256"]==sha(OUT/"PROTOCOL.json"),"Action protocol seal drift")
    hashes=protocol["bound_files"]+protocol["protected_files"]+seal["files"]
    for item in hashes:
        require(sha(ROOT/item["path"])==item["sha256"],f'Frozen file changed: {item["path"]}')
    access=read(OUT/"OUTCOME_ACCESS.json")
    require(seal["created_utc"]<=access["created_utc"]<=result["created_utc"],"Action seal/outcome order differs")
    require(not seal["qrels_read_by_action_construction"] and not access["H_effects_read"],"Invalid label access")
    qids=read(PARENT/"PROTOCOL.json")["query_ids"]
    groups=read(PARENT/"data/groups.json")
    roles=read(PARENT/"ROLE_MANIFEST.json")
    require(set(qids)==set(roles["development_query_ids"])==set(groups),"D role mismatch")
    require(not set(qids)&set(roles["confirmation_pool_query_ids"]),"H query leakage")
    require(not set(groups.values())&set(roles["confirmation_pool_group_ids"]),"H group leakage")
    queries={r["query_id"]:r["query"] for r in lines(PARENT/"data/queries.jsonl")}
    candidates=defaultdict(list)
    for row in lines(PARENT/"candidates.jsonl"):
        candidates[row["query_id"]].append(row)
    selections={r["query_id"]:[v["position"] for v in sorted(r["selected"],key=lambda v:v["selected_rank"])]
                for r in lines(PARENT/"selections.jsonl") if r["policy"]=="ENTR"}
    qrels=defaultdict(set)
    for r in lines(PARENT/"data/private/development_qrels.jsonl"):
        require(r["query_id"] in groups,"Qrels outside D")
        if r.get("relevance",1)>0:
            qrels[r["query_id"]].add(r["doc_id"])
    evaluated={(r["retriever"],r["query_id"]):r for r in lines(OUT/"per_query.jsonl")}
    require(len(qids)==1490 and len(set(groups.values()))==512 and len(evaluated)==2980,"Denominator mismatch")
    counts=Counter();max_errors=defaultdict(float);cross={}

    def near(a,b,name):
        error=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
        counts[name]+=1;max_errors[name]=max(max_errors[name],error)
        require(error<2e-12,f"{name} differs {error}")

    for retriever in ("CodeBERT","BM25"):
        source={(r["query_id"],r["position"]):r for r in lines(PARENT/f"rankings/{retriever}.jsonl")}
        rows=lines(OUT/f"actions/{retriever}.jsonl")
        require([r["query_id"] for r in rows]==qids,"Query order mismatch")
        values=defaultdict(list);changed=defaultdict(list);costs=defaultdict(list)
        for action in rows:
            q=action["query_id"]
            pool=sorted(candidates[q],key=lambda r:(r["full_vocab_entropy"],r["position"]))[:3]
            positions=[r["position"] for r in pool]
            require(positions==selections[q]==action["selected_positions"],"Candidate identity mismatch")
            require(action["group"]==groups[q] and action["generation_positions"]==len(candidates[q])==len(queries[q].split()),"Metadata mismatch")
            original=source[q,None]
            rewrite=[source[q,p] for p in positions]
            for row,candidate in zip(rewrite,pool):
                require(row["text"]==candidate["rewrite"],"Candidate ranking text mismatch")
            r3,m3=independent_merge(rewrite)
            r4,m4=independent_merge([original]+rewrite)
            expected={"NoRewrite":original["ranked_doc_ids"],"ENTRTop1":rewrite[0]["ranked_doc_ids"],"RRF3":r3,"RRF4":r4}
            require(expected==action["rankings"],"Independent exact Fraction full Top100 fusion mismatch")
            require(m3==action["fusion_metadata"]["RRF3"] and m4==action["fusion_metadata"]["RRF4"],"Dedup/union/tie metadata mismatch")
            require(action["common_original_plus_ENTR3_inputs"]==m4["unique_lists"],"Common budget differs")
            require(action["original_plus_top1_inputs"]==len({original["text"],rewrite[0]["text"]}),"Original+Top1 budget differs")
            counts["independent_fused_full_top100"]+=2
            rrvals={p:float(exact_rr(order,qrels[q])) for p,order in expected.items()}
            rrvals["ENTRBest3"]=float(max(exact_rr(r["ranked_doc_ids"],qrels[q]) for r in rewrite))
            per=evaluated[retriever,q]
            for p,v in rrvals.items():
                near(v,per["RR"][p],"exact_rr_values")
                values[p].append(v)
                if p in expected:
                    is_changed=expected[p]!=original["ranked_doc_ids"]
                    require(is_changed==per["ranking_changed"][p],"Changed ranking mismatch")
                    changed[p].append(is_changed)
                    cost={"NoRewrite":1,"ENTRTop1":1,"RRF3":m3["unique_lists"],"RRF4":m4["unique_lists"]}[p]
                    require(cost==action["cost_unique_retrieval_inputs"][p]==per["cost_unique_retrieval_inputs"][p],"Cost mismatch")
                    costs[p].append(cost)
            for name,effect in per["effects"].items():
                a,b=name.split("-")
                near(rrvals[a]-rrvals[b],effect,"per_query_effects")
        values={p:np.asarray(v) for p,v in values.items()}
        for p,v in values.items():
            u=result["views"][retriever]["policies"][p]
            delta=v-values["NoRewrite"]
            near(v.mean(),u["mrr"],"mrr")
            near(delta.mean(),u["delta_mrr"],"delta_mrr")
            require(sum(delta>0)==u["beneficial_queries"] and sum(delta<0)==u["harmful_queries"] and sum(delta==0)==u["zero_gain_queries"],"Harm/benefit counts mismatch")
            near(np.maximum(delta,0).mean(),u["benefit_mass_per_query"],"benefit_mass")
            near(np.maximum(-delta,0).mean(),u["harm_mass_per_query"],"harm_mass")
            if (delta<0).any():
                near(-delta[delta<0].mean(),u["mean_loss_among_harms"],"harm_magnitude")
                near(-delta.min(),u["max_loss"],"max_loss")
            if p in costs:
                c=result["views"][retriever]["costs"][p]
                require(sum(costs[p])==c["standalone_unique_retrieval_inputs_total"],"Total cost mismatch")
                near(np.mean(costs[p]),c["standalone_unique_retrieval_inputs_mean"],"mean_retrieval_inputs")
                require(sum(changed[p])==u["ranking_changed_queries"],"Changed count mismatch")
        require(sum(costs["RRF4"])==result["views"][retriever]["common_four_list_available_cache_budget_total"],"Total common budget differs")
        group_list=[groups[q] for q in qids]
        effects={}
        for name,e in result["views"][retriever]["estimates"].items():
            a,b=name.split("-");vector=values[a]-values[b];effects[name]=vector
            independent=interval(vector,group_list,family=name in ("RRF3-NoRewrite","RRF4-NoRewrite"))
            for key,val in independent.items():
                near(val,e[key],"independent_interval_fields")
            counts["scalar_contrasts"]+=1
        cross[retriever]=effects
        print({"independent_fusion_audit":retriever,"queries":len(rows),"passed":True},flush=True)
    for name,e in result["cross_retriever_CodeBERT_minus_BM25"].items():
        vector=cross["CodeBERT"][name]-cross["BM25"][name]
        for key,val in interval(vector,[groups[q] for q in qids]).items():
            near(val,e[key],"independent_cross_retriever_fields")
        counts["scalar_contrasts"]+=1
    for item in hashes:
        require(sha(ROOT/item["path"])==item["sha256"],"Final hash drift")
    report={"passed":True,"created_utc":datetime.now(timezone.utc).isoformat(),"checks":dict(counts),
            "max_errors":dict(max_errors),"frozen_input_source_protected_files":len(protocol["bound_files"])+len(protocol["protected_files"]),
            "sealed_action_files":len(seal["files"]),"full_top100_fraction_sorting_independent":True,
            "H_effects_read":False,"official_state_changed":False}
    with (OUT/"AUDIT.json").open("x",encoding="utf-8") as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")
    tests=read(OUT/"TEST_RESULTS.json")
    require(tests["passed"],"Contract tests failed")
    acceptance={"passed":True,"created_utc":datetime.now(timezone.utc).isoformat(),
                "tests":tests,"protocol_sha256":sha(OUT/"PROTOCOL.json"),"results_sha256":sha(OUT/"RESULTS.json"),
                "audit_sha256":sha(OUT/"AUDIT.json"),"role":protocol["role"],"risk_certified":False,
                "new_confirmation":False,"official_state_changed":False}
    with (OUT/"FINAL_ACCEPTANCE.json").open("x",encoding="utf-8") as handle:
        json.dump(acceptance,handle,ensure_ascii=False,indent=2);handle.write("\n")
    print({"fusion_final_acceptance":"PASS","independent_fraction_rankings":counts["independent_fused_full_top100"],"exact_rr":counts["exact_rr_values"]},flush=True)


if __name__=="__main__":
    main()
