from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

from experiments.rene_ssqr_bridge_20261019.common import (
    ROOT, OUT as BRIDGE, DEVELOPMENT, PROFILES, read, readl, write, writel, record, sha,
    now, require, text_id, pools_for, verify_seal,
)
from experiments.rene_ssqr_bridge_20261019.evaluation import cluster_interval

OUT=ROOT/"results/rene_ssqr_aggregation_audit_20261019"
AUTHOR_BLOB="a75b4944ca9d656641ed6214ce14c2709dca946e"


def freeze():
    if (OUT/"PROTOCOL.json").exists():return read(OUT/"PROTOCOL.json")
    require(not (BRIDGE/"metrics/RESULTS.json").exists(),"aggregation supplement must freeze before new bridge effects")
    bridge=read(BRIDGE/"PROTOCOL.json")
    api=ROOT/f"results/rene_cosqa_ssqr_audit_20261019/sources/ssqr_blob_{AUTHOR_BLOB}.json"
    source=base64.b64decode(read(api)["content"])
    require(hashlib.sha1(f"blob {len(source)}\0".encode()+source).hexdigest()==AUTHOR_BLOB,"author evaluator Git blob identity mismatch")
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"author_evaluator.py").write_bytes(source)
    queries={"source":readl(BRIDGE/"data/source/queries.jsonl"),"cosqa":readl(DEVELOPMENT/"data/queries.jsonl")}
    eligible={}
    for domain,rows in queries.items():
        valid={r["query_id"] for r in rows if len(r["query"].split())>=3}
        eligible[domain]={scope:[qid for qid in bridge["samples"][domain][key] if qid in valid]
                          for scope,key in (("main","main_ids"),("sensitivity","sensitivity_ids"))}
    protocol={"status":"FROZEN","created_utc":now(),"role":"post_confirmation_aggregation_diagnostic",
              "prior_H_unchanged":True,"CoSQA_H_used":False,"new_bridge_effects_already_open":False,
              "purpose":"disambiguate per-query best3, best aggregate of three rank-wise MRRs, and public URL-last-row evaluator behavior",
              "eligible_ids":eligible,"eligibility":"query has >=3 whitespace words in every profile; same complete-three subset across profiles; no zero/identity padding",
              "profiles":list(PROFILES),"mask":"full","selections":"same sealed ENTR/PROB/RAND20 positions as bridge",
              "aggregations":["per_query_best3","best_of_three_rankwise_MRRs","last_selected_row"],
              "uncertainty":"per-query best3 and last row use connected-group intervals; best rank-wise aggregate uses simultaneous Bonferroni cluster-t bounds on all 66 policy-rank means; descriptive, no selection promotion",
              "unavailable":"public evaluator does not establish which unpublished aggregation pipeline produced original paper tables",
              "author_blob":AUTHOR_BLOB,"bound_files":[record(api),record(OUT/"author_evaluator.py"),record(Path(__file__)),record(BRIDGE/"PROTOCOL.json")],
              "new_model_training":False,"new_generation":False}
    write(OUT/"PROTOCOL.json",protocol)
    print({"aggregation_frozen":True,"eligible_queries":{d:{s:len(v) for s,v in x.items()} for d,x in eligible.items()}},flush=True)
    return protocol


def author_evaluator_check():
    spec=importlib.util.spec_from_file_location("ssqr_original_evaluator",OUT/"author_evaluator.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory=Path(tempfile.mkdtemp(prefix="rene-ssqr-author-eval-toy-"))
    answers=directory/"answers.jsonl"
    predictions=directory/"predictions.jsonl"
    writel(answers,[{"url":"synthetic-query","idx":7}])
    writel(predictions,[{"url":"synthetic-query","answers":[7,8,9]},
                        {"url":"synthetic-query","answers":[8,9,7]},
                        {"url":"synthetic-query","answers":[9,8,6]}])
    actual=module.calculate_scores(module.read_answers(str(answers)),module.read_predictions(str(predictions)))
    require(actual["MRR"]==0.,"unexpected author evaluator duplicate-URL semantics")
    result={"passed":True,"author_Git_blob":AUTHOR_BLOB,"toy_directory":str(directory),
            "same_query_prediction_rows":3,"public_evaluator_MRR":actual["MRR"],
            "per_query_best3_MRR":1.,"best_rankwise_MRR":1.,
            "conclusion":"public read_predictions overwrites earlier rows for duplicate URLs; not evidence of actual paper production evaluator"}
    write(OUT/"AUTHOR_EVALUATOR_CHECK.json",result)
    return result


def simultaneous_max_mean_bounds(values,groups):
    labels=sorted(set(groups))
    index={g:i for i,g in enumerate(labels)}
    group_sums=np.zeros((len(labels),values.shape[1],values.shape[2]),dtype=np.float64)
    sizes=np.zeros(len(labels),dtype=np.float64)
    for row,group in zip(values,groups):
        group_sums[index[group]]+=row
        sizes[index[group]]+=1
    means=values.mean(0)
    influence=group_sums-sizes[:,None,None]*means[None,:,:]
    se=np.sqrt(len(labels)/(len(labels)-1)*np.square(influence).sum(0)/len(values)**2)
    components=int(means.size)
    critical=float(t.isf(.05/(2*components),len(labels)-1))
    lower,upper=means-critical*se,means+critical*se
    policy_point=means.max(1)
    policy_low,policy_high=lower.max(1),upper.max(1)
    return {"means":means.tolist(),"components":components,"critical":critical,
            "ENTR":float(policy_point[0]),"PROB":float(policy_point[1]),"RAND20":float(policy_point[2:].mean()),
            "ENTR_minus_RAND":float(policy_point[0]-policy_point[2:].mean()),
            "ENTR_minus_PROB":float(policy_point[0]-policy_point[1]),
            "ENTR_minus_RAND_conservative95":[float(policy_low[0]-policy_high[2:].mean()),float(policy_high[0]-policy_low[2:].mean())],
            "ENTR_minus_PROB_conservative95":[float(policy_low[0]-policy_high[1]),float(policy_high[0]-policy_low[1])],
            "coverage_assumption":"simultaneous cluster-t approximation; no smooth-argmax bootstrap or independent-group risk guarantee"}


def evaluate():
    protocol=read(OUT/"PROTOCOL.json")
    for row in protocol["bound_files"]:require(sha(ROOT/row["path"])==row["sha256"],"aggregation frozen input changed")
    verify_seal("CANDIDATES_SEALED.json")
    verify_seal("RANKINGS_SEALED.json")
    require(read(BRIDGE/"PRE_EFFECT_AUDIT.json")["passed"],"bridge pre-effect audit missing")
    write(OUT/"OPENING.json",{"created_utc":now(),"protocol_sha256":sha(OUT/"PROTOCOL.json"),"role":protocol["role"]})
    toy=author_evaluator_check()
    selections={(r["domain"],r["scope"],r["profile"],r["mask"],r["query_id"]):r for r in readl(BRIDGE/"selections.jsonl")}
    joint=read(BRIDGE/"data/joint_groups.json")
    results=[]
    for domain in ("source","cosqa"):
        qrel_path=BRIDGE/"data/private/source_qrels.jsonl" if domain=="source" else DEVELOPMENT/"data/private/development_qrels.jsonl"
        target={r["query_id"]:r["doc_id"] for r in readl(qrel_path)}
        for retriever in ("CodeBERT","BM25"):
            ranks={r["text_id"]:r["ranked_doc_ids"] for r in readl(BRIDGE/f"rankings/{domain}/{retriever}.jsonl")}
            for scope,profiles in (("main",["greedy20_source"]),("sensitivity",list(PROFILES))):
                ids=protocol["eligible_ids"][domain][scope]
                groups=[joint[domain][qid] for qid in ids]
                for profile in profiles:
                    pools=pools_for(domain,profile,sensitivity=scope=="sensitivity")
                    tensor=np.empty((len(ids),22,3),dtype=np.float64)
                    for i,qid in enumerate(ids):
                        chosen=selections[(domain,scope,profile,"full",qid)]
                        by_position={r["position"]:r["rewrite"] for r in pools[qid]}
                        selections_= [chosen["ENTR"],chosen["PROB"]]+chosen["RAND"]
                        for policy,positions in enumerate(selections_):
                            require(len(positions)==3,"aggregation diagnostic must have exactly 3 real positions")
                            for j,p in enumerate(positions):
                                ordered=ranks[text_id(by_position[p])]
                                tensor[i,policy,j]=1./(ordered.index(target[qid])+1) if target[qid] in ordered else 0.
                    modes={}
                    for mode,policy_rr in (("per_query_best3",tensor.max(2)),("last_selected_row",tensor[:,:,2])):
                        er=policy_rr[:,0]-policy_rr[:,2:].mean(1)
                        ep=policy_rr[:,0]-policy_rr[:,1]
                        modes[mode]={"ENTR":float(policy_rr[:,0].mean()),"PROB":float(policy_rr[:,1].mean()),
                                     "RAND20":float(policy_rr[:,2:].mean()),"ENTR_minus_RAND":cluster_interval(er,groups),
                                     "ENTR_minus_PROB":cluster_interval(ep,groups)}
                    modes["best_of_three_rankwise_MRRs"]=simultaneous_max_mean_bounds(tensor,groups)
                    results.append({"domain":domain,"retriever":retriever,"scope":scope,"profile":profile,
                                    "queries":len(ids),"groups":len(set(groups)),"aggregations":modes})
    result={"role":protocol["role"],"views":results,"author_evaluator_toy":toy,
            "primary_H_modified":False,"original_paper_production_aggregation_recovered":False,
            "no_profile_promoted":True}
    write(OUT/"RESULTS.json",result)
    lines=['# SSQR Checking of the Triple Convergence with Open Evaluator',"",
           "Public JSON generation path with multiple rewrites by the same URL; The public evaluator saves the projections as URLs' dict, backline covers the front line. Directly execute the author assessor verified by Git blob, 3 Coding URL Synthetic results 0,PV query best3 1; This proves the behaviour of the interface and does not prove that the wrong path was used in the author ' s production form for the year.","",
           'Papers 5.1 Write first three articles and report highest MRR; The open assessment chain does not explicitly achieve query maximum RR or total of three positions MRR to maximize. The following three calibres are therefore reported as a limited additional diagnosis and do not respond to the main comparison of frozen CoSQA.',"",
           'All data have been consumed for source data/CoSQA D and new bridge effects are frozen before opening. Requires the original query at least 3 words to give all profile three real positions; Query less than 3 does not add zero and no sentence, and the master test full denominator remains as it is. The number of subsets of this supplement is specified in PROTOCOL.',"",
           '| Retrieval Data/Major Bridge Query Best3: E-R | Three runk-wise MRRs Best available: E-R | Public Reader End caliber: E-R |',
           "|---|---|---:|---:|---:|"]
    for row in results:
        if row["scope"]!="main":continue
        modes=row["aggregations"]
        a=modes["per_query_best3"]["ENTR_minus_RAND"]["mean"]
        b=modes["best_of_three_rankwise_MRRs"]["ENTR_minus_RAND"]
        c=modes["last_selected_row"]["ENTR_minus_RAND"]["mean"]
        lines.append(f'| {row["retriever"]} | {row["domain"]}/{row["queries"]} | {a:+.8f} | {b:+.8f} | {c:+.8f} |')
    lines += ["",'The maximum values of the three rank-wise averages are non-sliding statistics, and the best position selected for viewing labels is not used as a fixed policy to calculate normal pair ranges; RESULTS Provide 66 policy x rank Mean values are simultaneously Bonferroni cluster-t Conservative contrast ranges after transmission of the approximation boundary. The remaining two calibres are paired into groups. All for diagnosis, no additional confirmation tests.',"",
              "The paper should be presented as `This document explicitly uses the three candidate indicators of maximum search-by-search RR' and registers that open convergence paths have not been restored. The final act public evaluator cannot be considered as a real production process for the original paper, nor can it be claimed that the result of the original author was overturned by the synthesis.","",
              'Original public evaluator blob: `a75b4944ca9d656641ed6214ce14c2709dca946e`; The current round of source codes, protocols and synthetic evidence of execution are retained. Full six profile and PROB results at RESULTS.json.',""]
    (OUT/"READOUT.md").write_text("\n".join(lines),encoding="utf-8")
    write(OUT/"FINAL_ACCEPTANCE.json",{"passed":True,"views":len(results),"author_evaluator_reexecuted":True,
          "same_cached_selection_and_ranking":True,"no_new_T5_or_model_fit":True,"protocol_sha256":sha(OUT/"PROTOCOL.json"),
          "declared_post_confirmation_diagnostic":True})
    print({"completed":True,"views":len(results),"public_evaluator_toy":toy},flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("freeze","evaluate"))
    args=parser.parse_args()
    freeze() if args.mode=="freeze" else evaluate()
