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
    lines=["# SSQR三候选聚合与公开评估器核对","",
           "公开JSON生成路径按同一URL写出多条改写；公开评估器把预测存为URL→answers的dict，后行覆盖前行。直接执行经Git blob校验的作者评估器，3条同URL合成例结果0，逐query best3为1；这证明接口行为，不证明作者当年生产表格采用该错误路径。","",
           "论文5.1写选前三条并报告最高MRR；公开评估链未明确实现逐query最大RR或三个位置的整体MRR取最大。故将以下三口径作为有限补充诊断同时报告，不回写已冻结CoSQA主比较。","",
           "所有数据均已消费的源数据/CoSQA D，新增桥接效果打开前冻结。要求原query至少3词，使所有profile都具有真实三位置；少于3的查询不补零、不补原句，主实验全分母保持原样。本补充子集数量在PROTOCOL列清。","",
           "| 检索器 | 数据/主桥接查询 | 逐query best3: E−R | 三个rank-wise MRR取最好: E−R | 公开reader末行口径: E−R |",
           "|---|---|---:|---:|---:|"]
    for row in results:
        if row["scope"]!="main":continue
        modes=row["aggregations"]
        a=modes["per_query_best3"]["ENTR_minus_RAND"]["mean"]
        b=modes["best_of_three_rankwise_MRRs"]["ENTR_minus_RAND"]
        c=modes["last_selected_row"]["ENTR_minus_RAND"]["mean"]
        lines.append(f'| {row["retriever"]} | {row["domain"]}/{row["queries"]} | {a:+.8f} | {b:+.8f} | {c:+.8f} |')
    lines += ["","三个rank-wise均值的最大值是非光滑统计量，未把看标签选出的最佳位置当固定政策来计算普通配对区间；RESULTS提供66个policy×rank均值同时Bonferroni cluster-t近似界传播后的保守对比区间。其余两种口径按查询配对做组区间。全部为诊断，不增加确认性检验。","",
              "论文应表述为‘本文明确采用逐查询最大RR的三候选机会指标’，并登记公开聚合路径未恢复。不能把public evaluator末行行为当作原论文真实生产流程，也不能声称原作者结果已被该合成反例推翻。","",
              "原公开evaluator blob: `a75b4944ca9d656641ed6214ce14c2709dca946e`；本轮源码、协议及合成执行证据均保留。完整六profile和PROB结果见RESULTS.json。",""]
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
