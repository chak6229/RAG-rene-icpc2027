from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/rene_evidence_integration_20261020"
PAPER=ROOT/"paper/rene2027"
SOURCES={
    "confirmation":"results/rene_cosqa_confirmation_20261019/PRIMARY_RESULT.json",
    "H_CodeBERT":"results/rene_cosqa_confirmation_20261019/metrics/CodeBERT_full_H.json",
    "H_BM25":"results/rene_cosqa_confirmation_20261019/metrics/BM25_full_H.json",
    "A":"results/rene_core_step_a_20261020/RESULTS.json",
    "B":"results/rene_core_step_b_20261020/RESULTS.json",
    "fusion":"results/rene_multicandidate_fusion_20261020/RESULTS.json",
    "envelope":"results/rene_envelope_completion_20261020/RESULTS.json",
    "seed1536":"results/rene_strict_confirmation_20261019/RESULTS.json",
    "risk1024":"results/rene_uq_independent_proof_20261019/RESULTS.json",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")


def get(data,pointer):
    result=data
    for part in pointer.split("/"):
        if part:
            result=result[int(part)] if isinstance(result,list) else result[part.replace("~1","/").replace("~0","~")]
    return result


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    data={key:load(ROOT/path) for key,path in SOURCES.items()}
    cells=[]

    def value(source,pointer):
        result=get(data[source],pointer)
        cells.append({"source":SOURCES[source],"pointer":pointer,"value":result})
        return result

    tables={}
    primary=data["confirmation"]
    tables["T1_confirmation"]=[{
        "comparison":"CodeBERT ENTR-RAND20 best3", "role":"single frozen confirmatory comparison",
        "queries":value("confirmation","queries"),"groups":value("confirmation","groups"),
        "delta":value("confirmation","mean"),"ci95":value("confirmation","ci95_group_bootstrap"),
        "practical_margin":.005,"equivalence":value("confirmation","conclusion/practical_equivalence_within_prespecified_plusminus005"),
    }]
    tables["T2_same_pool_contracts"]=[]
    for index,view in enumerate(data["A"]["views"]):
        prefix=f"views/{index}"
        row={"domain":value("A",prefix+"/domain"),"retriever":value("A",prefix+"/retriever"),
             "queries":value("A",prefix+"/queries"),"groups":value("A",prefix+"/groups"),"role":"post-hoc same-pool paired diagnostic"}
        for name,contrast in [("single_advantage","a_ENTR_single_minus_uniform"),("best3_advantage","b_ENTR_best3_minus_RAND20"),("direct_difference","j_ENTR_contract_difference")]:
            row[name]=value("A",prefix+f"/contrasts/{contrast}/mean")
            row[name+"_ci95"]=value("A",prefix+f"/contrasts/{contrast}/ci95_group_bootstrap")
        row["direct_difference_family95"]=value("A",prefix+"/contrasts/j_ENTR_contract_difference/four_cell_family95_bonferroni_group_bootstrap")
        tables["T2_same_pool_contracts"].append(row)
    tables["T3_adoption_fixed20"]=[]
    for retriever in ("CodeBERT","BM25"):
        for model in ("M0","MT","MR","MRT"):
            prefix=f"views/{retriever}/policies/fixed20/{model}"
            tables["T3_adoption_fixed20"].append({"retriever":retriever,"model":model,"role":"development fixed OOF outputs; paired intervals conditional on fitted heads",
                "delta":value("B",prefix+"/delta_mrr"),"adopted":value("B",prefix+"/adopted"),"harmful":value("B",prefix+"/harmful"),
                "ci95":value("B",f"views/{retriever}/estimates/fixed20:{model}-NoRewrite/ci95_group_bootstrap")})
    tables["T4_multicandidate_fusion"]=[]
    for retriever in ("CodeBERT","BM25"):
        for policy in ("ENTRTop1","RRF3","RRF4","ENTRBest3"):
            prefix=f"views/{retriever}/policies/{policy}"
            row={"retriever":retriever,"policy":policy,"role":"consumed-development post-hoc extension",
                 "delta":value("fusion",prefix+"/delta_mrr"),"harmful":value("fusion",prefix+"/harmful_queries"),
                 "ci95":value("fusion",f"views/{retriever}/estimates/{policy}-NoRewrite/ci95_group_bootstrap"),
                 "executable":policy!="ENTRBest3"}
            if policy in ("RRF3","RRF4"):
                row["family95"]=value("fusion",f"views/{retriever}/estimates/{policy}-NoRewrite/four_effect_family95_bonferroni_group_bootstrap")
            tables["T4_multicandidate_fusion"].append(row)
    tables["T5_frozen_H_stages"]=[]
    for ret in ("CodeBERT","BM25"):
        source="H_"+ret
        for policy in ("ENTR_top1","HGB_gain_control","RIDGE_gain","RIDGE_verified","RIDGE_risk"):
            prefix=f"policy_results/{policy}"
            tables["T5_frozen_H_stages"].append({"retriever":ret,"policy":policy,"queries":value(source,"queries"),"groups":value(source,"groups"),
                "role":"predefined descriptive full-H stage estimate; not an extra confirmatory test",
                "delta":value(source,prefix+"/delta_MRR/mean"),"ci95":value(source,prefix+"/delta_MRR/ci95_group_bootstrap"),
                "adopted":value(source,prefix+"/adopted"),"harmful":value(source,prefix+"/harmed"),
                "positive_gain_retention":value(source,prefix+"/positive_gain_retention/value"),"CE_pairs":value(source,prefix+"/CE_pairs_charged")})
    tables["T6_two_T5_weights"]=[]
    for key,view in data["envelope"]["views"].items():
        domain,weight=key.split("/")
        prefix=f"views/{domain}~1{weight}"
        row={"domain":domain,"T5_weights":weight,"queries":value("envelope",prefix+"/queries"),"groups":value("envelope",prefix+"/groups"),"role":"same historical sensitivity subset; post-hoc two-weight diagnostic"}
        for label,contrast in [("top1_delta","ENTR_top1-NoRewrite"),("best3_delta","ENTR_best3-NoRewrite")]:
            row[label]=value("envelope",prefix+f"/effects/{contrast}/mean")
            row[label+"_ci95"]=value("envelope",prefix+f"/effects/{contrast}/ci95_group_bootstrap")
            row[label+"_family95"]=value("envelope",prefix+f"/effects/{contrast}/eight_effect_family95_bonferroni_bootstrap")
        row["entropy_single_vs_uniform"]=value("envelope",prefix+"/effects/ENTR_top1-Uniform_single_exact/mean")
        row["entropy_single_vs_uniform_ci95"]=value("envelope",prefix+"/effects/ENTR_top1-Uniform_single_exact/ci95_group_bootstrap")
        tables["T6_two_T5_weights"].append(row)
    tables["T7_previous_seed_UQ"]=[]
    for source,contrast,population,ci,role in [
        ("seed1536","uq_increment","1536 downstream repositories; full pipeline","simultaneous_ci97_5","prior frozen full-pipeline comparison"),
        ("risk1024","full_minus_common","1024 downstream repositories; fixed upstream action","simultaneous_envelope","prior frozen risk-head comparison"),
        ("risk1024","full_minus_support","1024 downstream repositories; fixed upstream action","simultaneous_envelope","prior frozen risk-head comparison"),
    ]:
        tables["T7_previous_seed_UQ"].append({"population":population,"comparison":contrast,"role":role,
            "delta":value(source,f"primary_contrasts/{contrast}/mean"),"ci97_5":value(source,f"primary_contrasts/{contrast}/{ci}")})
    bundle={"created_utc":datetime.now(timezone.utc).isoformat(),"sources":[{"key":key,"path":path,"sha256":sha(ROOT/path)} for key,path in SOURCES.items()],
            "tables":tables,"traceable_cells":cells,"no_new_estimates_from_H":True,"no_cross_population_pooling":True,
            "original_primary_unchanged":True,"risk_certified":False}
    write_json(OUT/"TABLES.json",bundle)
    for name,records in tables.items():
        with (OUT/(name+".csv")).open("x",encoding="utf-8-sig",newline="") as handle:
            columns=list(dict.fromkeys(key for row in records for key in row))
            writer=csv.DictWriter(handle,fieldnames=columns)
            writer.writeheader()
            for record in records:
                writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in record.items()})
    def n(x):return f"{x:+.6f}"
    def ci(x):return "["+", ".join(n(v) for v in x)+"]"
    lines=["# Integrated replication and controlled-extension results","",
           "Generated from sealed machine results. Primary confirmation, predefined descriptive estimates, and post-hoc diagnostics retain their original roles. All effects are MRR@100 differences; populations are not pooled.","",
           "## RQ1: external replication of position selection","",
           "|Population|Comparison|ΔMRR|Group-bootstrap 95% CI|Evidence role|","|---|---|---:|---|---|"]
    for r in tables["T1_confirmation"]:
        lines.append(f"|{r['queries']} queries / {r['groups']} groups|{r['comparison']}|{n(r['delta'])}|{ci(r['ci95'])}|Single frozen comparison; both 95% intervals inside prespecified ±.005|")
    lines += ["","The result concerns the declared per-query best-of-three opportunity contract in the reconstructed implementation. It does not establish identity with unavailable author weights or a causal effect of query provenance.","",
              "## RQ1 extension: same-pool candidate budget and evaluation contract","",
              "|Dataset / retriever|ENTR single−uniform single|ENTR best3−RAND20 best3|Direct difference|Four-effect family interval for direct difference|",
              "|---|---:|---:|---:|---|"]
    for r in tables["T2_same_pool_contracts"]:
        lines.append(f"|{r['domain']} / {r['retriever']}|{n(r['single_advantage'])}|{n(r['best3_advantage'])}|{n(r['direct_difference'])}|{ci(r['direct_difference_family95'])}|")
    lines += ["","CoSQA/CodeBERT directly supports a larger relative ENTR advantage under the single-candidate contract. PROB behaves similarly; ENTR-specific superiority is not established. Budget and maximum aggregation change together, so the contrast is not a pure causal effect of an uncertainty mechanism.","",
              "## RQ2: fixed-action adoption with matched supervision","",
              "M0: common evidence; MT: M0 plus generation confidence; MR: M0 plus original-query reference; MRT: both. Fixed20 adopts the top 20% of eligible actions within each frozen OOF fold. The denominator remains all 1,490 development queries.","",
              "|Retriever|Head|ΔMRR|Group-bootstrap 95% CI|Harmful / adopted|","|---|---|---:|---|---|"]
    for r in tables["T3_adoption_fixed20"]:
        lines.append(f"|{r['retriever']}|{r['model']}|{n(r['delta'])}|{ci(r['ci95'])}|{r['harmful']}/{r['adopted']}|")
    lines += ["","At fixed adoption count, CodeBERT MRT−M0 is +.00002276. This comparison does not establish extra adoption utility from generation confidence. The two decision tasks in the previous table differ; significance in one and non-significance in another is not a tested stage interaction.","",
              "## RQ2 extension: executable multi-candidate fusion","",
              "All rows use the same native-ENTR candidate pool. RRF3 merges the three candidate lists; RRF4 also includes the original query. RRF constant 60, equal unique-input weights, Top100 inputs and exact rational ordering were fixed before these fusion effects. Best3 uses relevance labels and is not executable or a bound on fused rankings.","",
              "|Retriever|Policy|ΔMRR|Group-bootstrap 95% CI|Harmful queries / 1490|","|---|---|---:|---|---:|"]
    for r in tables["T4_multicandidate_fusion"]:
        lines.append(f"|{r['retriever']}|{r['policy']}|{n(r['delta'])}|{ci(r['ci95'])}|{r['harmful']}|")
    lines += ["","The four fusion-versus-original effects do not establish positive utility; after four-effect correction, only BM25 RRF3 remains negative. BM25 fusion improves over forced Top1 in paired secondary comparisons. Including the original query has a positive point trend over RRF3, with intervals crossing zero for both retrievers. Unique retrieval inputs per query are 1 for NoRewrite, 1 for standalone Top1, 2.9799 for RRF3, and 3.9799 for RRF4; these counts are not measured latency.","",
              "## Frozen full-H selection, verification and fallback","",
              "These predefined descriptive stages use all 8,530 H queries / 1,963 groups. They are not the 7,718-query primary sample and are not interchangeable with development results. No H outcomes were reopened to build these tables.","",
              "|Retriever|Stage|ΔMRR|Group-bootstrap 95% CI|Harmful / adopted|Positive-gain retention|CE pairs|",
              "|---|---|---:|---|---|---:|---:|"]
    for r in tables["T5_frozen_H_stages"]:
        lines.append(f"|{r['retriever']}|{r['policy']}|{n(r['delta'])}|{ci(r['ci95'])}|{r['harmful']}/{r['adopted']}|{100*r['positive_gain_retention']:.2f}%|{r['CE_pairs']}|")
    lines += ["","Positive-gain retention is defined against the same candidate pool with an original-query safe arm. It is not a fraction of net mean gain, and it is not used for the new RRF rankings. Risk-stage empirical harm and cost accounting do not issue a population risk certificate.","",
              "## Robustness: two T5 weight sets on matched sensitivity queries","",
              "Each domain uses its historical one-query-per-group 512-query sample, identical insertion-position and retrieval contracts. The additional base-weight rankings have now been persisted and independently audited. Both weights belong to T5-base; these are not independent model families or fresh confirmations.","",
              "|Dataset|T5 weights|Top1−original|95% CI|Best3−original|95% CI|",
              "|---|---|---:|---|---:|---|"]
    for r in tables["T6_two_T5_weights"]:
        lines.append(f"|{r['domain']}|{r['T5_weights']}|{n(r['top1_delta'])}|{ci(r['top1_delta_ci95'])}|{n(r['best3_delta'])}|{ci(r['best3_delta_ci95'])}|")
    lines += ["","All four best3 gains remain positive under the eight-effect Bonferroni sensitivity. Two of four Top1 intervals cross zero after that correction: source/t5-base and CoSQA/local. CoSQA/t5-base single-candidate ENTR outperforms exact uniform selection by +.03475321, 95% [.01882089,.05044681], while remaining below the original query. Therefore below-original utility cannot by itself diagnose an uninformative entropy ranking.","",
              "## RQ3: preserve the distinct prior multi-seed results","",
              "|Population / target|Comparison|ΔMRR|97.5% simultaneous interval|","|---|---|---:|---|"]
    for r in tables["T7_previous_seed_UQ"]:
        lines.append(f"|{r['population']}|{r['comparison']}|{n(r['delta'])}|{ci(r['ci97_5'])}|")
    lines += ["","The 1,536-query comparison supports a positive increment in that frozen full pipeline. The 1,024-query fixed-action full−common comparison meets its prespecified ±.005 equivalence rule; full−support remains unresolved because its event guard is not met. These datasets, upstream actions and estimands differ; the later comparison is not a causal decomposition or retraction of the earlier positive result.","",
              "## Mainline and practical contribution","",
              "The study tests SSQR's specified position-selection claim on a new query source and measures its interpretation under explicit candidate budgets, opportunity statistics and executable policies. The most specific controlled result is the same-query contract contrast, alongside the preserved external confirmation. Fixed-action evidence controls, simple fusion and frozen pipeline stages delimit what is achieved automatically in these settings. They support a replication with controlled empirical extensions, rather than a newly validated UQ algorithm.","",
              "SSQR already reports Top1/Top3 differences; QueryRoute and QPP variant studies cover general selection gaps and decision-oriented evaluation. The study's contribution therefore rests on the original method-specific replication and controlled contrasts, not on claiming those general concepts as new. Practitioners should report both original-query and random-selection references, distinguish label-assisted opportunity from execution, and measure net benefit separately from harm frequency and cost.","",
              "Sources, JSON pointers and exact values are recorded in results/rene_evidence_integration_20261020/TABLES.json. Implementation fidelity remains in implementation_fidelity.md; focused full-text comparisons are in results/rene_multicandidate_fusion_20261020/PRIOR_WORK_COMPARISON.md.",""]
    text="\n".join(lines)
    with (OUT/"MAIN_RESULTS.md").open("x",encoding="utf-8") as handle:handle.write(text)
    with (PAPER/"integrated_results.md").open("x",encoding="utf-8") as handle:handle.write(text)
    print(json.dumps({"tables":len(tables),"rows":sum(len(v) for v in tables.values()),"traceable_cells":len(cells)},ensure_ascii=False))


if __name__=="__main__":
    build()
