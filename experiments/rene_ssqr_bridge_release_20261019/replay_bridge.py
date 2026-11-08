"""Portable independent cache replay. Standard library only; no workspace imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def readl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--output",type=Path,default=Path("replay_output"))
    args=parser.parse_args()
    root=args.bundle.resolve()
    manifest=read(root/"BUNDLE_MANIFEST.json")
    for row in manifest["files"]:
        path=(root/row["path"]).resolve()
        require(path.is_relative_to(root),"bundle path escapes artifact directory")
        require(digest(path)==row["sha256"],f'artifact hash changed: {row["path"]}')
    protocol=read(root/"PROTOCOL.json")
    frozen_selections=readl(root/"selections.jsonl")
    selected={(r["domain"],r["scope"],r["profile"],r["mask"],r["query_id"]):r for r in frozen_selections}
    expected=readl(root/"metrics/per_query.jsonl")
    aggregation_protocol=read(root/"aggregation_audit/PROTOCOL.json")
    aggregation_bins={}
    group_map=read(root/"data/joint_groups.json")
    reference_index={tuple(r[k] for k in ("domain","scope","profile","mask","query_id","retriever")):r for r in expected}
    require(len(reference_index)==len(expected),"duplicate reference metric row")
    output_rows=[]
    comparisons=0
    selection_checks=0
    max_error=0.
    profiles={
        "greedy20_source":("baseline",False),"greedy20_all":("baseline",True),
        "greedy10_source":("greedy10",False),"greedy10_all":("greedy10",True),
        "sample10_source":("sample10",False),"sample10_all":("sample10",True),
    }
    planned_selections=set()
    for domain in ("source","cosqa"):
        for scope in ("main","sensitivity"):
            ids=protocol["samples"][domain]["main_ids" if scope=="main" else "sensitivity_ids"]
            for profile in (["greedy20_source"] if scope=="main" else list(profiles)):
                for mask in ("full","lexical"):
                    planned_selections.update((domain,scope,profile,mask,qid) for qid in ids)
    require(set(selected)==planned_selections and len(selected)==len(frozen_selections),"frozen selection denominator differs from design")
    planned_metric_keys={key+(retriever,) for key in planned_selections for retriever in ("CodeBERT","BM25")}
    require(set(reference_index)==planned_metric_keys,"reference metric denominator differs from frozen design")
    for domain in ("source","cosqa"):
        queries={r["query_id"]:r["query"] for r in readl(root/f"data/{domain}/queries.jsonl")}
        target={r["query_id"]:r["doc_id"] for r in readl(root/f"data/private/{domain}_qrels.jsonl") if r["relevance"]>0}
        require(set(queries)==set(target)==set(protocol["samples"][domain]["main_ids"]),"query/qrel population mismatch")
        banks={name:readl(root/f"candidates/{domain}/{name}.jsonl") for name in ("baseline","greedy20_position0","greedy10","sample10")}
        pool_index={}
        for profile,(bank,include_zero) in profiles.items():
            rows=banks[bank]+(banks["greedy20_position0"] if profile=="greedy20_all" else [])
            pools=defaultdict(dict)
            for row in rows:
                if row["position"]==0 and not include_zero:continue
                require(row["position"] not in pools[row["query_id"]],"duplicate candidate position")
                pools[row["query_id"]][row["position"]]=row
            pool_index[profile]=pools
        for retriever in ("CodeBERT","BM25"):
            ranking_rows=readl(root/f"rankings/{domain}/{retriever}.jsonl")
            ranks={r["text"]:r["ranked_doc_ids"] for r in ranking_rows}
            require(len(ranks)==len(ranking_rows),"duplicate text ranking")
            for row in ranking_rows:
                require(len(row["ranked_doc_ids"])==len(set(row["ranked_doc_ids"]))==100,"incomplete or duplicate ranking")
                require(hashlib.sha256(row["text"].encode()).hexdigest()==row["text_id"],"text hash mismatch")
            for chosen in frozen_selections:
                if chosen["domain"]!=domain:continue
                qid=chosen["query_id"]
                profile=chosen["profile"]
                pool=pool_index[profile][qid]
                reference=reference_index[(domain,chosen["scope"],profile,chosen["mask"],qid,retriever)]
                values=list(pool.values())
                if chosen["mask"]=="full":
                    order=sorted(values,key=lambda r:(r["full_vocab_entropy"],r["position"]))
                else:
                    order=sorted(values,key=lambda r:(r["span_full_vocab_entropy"] is None,
                        math.inf if r["span_full_vocab_entropy"] is None else r["span_full_vocab_entropy"],r["position"]))
                entr=[r["position"] for r in order[:min(3,len(order))]]
                prob=[r["position"] for r in sorted(values,key=lambda r:(-r["label_mean_log_probability"],r["position"]))[:min(3,len(values))]]
                rand=[]
                for seed in protocol["rand_seeds"]:
                    seed_int=int.from_bytes(hashlib.sha256(f"{seed}\0{qid}".encode()).digest(),"big")
                    rand.append([r["position"] for r in random.Random(seed_int).sample(sorted(values,key=lambda r:r["position"]),len(values))[:min(3,len(values))]])
                require((entr,prob,rand)==(chosen["ENTR"],chosen["PROB"],chosen["RAND"]),"outcome-blind selection replay mismatch")
                selection_checks+=1
                def rr(text):
                    ordered=ranks[text]
                    return Fraction(1,ordered.index(target[qid])+1) if target[qid] in ordered else Fraction(0)
                rrs={p:rr(candidate["rewrite"]) for p,candidate in pool.items()}
                if chosen["mask"]=="full" and qid in aggregation_protocol["eligible_ids"][domain][chosen["scope"]]:
                    agg_key=(domain,chosen["scope"],profile,retriever)
                    if agg_key not in aggregation_bins:
                        aggregation_bins[agg_key]={"n":0,"sums":[[Fraction(0) for _ in range(3)] for _ in range(22)],
                                                   "query_max_sums":[Fraction(0) for _ in range(22)]}
                    aggregate=aggregation_bins[agg_key]
                    aggregate["n"]+=1
                    for policy,positions in enumerate([entr,prob]+rand):
                        require(len(positions)==3,"aggregation subset must have exactly three positions")
                        for j,p in enumerate(positions):aggregate["sums"][policy][j]+=rrs[p]
                        aggregate["query_max_sums"][policy]+=max(rrs[p] for p in positions)
                best_entr=max(rrs[p] for p in entr)
                best_prob=max(rrs[p] for p in prob)
                best_rand=sum((max(rrs[p] for p in positions) for positions in rand),Fraction(0))/len(rand)
                calculated={"NoRewrite":rr(queries[qid]),"ENTR_top1":rrs[entr[0]],"ENTR_best3":best_entr,
                            "PROB_best3":best_prob,"RAND20_best3":best_rand,
                            "ENTR_minus_RAND":best_entr-best_rand,"ENTR_minus_PROB":best_entr-best_prob}
                for field,value in calculated.items():
                    max_error=max(max_error,abs(float(value)-reference[field]))
                    comparisons+=1
                output_rows.append({**{k:chosen[k] for k in ("domain","scope","profile","mask","query_id")},
                                    "retriever":retriever,"group":group_map[domain][qid],
                                    **{k:float(v) for k,v in calculated.items()}})
    require(max_error<1e-12,"independent Fraction RR calculation differs")
    by_view=defaultdict(list)
    for row in output_rows:
        key=tuple(row[k] for k in ("domain","scope","profile","mask","retriever"))
        by_view[key].append(row)
    tables=[]
    expected_views={tuple(r[k] for k in ("domain","scope","profile","mask","retriever")):r for r in read(root/"metrics/RESULTS.json")["views"]}
    for key,rows in sorted(by_view.items()):
        mean=lambda field:math.fsum(row[field] for row in rows)/len(rows)
        fields=("NoRewrite","ENTR_top1","ENTR_best3","PROB_best3","RAND20_best3","ENTR_minus_RAND","ENTR_minus_PROB")
        table={**dict(zip(("domain","scope","profile","mask","retriever"),key)),"queries":len(rows),
               **{field:mean(field) for field in fields}}
        table["ENTR_best3_gain"]=mean("ENTR_best3")-mean("NoRewrite")
        reference=expected_views[key]
        for field in ("NoRewrite","ENTR_top1","ENTR_best3","PROB_best3","RAND20_best3"):
            require(abs(table[field]-reference["mrr"][field])<1e-12,"independent table MRR mismatch")
        for field in ("ENTR_minus_RAND","ENTR_minus_PROB"):
            require(abs(table[field]-reference["contrasts"][field]["mean"])<1e-12,"independent table contrast mismatch")
        tables.append(table)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"REPLAY_TABLES.json").write_text(json.dumps(tables,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    aggregation_expected={tuple(r[k] for k in ("domain","scope","profile","retriever")):r
                          for r in read(root/"aggregation_audit/RESULTS.json")["views"]}
    require(set(aggregation_bins)==set(aggregation_expected),"aggregation diagnostic view denominator mismatch")
    aggregation_tables=[]
    aggregation_comparisons=0
    for key,bin_ in sorted(aggregation_bins.items()):
        n=bin_["n"]
        reference=aggregation_expected[key]
        require(n==reference["queries"]==len(aggregation_protocol["eligible_ids"][key[0]][key[1]]),"aggregation query denominator mismatch")
        means=[[value/n for value in policy] for policy in bin_["sums"]]
        for p in range(22):
            for j in range(3):
                require(abs(float(means[p][j])-reference["aggregations"]["best_of_three_rankwise_MRRs"]["means"][p][j])<1e-12,"rank-wise mean mismatch")
                aggregation_comparisons+=1
        modes={"per_query_best3":[v/n for v in bin_["query_max_sums"]],
               "best_of_three_rankwise_MRRs":[max(policy) for policy in means],
               "last_selected_row":[policy[2] for policy in means]}
        calculated={}
        for mode,policy in modes.items():
            rand_mean=sum(policy[2:],Fraction(0))/20
            values={"ENTR":policy[0],"PROB":policy[1],"RAND20":rand_mean,
                    "ENTR_minus_RAND":policy[0]-rand_mean,"ENTR_minus_PROB":policy[0]-policy[1]}
            ref_mode=reference["aggregations"][mode]
            for field,value in values.items():
                expected_value=ref_mode[field]["mean"] if isinstance(ref_mode[field],dict) else ref_mode[field]
                require(abs(float(value)-expected_value)<1e-12,"independent aggregation contrast mismatch")
                aggregation_comparisons+=1
            calculated[mode]={field:float(value) for field,value in values.items()}
        aggregation_tables.append({**dict(zip(("domain","scope","profile","retriever"),key)),"queries":n,"aggregations":calculated})
    (args.output/"AGGREGATION_TABLES.json").write_text(json.dumps(aggregation_tables,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    result={"passed":True,"file_hash_checks":len(manifest["files"]),"independent_selection_checks":selection_checks,
            "fraction_rr_comparisons":comparisons,"maximum_error":max_error,"table_views":len(tables),
            "aggregation_views":len(aggregation_tables),"independent_aggregation_comparisons":aggregation_comparisons,
            "scope":"independent cached computation; no model training or outcome-based policy selection"}
    (args.output/"REPLAY_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
