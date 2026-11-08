"""Independent persisted-score/ranking/Fraction-RR/statistical replay.

Does not import the production package. --cached skips the full score matrix
and neural spot checks, for the compact publicly reproducible supplement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t
from threadpoolctl import threadpool_limits


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(2**23), b""):
            h.update(block)
    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def exact_rr(order, target):
    check(len(order) == 100 and len(set(order)) == 100, "Not a full unique Top100")
    return Fraction(1, order.index(target)+1) if target in order else Fraction(0)


def interval(values, labels):
    # Independent scalar formulation, with one bootstrap draw per iteration.
    names = sorted(set(labels))
    grouped = {label:[] for label in names}
    for label,value in zip(labels,values):
        grouped[label].append(float(value))
    sizes = np.array([len(grouped[label]) for label in names], dtype=float)
    sums = np.array([math.fsum(grouped[label]) for label in names])
    mean = math.fsum(map(float,values))/len(values)
    se = math.sqrt(len(names)/(len(names)-1)*math.fsum((s-c*mean)**2 for s,c in zip(sums,sizes))/len(values)**2)
    rng = np.random.default_rng(20261020)
    boot=[]
    for _ in range(10000):
        pick=rng.integers(0,len(names),size=len(names))
        boot.append(float(sums[pick].sum()/sizes[pick].sum()))
    critical=float(t.ppf(.975,len(names)-1))
    family=float(t.ppf(1-.05/16,len(names)-1))
    return {"mean":mean,"queries":len(values),"groups":len(names),"se_cluster":se,
            "ci95_group_bootstrap":np.quantile(boot,[.025,.975]).tolist(),
            "ci95_cluster_t":[mean-critical*se,mean+critical*se],
            "eight_effect_family95_bonferroni_bootstrap":np.quantile(boot,[.05/16,1-.05/16]).tolist(),
            "eight_effect_family95_bonferroni_t":[mean-family*se,mean+family*se]}


def audit(root, *, cached=False):
    out=root/"results/rene_envelope_completion_20261020"
    bridge=root/"results/rene_ssqr_bridge_20261019"
    old=root/"results/rene_ssqr_rewriter_envelope_20261020"
    p=read(out/"PROTOCOL.json")
    results=read(out/"RESULTS.json")
    counts=defaultdict(int)
    maximum=defaultdict(float)

    def near(a,b,key,tolerance=1e-12):
        if isinstance(a,list):
            check(len(a)==len(b),"Mismatched vector size")
            for x,y in zip(a,b):
                near(x,y,key,tolerance)
        else:
            error=abs(float(a)-float(b)); maximum[key]=max(maximum[key],error)
            check(error<tolerance,f"{key} differs: {a} vs {b}")
        counts[key]+=1

    if not cached:
        for item in p["bound_files"]+p["protected_files"]:
            check(digest(root/item["path"])==item["sha256"],"Frozen input changed: "+item["path"])
            counts["frozen_files"]+=1
        seal=read(out/"RANKINGS_SEALED.json")
        check(seal["protocol_sha256"]==digest(out/"PROTOCOL.json"),"Protocol seal differs")
        for item in seal["files"]:
            check(digest(root/item["path"])==item["sha256"],"Sealed output changed")
            counts["sealed_files"]+=1

    exported={(r["domain"],r["rewriter"],r["query_id"]):r for r in rows(out/"per_query.jsonl")}
    candidate_export={(r["domain"],r["rewriter"],r["query_id"],r["position"]):r for r in rows(out/"candidate_RR.jsonl")}
    computed={}
    old_errors={}
    for domain in ("source","cosqa"):
        ids=p["samples"][domain]["query_ids"]
        groups=[p["samples"][domain]["groups"][qid] for qid in ids]
        query={r["query_id"]:r["query"] for r in rows(bridge/f"data/{domain}/queries.jsonl")}
        qrels={r["query_id"]:r["doc_id"] for r in rows(bridge/f"data/private/{domain}_qrels.jsonl")}
        corpus_path=bridge/"data/source/corpus.jsonl" if domain=="source" else root/"results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl"
        corpus=rows(corpus_path)
        doc_ids=np.array([r["doc_id"] for r in corpus])
        doc_set=set(map(int,doc_ids))
        local={r["text_id"]:r for r in rows(bridge/f"rankings/{domain}/CodeBERT.jsonl")}
        base_rows=rows(out/f"rankings/{domain}/Top100.jsonl")
        base={r["text_id"]:r for r in base_rows}
        text_list=read(out/f"rankings/{domain}/texts.json")
        check([r["text"] for r in base_rows]==text_list,"Text-vector row mapping differs")
        for row in base_rows:
            check(hashlib.sha256(row["text"].encode()).hexdigest()==row["text_id"],"Text ID mismatch")
            check(set(row["ranked_doc_ids"]).issubset(doc_set),"Unknown document")
        if not cached:
            dense=np.load(out/f"rankings/{domain}/scores.npy",mmap_mode="r")
            qvec=np.load(out/f"rankings/{domain}/query_vectors.npy")
            code_path=bridge/"retrieval_cache/source/CodeBERT/code_vectors.npy" if domain=="source" else root/"results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy"
            cvec=np.load(code_path)
            check(dense.shape==(len(text_list),len(corpus)),"Score matrix size differs")
            by_id=np.argsort(doc_ids,kind="stable")
            with threadpool_limits(limits=4):
                for offset in range(0,len(text_list),128):
                    recomputed=np.matmul(qvec[offset:offset+128],cvec.T)
                    near(float(np.max(np.abs(recomputed-dense[offset:offset+len(recomputed)]))),0,"all_full_corpus_scores",1e-6)
                    counts["score_cells"]+=int(recomputed.size)
                    for j in range(len(recomputed)):
                        i=offset+j; row=base_rows[i]
                        # Stable score sort after a separate ID sort, not production lexsort.
                        expected=by_id[np.argsort(-dense[i,by_id],kind="stable")[:100]]
                        check(doc_ids[expected].tolist()==row["ranked_doc_ids"],"Independent full Top100 differs")
                        near(dense[i,expected].tolist(),row["scores"],"Top100_scores",1e-6)
                        counts["full_ranking_rows"]+=1
            # A separate FP64 path checks the dot-product interpretation, not FP32 tie identity.
            for i in np.linspace(0,len(text_list)-1,8,dtype=int):
                oracle=qvec[i].astype(np.float64)@cvec.astype(np.float64).T
                error=float(np.max(np.abs(oracle-dense[i])))
                maximum["FP64_score_sanity_abs"]=max(maximum["FP64_score_sanity_abs"],error)
                check(error<1e-4,"Raw-dot-product FP64 spot differs")
                counts["FP64_score_spots"]+=1
            del dense,qvec,cvec

        per_arm={}
        for arm in ("local","t5base"):
            path=old/f"candidates/{domain}/t5base.jsonl" if arm=="t5base" else bridge/f"candidates/{domain}/baseline.jsonl" if domain=="source" else root/"results/rene_cosqa_development_20261019/candidates.jsonl"
            grouped=defaultdict(list)
            for row in rows(path):
                if row["query_id"] in set(ids):
                    grouped[row["query_id"]].append(row)
            ranks=local if arm=="local" else base
            statistics=[]
            for qid in ids:
                pool=sorted(grouped[qid],key=lambda r:r["position"])
                check([r["position"] for r in pool]==list(range(1,len(query[qid].split())+1)),"Position pool differs")
                values=[]
                for candidate in pool:
                    tid=hashlib.sha256(candidate["rewrite"].encode()).hexdigest()
                    value=exact_rr(ranks[tid]["ranked_doc_ids"],qrels[qid]);values.append(value)
                    near(float(value),candidate_export[domain,arm,qid,candidate["position"]]["rr"],"Fraction_candidate_RR")
                baseline=exact_rr(ranks[hashlib.sha256(query[qid].encode()).hexdigest()]["ranked_doc_ids"],qrels[qid])
                order=sorted(range(len(pool)),key=lambda i:(pool[i]["full_vocab_entropy"],pool[i]["position"]))
                probs=sorted(range(len(pool)),key=lambda i:(-pool[i]["label_mean_log_probability"],pool[i]["position"]))
                k=min(3,len(pool)); ordered=sorted(values)
                exact_uniform=sum((value*math.comb(i,k-1) for i,value in enumerate(ordered) if i>=k-1),Fraction(0))/math.comb(len(pool),k)
                answer={"NoRewrite":baseline,"ENTR_top1":values[order[0]],"ENTR_best3":max(values[i] for i in order[:3]),"Oracle_all":max(values),
                        "Uniform_single_exact":sum(values,Fraction(0))/len(values),"Uniform_best3_exact":exact_uniform,
                        "PROB_top1":values[probs[0]],"PROB_best3":max(values[i] for i in probs[:3])}
                saved=exported[domain,arm,qid]
                for key,value in answer.items():
                    near(float(value),saved["RR"][key],"Fraction_query_RR")
                check(saved["selected_positions"]==[pool[i]["position"] for i in order[:3]],"Selected positions differ")
                entropy=[c["full_vocab_entropy"] for c in pool]
                if len(set(entropy))<2 or len(set(values))<2:
                    corr=None
                else:
                    x=rankdata(entropy);y=rankdata([float(v) for v in values]);corr=float(np.corrcoef(x,y)[0,1])
                if corr is None:
                    check(saved["entropy_spearman"] is None,"Degenerate correlation must remain missing")
                else:
                    near(corr,saved["entropy_spearman"],"correlation")
                statistics.append({"RR":answer,"corr":corr})
            view=results["views"][domain+"/"+arm]
            for key in statistics[0]["RR"]:
                near(math.fsum(float(s["RR"][key]) for s in statistics)/len(ids),view["mrr"][key],"MRR")
            for name,published in view["effects"].items():
                left,right=name.split("-")
                summary=interval([s["RR"][left]-s["RR"][right] for s in statistics],groups)
                for key,value in summary.items():
                    near(value,published[key],"independent_interval_fields")
            reference=read(old/f"{domain}_{arm}.json")
            errors={new:abs(view["mrr"][new]-reference["mrr"][oldkey]) for new,oldkey in [("NoRewrite","nowrite"),("ENTR_top1","entr_top1"),("ENTR_best3","entr_best3"),("Oracle_all","oracle")]}
            corrs=[s["corr"] for s in statistics if s["corr"] is not None]
            errors["correlation"]=abs(math.fsum(corrs)/len(corrs)-reference["spearman_entropy"])
            old_errors[domain+"/"+arm]=max(errors.values())
            check(view["correlation_valid_queries"]==len(corrs),"Correlation denominator differs")
            for key in ("ENTR_top1","ENTR_best3","Oracle_all"):
                counts_expected={"beneficial_queries":sum(s["RR"][key]>s["RR"]["NoRewrite"] for s in statistics),"harmful_queries":sum(s["RR"][key]<s["RR"]["NoRewrite"] for s in statistics)}
                check(counts_expected==view["outcomes"][key],"Outcome frequencies differ")
            per_arm[arm]=statistics
            computed[domain+"/"+arm]=statistics
        for key,published in results["paired_base_minus_local"][domain].items():
            summary=interval([b["RR"][key]-a["RR"][key] for a,b in zip(per_arm["local"],per_arm["t5base"])],groups)
            for field,value in summary.items():
                near(value,published[field],"paired_weight_intervals")
        check(all(a["RR"]["NoRewrite"]==b["RR"]["NoRewrite"] for a,b in zip(per_arm["local"],per_arm["t5base"])),"Original control is not shared")
        counts["common_original_RR"]+=len(ids)

    if not cached:
        # Independent actual model forward: no production loader or encoder.
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model_path=root/"models/codebert-ssqr-pooler-2epoch-20261019"
        model=AutoModel.from_pretrained(model_path,local_files_only=True).float().cuda().eval()
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        with torch.inference_mode():
            for domain in ("source","cosqa"):
                corpus_path=bridge/"data/source/corpus.jsonl" if domain=="source" else root/"results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl"
                corpus=rows(corpus_path)
                texts=read(out/f"rankings/{domain}/texts.json")
                qvec=np.load(out/f"rankings/{domain}/query_vectors.npy")
                code_path=bridge/"retrieval_cache/source/CodeBERT/code_vectors.npy" if domain=="source" else root/"results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy"
                cvec=np.load(code_path)
                qpick=np.linspace(0,len(texts)-1,8,dtype=int)
                cpick=np.linspace(0,len(corpus)-1,8,dtype=int)
                for label,inputs,expected in [("query",[texts[i] for i in qpick],qvec[qpick]),("corpus",[corpus[i]["code_text"] for i in cpick],cvec[cpick])]:
                    batch=tokenizer(inputs,padding="max_length",truncation=True,max_length=256,return_tensors="pt")
                    actual=model(**{k:v.cuda() for k,v in batch.items()}).pooler_output.float().cpu().numpy()
                    error=float(np.max(np.abs(actual-expected)))
                    maximum["neural_"+label+"_max_abs"]=max(maximum["neural_"+label+"_max_abs"],error)
                    check(error<5e-6,"Independent actual model encoding differs")
                    counts["actual_neural_"+label+"_examples"]+=len(inputs)
        del model
        torch.cuda.empty_cache()

    result={"passed":True,"created_utc":datetime.now(timezone.utc).isoformat(),"cached_only":cached,"no_production_imports":True,
            "counts":dict(counts),"max_errors":dict(maximum),"old_summary_max_errors":old_errors,
            "old_summaries_reproduced_at_1e12":all(v<1e-12 for v in old_errors.values()),"new_confirmation":False,"H_read":False}
    destination=out/("CACHED_AUDIT.json" if cached else "AUDIT.json")
    with destination.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2);handle.write("\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cached",action="store_true")
    args=parser.parse_args()
    audit(args.root,cached=args.cached)
