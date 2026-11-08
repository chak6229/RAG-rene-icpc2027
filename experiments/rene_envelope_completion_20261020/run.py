from __future__ import annotations

import argparse
import gc
import math
import platform
import time
from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from .common import (ROOT, PACKAGE, OUT, BRIDGE, OLD, MODEL, DOMAINS, PROTECTED,
                     deny_labels, now, paths, read, record, require, rows, sha,
                     text_id, verify, verify_seal, write, writel)
from .metrics import correlation, rr, summarize, uniform_max


def public_inputs(domain):
    ids = read(BRIDGE / "PROTOCOL.json")["samples"][domain]["sensitivity_ids"]
    p = paths(domain)
    queries = {r["query_id"]: r["query"] for r in rows(p["queries"])}
    candidates = rows(p["base_candidates"])
    require(set(r["query_id"] for r in candidates) == set(ids), "Base query scope differs")
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["query_id"]].append(row)
    for qid in ids:
        values = grouped[qid]
        require(sorted(r["position"] for r in values) == list(range(1, len(queries[qid].split()) + 1)), "Position scope differs")
        require([r["position"] for r in values] == sorted(r["position"] for r in values), "Preserve original within-query order")
        for row in values:
            words, pos = queries[qid].split(), row["position"]
            require(row["rewrite"] == " ".join(words[:pos] + ([row["span"]] if row["span"] else []) + words[pos:]), "Candidate identity differs")
            require(math.isfinite(row["full_vocab_entropy"]), "Nonfinite entropy")
    texts = list(dict.fromkeys([queries[qid] for qid in ids] + [r["rewrite"] for r in candidates]))
    require(len({text_id(t) for t in texts}) == len(texts), "Text digest collision")
    return ids, queries, candidates, texts


def freeze():
    require(not (OUT / "PROTOCOL.json").exists(), "Protocol already exists")
    bound = list(PACKAGE.glob("*.py"))
    bound += [BRIDGE / "PROTOCOL.json", BRIDGE / "RANKINGS_SEALED.json", OLD / "ENVELOPE.md",
              ROOT / "experiments/rene_ssqr_rewriter_envelope_20261020/generate_envelope.py",
              ROOT / "experiments/rene_ssqr_alignment_20261019/retriever.py",
              ROOT / "experiments/rene_ssqr_alignment_20261019/pipeline.py"]
    for domain in DOMAINS:
        bound += list(paths(domain).values())
        bound += [OLD / f"{domain}_{arm}.json" for arm in ("local", "t5base")]
    for model in [MODEL, ROOT / "models/t5-base", ROOT / "models/t5-cqc-codenn-1m/best"]:
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "spiece.model", "ssqr_representation_contract.json", "TRAINING_CONTRACT.json", "training_complete.json"):
            if (model / name).is_file():
                bound.append(model / name)
    samples = {}
    bridge = read(BRIDGE / "PROTOCOL.json")
    for domain in DOMAINS:
        ids, queries, candidates, texts = public_inputs(domain)
        groups = bridge["samples"][domain]["groups"]
        samples[domain] = {"query_ids": ids, "groups": {qid: groups[qid] for qid in ids},
                           "queries": len(ids), "unique_groups": len(set(groups[qid] for qid in ids)),
                           "candidate_rows": len(candidates), "unique_retrieval_inputs": len(texts)}
    write(OUT / "PROTOCOL.json", {
        "created_utc": now(), "status": "FROZEN_BEFORE_RERETRIEVAL",
        "role": "post_hoc_evidence_completion_not_new_confirmation",
        "prior_aggregate_results_known": True, "new_confirmation": False,
        "purpose": "Persist missing base rankings/embeddings and independently recover two-weight envelope on identical old 512-query samples",
        "samples": samples, "new_T5_generation": False, "new_training": False,
        "retrieval": {"model": str(MODEL.relative_to(ROOT)), "representation": "pooler_output", "normalization": "none", "query_batch": 16, "max_length": 256,
                       "precision": "FP32, TF32 disabled", "score": "numpy FP32 raw dot product; query chunks 128; four BLAS threads",
                       "tie_rule": "descending persisted FP32 score, ascending original integer doc_id", "depth": 100,
                       "code_vectors": "reuse bound frozen full-corpus vectors with corpus-row doc_id map; independently reencode fixed deterministic document spots"},
        "metrics": {"baseline": "each arm's persisted original-query ranking; independently verify common original RR matches old local baseline",
                    "single": "minimum entropy then original position", "best3": "max RR of three smallest entropy positions; no original fallback",
                    "oracle": "max RR over all candidate positions; not fused-ranking upper bound",
                    "uniform_single": "exact mean over positions; duplicate text keeps position probability mass",
                    "uniform_best3": "exact uniform without replacement over positions; no RNG", "PROB": "largest mean label log probability then original position"},
        "statistics": {"query_weighted": True, "resample_groups": True, "bootstrap": 10000, "seed": 20261020,
                       "main_descriptive_family": "8 effects = Top1 and best3 minus NoRewrite across 2 domains x 2 T5 weights",
                       "family_size": 8, "other_contrasts": "descriptive pointwise95; paired base-local differences on shared queries",
                       "no_equivalence_claim": True, "no_source_causal_claim": True},
        "no_H_evaluation": True, "no_parameter_search": True, "risk_certified": False,
        "bound_files": [record(f) for f in sorted(set(bound))],
        "protected_files": [record(ROOT / f) for f in PROTECTED],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    })
    print({"stage": "frozen", "samples": {d: {k: v for k, v in s.items() if k not in ("query_ids", "groups")} for d,s in samples.items()}}, flush=True)


def retrieve():
    p = verify()
    require(not (OUT / "RANKINGS_SEALED.json").exists(), "Rankings are already sealed")
    import torch
    from experiments.rene_ssqr_alignment_20261019.pipeline import load_trained_retriever
    from experiments.rene_ssqr_alignment_20261019.retriever import encode_texts
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(101)
    files, jobs = [], []
    with deny_labels():
        for domain in DOMAINS:
            ids, queries, candidates, texts = public_inputs(domain)
            corpus = rows(paths(domain)["corpus"])
            codes = np.load(paths(domain)["codes"], allow_pickle=False)
            require(codes.dtype == np.float32 and codes.shape == (len(corpus), 768), "Code-vector mapping differs")
            doc_ids = np.array([r["doc_id"] for r in corpus], dtype=np.int64)
            require(len(set(doc_ids.tolist())) == len(corpus), "Duplicate document identities")
            destination = OUT / f"rankings/{domain}"
            destination.mkdir(parents=True, exist_ok=True)
            vector_path, text_path = destination / "query_vectors.npy", destination / "texts.json"
            manifest_path = destination / "MANIFEST.json"
            if manifest_path.exists():
                manifest = read(manifest_path)
                for item in manifest["files"]:
                    require(sha(ROOT / item["path"]) == item["sha256"], "Existing retrieval artifact changed")
                files += [ROOT / item["path"] for item in manifest["files"]] + [manifest_path]
                jobs.append(manifest)
                continue
            require(not vector_path.exists(), "Unsealed vectors already exist")
            started = time.perf_counter()
            print({"stage": "encode", "domain": domain, "texts": len(texts)}, flush=True)
            model, tokenizer = load_trained_retriever(MODEL, "cuda")
            chunks = []
            # Chunk boundaries are multiples of the historical batch size.
            for offset in range(0, len(texts), 256):
                chunks.append(encode_texts(model, tokenizer, texts[offset:offset+256], batch_size=16, device="cuda"))
                print({"stage": "encode", "domain": domain, "completed": min(offset+256,len(texts)), "total":len(texts), "seconds":round(time.perf_counter()-started,1)}, flush=True)
            vectors = np.concatenate(chunks)
            del model, chunks
            gc.collect()
            torch.cuda.empty_cache()
            with vector_path.open("xb") as h:
                np.save(h, vectors, allow_pickle=False)
            write(text_path, texts)
            rank_path = destination / "Top100.jsonl"
            dense_path = destination / "scores.npy"
            require(not dense_path.exists(), "Full scores already exist")
            dense = np.lib.format.open_memmap(dense_path, mode="w+", dtype=np.float32, shape=(len(texts),len(corpus)))
            output_rows = []
            with threadpool_limits(limits=4):
                for offset in range(0,len(texts),128):
                    scores = vectors[offset:offset+128] @ codes.T
                    dense[offset:offset+len(scores)] = scores
                    for text, values in zip(texts[offset:offset+len(scores)],scores):
                        order = np.lexsort((doc_ids,-values))[:100]
                        output_rows.append({"text_id":text_id(text),"text":text,"ranked_doc_ids":doc_ids[order].tolist(),"scores":values[order].astype(float).tolist()})
            dense.flush()
            del dense
            writel(rank_path,output_rows)
            manifest = {"created_utc":now(),"domain":domain,"qrels_read":False,"texts":len(texts),"documents":len(corpus),"seconds":time.perf_counter()-started,
                        "files":[record(f) for f in [vector_path,text_path,rank_path,dense_path]],"code_vectors":record(paths(domain)["codes"]),"corpus":record(paths(domain)["corpus"])}
            write(manifest_path,manifest)
            files += [vector_path,text_path,rank_path,dense_path,manifest_path]
            jobs.append(manifest)
            print({"stage":"retrieved","domain":domain,"seconds":round(manifest["seconds"],1)},flush=True)
    verify()
    write(OUT/"RANKINGS_SEALED.json",{"created_utc":now(),"protocol_sha256":sha(OUT/"PROTOCOL.json"),"files":[record(f) for f in files],"jobs":jobs,"qrels_read":False})


def evaluate():
    p = verify()
    verify_seal("RANKINGS_SEALED.json")
    require(not (OUT/"RESULTS.json").exists(),"Results exist")
    write(OUT/"OUTCOME_ACCESS.json",{"created_utc":now(),"rankings_seal_sha256":sha(OUT/"RANKINGS_SEALED.json"),"protocol_sha256":sha(OUT/"PROTOCOL.json"),"H_read":False,"prior_aggregate_results_known":True})
    all_query, candidate_rr, views, paired, controls = [], [], {}, {}, {}
    for domain in DOMAINS:
        ids, queries, base_candidates, texts = public_inputs(domain)
        qrels = {r["query_id"]:int(r["doc_id"]) for r in rows(paths(domain)["qrels"])}
        local_rankings = {r["text_id"]:r["ranked_doc_ids"] for r in rows(paths(domain)["local_rankings"])}
        base_rankings = {r["text_id"]:r["ranked_doc_ids"] for r in rows(OUT/f"rankings/{domain}/Top100.jsonl")}
        group_map = p["samples"][domain]["groups"]
        groups = [group_map[qid] for qid in ids]
        original_diff = [rr(base_rankings[text_id(queries[qid])],qrels[qid])-rr(local_rankings[text_id(queries[qid])],qrels[qid]) for qid in ids]
        controls[domain]={"original_RR_differences":sum(v!=0 for v in original_diff),"max_original_RR_difference":max(map(abs,original_diff))}
        require(not any(original_diff),"New original RR differs from frozen common baseline; investigate before matched comparisons")
        by_arm = {}
        for arm in ("local","t5base"):
            candidate_rows = rows(paths(domain)["local_candidates"]) if arm=="local" else base_candidates
            grouped=defaultdict(list)
            for row in candidate_rows:
                if row["query_id"] in group_map:
                    grouped[row["query_id"]].append(row)
            rankings=local_rankings if arm=="local" else base_rankings
            stats=[]
            for qid in ids:
                pool=sorted(grouped[qid],key=lambda r:r["position"])
                require([r["position"] for r in pool]==list(range(1,len(queries[qid].split())+1)),"Incomplete local position coverage")
                vals=[rr(rankings[text_id(r["rewrite"])],qrels[qid]) for r in pool]
                entropy=[r["full_vocab_entropy"] for r in pool]
                order=sorted(range(len(pool)),key=lambda i:(entropy[i],pool[i]["position"]))
                prob_order=sorted(range(len(pool)),key=lambda i:(-pool[i]["label_mean_log_probability"],pool[i]["position"]))
                nw=rr(rankings[text_id(queries[qid])],qrels[qid])
                values={"NoRewrite":nw,"ENTR_top1":vals[order[0]],"ENTR_best3":max(vals[i] for i in order[:3]),"Oracle_all":max(vals),
                        "Uniform_single_exact":math.fsum(vals)/len(vals),"Uniform_best3_exact":uniform_max(vals),
                        "PROB_top1":vals[prob_order[0]],"PROB_best3":max(vals[i] for i in prob_order[:3])}
                row={"domain":domain,"rewriter":arm,"query_id":qid,"group":group_map[qid],"positions":len(pool),"RR":values,
                     "entropy_spearman":correlation(entropy,vals),"selected_positions":[pool[i]["position"] for i in order[:3]]}
                stats.append(row)
                all_query.append(row)
                for candidate,value in zip(pool,vals):
                    candidate_rr.append({"domain":domain,"rewriter":arm,"query_id":qid,"position":candidate["position"],"text_id":text_id(candidate["rewrite"]),"rr":value,"full_vocab_entropy":candidate["full_vocab_entropy"]})
            keys=list(stats[0]["RR"])
            means={key:float(np.mean([s["RR"][key] for s in stats])) for key in keys}
            effects={}
            for left,right in [(key,"NoRewrite") for key in ("ENTR_top1","ENTR_best3","Oracle_all")]+[("ENTR_top1","Uniform_single_exact"),("ENTR_best3","Uniform_best3_exact"),("ENTR_top1","PROB_top1")]:
                effects[left+"-"+right]=summarize([s["RR"][left]-s["RR"][right] for s in stats],groups)
            rates={key:{"beneficial_queries":sum(s["RR"][key]>s["RR"]["NoRewrite"] for s in stats),"harmful_queries":sum(s["RR"][key]<s["RR"]["NoRewrite"] for s in stats)} for key in ("ENTR_top1","ENTR_best3","Oracle_all")}
            correlations=[s["entropy_spearman"] for s in stats if s["entropy_spearman"] is not None]
            old=read(OLD/f"{domain}_{arm}.json")
            errors={new:abs(means[new]-old["mrr"][prior]) for new,prior in [("NoRewrite","nowrite"),("ENTR_top1","entr_top1"),("ENTR_best3","entr_best3"),("Oracle_all","oracle")]}
            errors["spearman"]=abs(float(np.mean(correlations))-old["spearman_entropy"])
            views[domain+"/"+arm]={"queries":len(ids),"groups":len(set(groups)),"mrr":means,"effects":effects,"outcomes":rates,
                                        "entropy_correlation_mean":float(np.mean(correlations)),"correlation_valid_queries":len(correlations),
                                        "old_summary_max_error":max(errors.values()),"old_summary_field_errors":errors,
                                        "old_summary_reproduced_at_1e12":max(errors.values())<1e-12}
            by_arm[arm]=stats
        paired[domain]={key:summarize([b["RR"][key]-a["RR"][key] for a,b in zip(by_arm["local"],by_arm["t5base"])],groups) for key in ("ENTR_top1","ENTR_best3","Oracle_all")}
    writel(OUT/"per_query.jsonl",all_query)
    writel(OUT/"candidate_RR.jsonl",candidate_rr)
    write(OUT/"RESULTS.json",{"created_utc":now(),"role":p["role"],"new_confirmation":False,"risk_certified":False,"views":views,"paired_base_minus_local":paired,"common_original_controls":controls,
                              "not_cross_family":True,"no_H_evaluation":True,"no_source_causal_attribution":True})
    verify()
    verify_seal("RANKINGS_SEALED.json")
    for name,view in views.items():
        print({"view":name,"delta":{k:v["mean"] for k,v in view["effects"].items()},"old_summary_error":view["old_summary_max_error"]},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["freeze","retrieve","evaluate"])
    mode=parser.parse_args().mode
    {"freeze":freeze,"retrieve":retrieve,"evaluate":evaluate}[mode]()
