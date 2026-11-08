"""Independent identity/RR arithmetic and clean-directory replay acceptance."""
from __future__ import annotations

import json
import math
import random
import shutil
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from .common import (
    ROOT, PACKAGE, OUT, DEVELOPMENT, PROFILES, T5, SOURCE, read, readl, write,
    now, sha, record, require, verify_frozen, verify_seal, pools_for, text_id, position_seed,
)


def before_effects():
    protocol = verify_frozen()
    verify_seal("CANDIDATES_SEALED.json")
    verify_seal("RANKINGS_SEALED.json")
    require(not (OUT / "metrics").exists(), "pre-effect check ran after new outcomes")
    query_maps = {"source": {q["query_id"]:q["query"] for q in readl(OUT / "data/source/queries.jsonl")},
                  "cosqa": {q["query_id"]:q["query"] for q in readl(DEVELOPMENT / "data/queries.jsonl")}}
    roles = read(DEVELOPMENT / "ROLE_MANIFEST.json")
    require(not set(query_maps["cosqa"]) & set(roles["confirmation_pool_query_ids"]), "H contamination")
    checks, candidate_count = [], 0
    for domain in query_maps:
        for profile in PROFILES:
            pools = pools_for(domain, profile, sensitivity=True)
            for qid, rows in pools.items():
                n = len(query_maps[domain][qid].split())
                expected = set(range(n+1)) if PROFILES[profile]["positions"] == "paper_0_to_n" else set(range(1,n+1))
                require({r["position"] for r in rows} == expected, "insertion position coverage mismatch")
                for row in rows:
                    require(all(math.isfinite(row[field]) for field in ("full_vocab_entropy", "label_mean_log_probability")), "nonfinite candidate scores")
                    require(row["label_token_count"] >= 3, "wrapper count missing")
                    if PROFILES[profile]["sample"]:
                        require(row["position_rng_seed"] == position_seed(qid,row["position"]), "position RNG mismatch")
                        require(len(row["generated_ids"]) <= 10, "sample10 violated total decoder length")
                    candidate_count += 1
            if profile in ("greedy20_source", "greedy10_source", "sample10_source"):
                all_profile = profile.replace("_source", "_all")
                full = pools_for(domain, all_profile, sensitivity=True)
                for qid in pools:
                    require(pools[qid] == [r for r in full[qid] if r["position"] != 0], "common positions changed in position ablation")
            checks.append(f"{domain}/{profile}: full position coverage and common-draw identity")
        required_texts = {text_id(q) for q in query_maps[domain].values()}
        for path in (OUT / f"candidates/{domain}").glob("*.jsonl"):
            required_texts.update(text_id(row["rewrite"]) for row in readl(path))
        if domain == "cosqa":
            required_texts.update(text_id(row["rewrite"]) for row in readl(DEVELOPMENT / "candidates.jsonl"))
        for retriever in protocol["retrievers"]:
            rows = readl(OUT / f"rankings/{domain}/{retriever}.jsonl")
            require(len(rows) == len({r["text_id"] for r in rows}) and {r["text_id"] for r in rows} == required_texts,
                    "ranking/query denominator coverage mismatch")
            for row in rows:
                require(row["text_id"] == text_id(row["text"]), "text identity mismatch")
                require(len(row["ranked_doc_ids"]) == len(set(row["ranked_doc_ids"])) == 100, "ranking depth mismatch")
                pairs = list(zip(row["ranked_doc_ids"],row["scores"]))
                require(all(a[1] > b[1] or (a[1] == b[1] and a[0] < b[0]) for a,b in zip(pairs,pairs[1:])), "unstable ranking tie order")
            checks.append(f"{domain}/{retriever}: complete unique text rankings/depth/order")
    result = {"passed":True,"created_utc":now(),"protocol_sha256":sha(OUT / "PROTOCOL.json"),
              "checks":checks,"candidate_profile_rows_checked":candidate_count,"outcomes_computed":False,
              "H_queries_in_sensitivity":0,"source_corpus_scope":19210}
    write(OUT / "PRE_EFFECT_AUDIT.json",result)
    return result


def independent_fraction_replay():
    protocol = read(OUT / "PROTOCOL.json")
    metrics = readl(OUT / "metrics/per_query.jsonl")
    selections = {(r["domain"],r["scope"],r["profile"],r["mask"],r["query_id"]):r for r in readl(OUT / "selections.jsonl")}
    max_error = 0.
    comparisons = 0
    for domain in ("source","cosqa"):
        queries = {r["query_id"]:r["query"] for r in readl(OUT / f"data/{domain}/queries.jsonl")}
        targets = {r["query_id"]:r["doc_id"] for r in readl(OUT / f"data/private/{domain}_qrels.jsonl")}
        for retriever in protocol["retrievers"]:
            ranks = {r["text_id"]:r["ranked_doc_ids"] for r in readl(OUT / f"rankings/{domain}/{retriever}.jsonl")}
            views = {}
            for scope, profiles in (("main",["greedy20_source"]),("sensitivity",list(PROFILES))):
                for profile in profiles:
                    views[(scope,profile)] = {qid:{r["position"]:r["rewrite"] for r in rows}
                        for qid,rows in pools_for(domain,profile,sensitivity=scope=="sensitivity").items()}
            for row in metrics:
                if row["domain"] != domain or row["retriever"] != retriever:
                    continue
                qid = row["query_id"]
                def exact_rr(text):
                    ordered = ranks[text_id(text)]
                    return Fraction(1,ordered.index(targets[qid])+1) if targets[qid] in ordered else Fraction(0)
                pool = views[(row["scope"],row["profile"])][qid]
                selection = selections[(domain,row["scope"],row["profile"],row["mask"],qid)]
                values = {p:exact_rr(text) for p,text in pool.items()}
                entr = max(values[p] for p in selection["ENTR"])
                prob = max(values[p] for p in selection["PROB"])
                rand = sum((max(values[p] for p in positions) for positions in selection["RAND"]),Fraction(0))/20
                expected = {"NoRewrite":exact_rr(queries[qid]),"ENTR_top1":values[selection["ENTR"][0]],
                            "ENTR_best3":entr,"PROB_best3":prob,"RAND20_best3":rand,
                            "ENTR_minus_RAND":entr-rand,"ENTR_minus_PROB":entr-prob}
                for key,value in expected.items():
                    max_error = max(max_error,abs(float(value)-row[key]))
                    comparisons += 1
    require(max_error < 1e-12,"independent Fraction replay mismatch")
    return {"comparisons":comparisons,"maximum_error":max_error,"passed":True}


def clean_replay():
    from .analyze import calculate
    # This directory is outside both the workspace and the source results tree.
    temp_root = Path(tempfile.mkdtemp(prefix="rene-bridge-cached-replay-"))
    for name in ("PROTOCOL.json","selections.jsonl"):
        shutil.copy2(OUT / name,temp_root / name)
    for folder in ("data","candidates","rankings"):
        shutil.copytree(OUT / folder,temp_root / folder)
    replay,_ = calculate(temp_root)
    original = read(OUT / "metrics/RESULTS.json")
    require(replay == original,"clean-directory cached tables differ")
    write(temp_root / "REPLAY_RESULT.json",replay)
    return {"passed":True,"directory":str(temp_root),"comparison":"all result fields exactly equal",
            "scope":"cached computational replay; model/data acquisition documented separately",
            "result_sha256":sha(temp_root / "REPLAY_RESULT.json")}


def real_model_spotcheck():
    import numpy as np
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from experiments.rene_ssqr_alignment_20261019.t5 import load_tokenizer, _inputs
    from .generate import generate_positions
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = load_tokenizer(T5)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5,local_files_only=True).float().cuda().eval()
    chosen = []
    for domain in ("source","cosqa"):
        rows = readl(OUT / f"candidates/{domain}/sample10.jsonl")
        ordered = sorted(rows,key=lambda r:(len(r["masked_query"]),r["query_id"],r["position"]))
        chosen += [(domain,r) for r in (ordered[0],ordered[len(ordered)//2],ordered[-1])]
    max_entropy_error = 0.
    for domain,row in chosen:
        inputs = _inputs(tokenizer,[row["masked_query"]],"cuda")
        labels = tokenizer(f'<extra_id_0> {row["span"]} <extra_id_1>',return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            logits = model(**inputs,labels=labels).logits[0].double().cpu()
        # Independent FP64 probability normalization: no production metric helper.
        log_norm = torch.logsumexp(logits,dim=-1,keepdim=True)
        log_p = logits-log_norm
        entropy = float(-(log_p.exp()*log_p).sum(-1).mean())
        max_entropy_error = max(max_entropy_error,abs(entropy-row["full_vocab_entropy"]))
        query = {q["query_id"]:q["query"] for q in readl(OUT / f"data/{domain}/queries.jsonl")}[row["query_id"]]
        generated = generate_positions(model,tokenizer,row["query_id"],query,positions={row["position"]},sample=True,max_new_tokens=9)[0]
        require(generated["generated_ids"] == row["generated_ids"],"actual per-position sampling regeneration mismatch")
    require(max_entropy_error < 2e-5,"FP64 entropy spotcheck mismatch")
    return {"passed":True,"actual_candidates":len(chosen),"fp64_entropy_max_error":max_entropy_error,
            "same_per_position_sampling_ids":True}


def after_effects():
    verify_frozen(private=True)
    verify_seal("CANDIDATES_SEALED.json")
    verify_seal("RANKINGS_SEALED.json")
    fraction = independent_fraction_replay()
    replay = clean_replay()
    actual = real_model_spotcheck()
    result = {"passed":True,"created_utc":now(),"independent_fraction_rr":fraction,
              "clean_directory_replay":replay,"actual_model_checks":actual,
              "five_formal_pointers_unchanged":True,"old_confirmation_unchanged":True,
              "risk_certified":False,"new_confirmation_claimed":False}
    write(OUT / "FINAL_ACCEPTANCE.json",result)
    return result
