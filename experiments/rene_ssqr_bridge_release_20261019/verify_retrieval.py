"""Label-free independent CodeBERT forward and BM25 full-corpus spot audit."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BRIDGE=ROOT/"results/rene_ssqr_bridge_20261019"
OUT=ROOT/"results/rene_ssqr_bridge_release_20261019"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def readl(path):
    with Path(path).open(encoding="utf-8-sig") as h:return [json.loads(line) for line in h if line.strip()]


def require(value,message):
    if not value:raise ValueError(message)


def tokens(text):
    return re.findall(r"[a-z][a-z0-9]*",re.sub(r"([a-z0-9])([A-Z])",r"\1 \2",text).lower())


def main():
    import torch
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    checkpoint=ROOT/"models/codebert-ssqr-pooler-2epoch-20261019"
    model=AutoModel.from_pretrained(checkpoint,local_files_only=True).float().cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    def actual_vector(text):
        encoded=tokenizer([text],padding="max_length",truncation=True,max_length=256,return_tensors="pt")
        with torch.inference_mode():
            value=model(**{k:v.cuda() for k,v in encoded.items()}).pooler_output
        require(value is not None,"missing independently loaded pooler")
        return value[0].float().cpu().numpy()
    rows=[]
    max_vector_error=max_dense_score_error=max_bm25_error=0.
    for domain in ("source","cosqa"):
        corpus_path=BRIDGE/"data/source/corpus.jsonl" if domain=="source" else ROOT/"results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl"
        corpus=readl(corpus_path)
        ids=np.array([r["doc_id"] for r in corpus])
        cache=BRIDGE/f"retrieval_cache/{domain}/CodeBERT"
        texts=read(cache/"new_texts.json")
        qv=np.load(cache/"new_query_vectors.npy")
        code_path=cache/"code_vectors.npy" if domain=="source" else ROOT/"results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy"
        cv=np.load(code_path)
        dense={r["text"]:r for r in readl(BRIDGE/f"rankings/{domain}/CodeBERT.jsonl")}
        bm25={r["text"]:r for r in readl(BRIDGE/f"rankings/{domain}/BM25.jsonl")}
        for index in (0,len(corpus)//2,len(corpus)-1):
            actual=actual_vector(corpus[index]["code_text"])
            error=float(np.max(np.abs(actual-cv[index])))
            max_vector_error=max(max_vector_error,error)
            require(error<5e-5,"actual code embedding differs from cache")
        counts=[Counter(tokens(r["code_text"])) for r in corpus]
        lengths=np.array([sum(c.values()) for c in counts],float)
        average=max(float(lengths.mean()),1.)
        document_frequency=Counter()
        for c in counts:document_frequency.update(c.keys())
        for index in (0,len(texts)//3,2*len(texts)//3,len(texts)-1):
            text=texts[index]
            actual=actual_vector(text)
            vector_error=float(np.max(np.abs(actual-qv[index])))
            max_vector_error=max(max_vector_error,vector_error)
            require(vector_error<5e-5,"actual query embedding differs from cache")
            # FP64 independent reduction checks full-corpus score identities.
            scores=np.einsum("ij,j->i",cv.astype(np.float64),qv[index].astype(np.float64))
            stored=dense[text]
            stored_index=[int(np.flatnonzero(ids==doc)[0]) for doc in stored["ranked_doc_ids"]]
            error=float(np.max(np.abs(scores[stored_index]-stored["scores"])))
            max_dense_score_error=max(max_dense_score_error,error)
            require(error<3e-4,"independent full-corpus dense dot scores differ")
            expected_order=np.lexsort((ids,-scores))[:100]
            strict_dense=ids[expected_order].tolist()==stored["ranked_doc_ids"]
            # A changed order is accepted only inside the measured FP32 rounding band.
            if not strict_dense:
                tolerance=3e-4
                require(all(scores[a]>=scores[b]-2*tolerance for a,b in zip(stored_index,stored_index[1:])),"dense ranking inversions exceed FP32 score tolerance")
                excluded=np.ones(len(ids),bool)
                excluded[stored_index]=False
                require(float(scores[excluded].max())<=float(scores[stored_index[-1]])+2*tolerance,"dense Top100 boundary misses a higher-scoring document")
            lexical=np.zeros(len(corpus),dtype=np.float64)
            for term in sorted(set(tokens(text))):
                idf=math.log1p((len(corpus)-document_frequency[term]+.5)/(document_frequency[term]+.5))
                tf=np.array([c.get(term,0) for c in counts],dtype=float)
                lexical += idf*tf*2.2/(tf+1.2*(.25+.75*lengths/average))
            expected_bm=np.lexsort((ids,-lexical))[:100]
            bm=bm25[text]
            require(ids[expected_bm].tolist()==bm["ranked_doc_ids"],"independent BM25 full ranking differs")
            bm_error=float(np.max(np.abs(lexical[expected_bm]-bm["scores"])))
            max_bm25_error=max(max_bm25_error,bm_error)
            require(bm_error<1e-10,"independent BM25 scores differ")
            rows.append({"domain":domain,"text_sha256":hashlib.sha256(text.encode()).hexdigest(),
                         "actual_query_vector_max_error":vector_error,"dense_score_fp64_max_error":error,
                         "dense_top100_exact_after_FP64_reduction":strict_dense,
                         "bm25_top100_identical":True,"bm25_score_error":bm_error})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    result={"passed":True,"queries":len(rows),"code_forward_cases":6,
            "actual_vector_max_error":max_vector_error,"dense_fp64_dot_max_error":max_dense_score_error,
            "dense_FP64_top100_exact_cases":sum(r["dense_top100_exact_after_FP64_reduction"] for r in rows),
            "remaining_dense_orders_within_FP32_rounding_band":True,"bm25_score_max_error":max_bm25_error,
            "qrels_read":False,"new_policy_or_effects":False,"rows":rows,
            "scope":"actual frozen CodeBERT query/code forwards; independent full-corpus dense reductions and BM25 formula"}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"RETRIEVAL_MODEL_AUDIT.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print({k:v for k,v in result.items() if k!="rows"},flush=True)


if __name__=="__main__":main()
