# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read_jsonl(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def agreement(a,b):
    if len(a)!=len(b) or not a: raise ValueError("paired labels must be equal nonzero length")
    labels=sorted(set(a)|set(b)); n=len(a); po=sum(x==y for x,y in zip(a,b))/n
    ca,cb=Counter(a),Counter(b); pe=sum(ca[k]*cb[k] for k in labels)/(n*n)
    return {"n":n,"exact_agreement":po,"cohen_kappa":(po-pe)/(1-pe) if pe<1 else 1.0}
def paired_labels(rows, item_type, field):
    paired=defaultdict(dict)
    for row in rows:
        if row["item_type"]==item_type:
            paired[row["item_ref"]][row["reviewer"]]=row["judgment"].get(field)
    if any(set(v)!={"A","B"} for v in paired.values()): raise ValueError(f"incomplete A/B pair: {item_type}")
    return [v["A"] for v in paired.values()],[v["B"] for v in paired.values()]
def main():
    manifest=json.loads((ROOT/"SHA256SUMS.json").read_text(encoding="utf-8")); bad=[]
    for rel,expected in manifest.items():
        p=ROOT/rel
        if not p.is_file(): bad.append(f"missing: {rel}"); continue
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected: bad.append(f"hash mismatch: {rel}")
    original=read_jsonl(ROOT/"human_review/data/original_annotations.jsonl")
    supplement=read_jsonl(ROOT/"human_review/data/supplement_annotations.jsonl")
    bounds=read_jsonl(ROOT/"human_review/data/weighted_query_bounds.jsonl")
    assert len(original)==1676,len(original); assert len(supplement)==688,len(supplement); assert len(bounds)==6144,len(bounds)
    oc=Counter((r["reviewer"],r["item_type"]) for r in original)
    expected=Counter({("A","query_clarity"):128,("B","query_clarity"):128,("A","rewrite_fidelity"):383,("B","rewrite_fidelity"):383,("A","code_relevance"):327,("B","code_relevance"):327})
    assert oc==expected,oc
    sc=Counter(r["reviewer"] for r in supplement); assert sc=={"A":344,"B":344},sc
    semantic=agreement(*paired_labels(original,"rewrite_fidelity","fidelity"))
    code=agreement(*paired_labels(original,"code_relevance","relevance"))
    supp=defaultdict(dict)
    for row in supplement: supp[row["panel_ref"]][row["reviewer"]]=row["judgment"]["relevance"]
    supp={k:v for k,v in supp.items() if set(v)=={"A","B"}}
    assert len(supp)==344,len(supp)
    supplement_agreement=agreement([v["A"] for v in supp.values()],[v["B"] for v in supp.values()])
    if bad: raise SystemExit("\n".join(bad))
    print(json.dumps({"passed":True,"hashed_files":len(manifest),"initial_annotations":len(original),"supplement_annotations":len(supplement),"weighted_query_bounds":len(bounds),"initial_rewrite_agreement":semantic,"initial_code_agreement_including_unknown":code,"supplement_paired_panels":len(supp),"supplement_agreement":supplement_agreement},indent=2))
if __name__=="__main__": main()
