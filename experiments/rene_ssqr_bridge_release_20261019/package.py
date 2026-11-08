"""Assemble a portable local review bundle after completed experiment acceptance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"results/rene_ssqr_bridge_20261019"
OUT=ROOT/"results/rene_ssqr_bridge_release_20261019"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def readl(path):
    with Path(path).open(encoding="utf-8-sig") as h:return [json.loads(line) for line in h if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    if not read(SOURCE/"FINAL_ACCEPTANCE.json")["passed"]:raise ValueError("experiment not accepted")
    aggregation=ROOT/"results/rene_ssqr_aggregation_audit_20261019"
    if not read(aggregation/"FINAL_ACCEPTANCE.json")["passed"]:raise ValueError("aggregation diagnostic not accepted")
    if not read(OUT/"RETRIEVAL_MODEL_AUDIT.json")["passed"]:raise ValueError("independent retrieval model audit not accepted")
    destination=OUT/"bundle"
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for name in ("PROTOCOL.json","selections.jsonl","CANDIDATES_SEALED.json","RANKINGS_SEALED.json", "PRE_EFFECT_AUDIT.json","FINAL_ACCEPTANCE.json","EFFECT_OPENING.json"):
        shutil.copy2(SOURCE/name,destination/name)
    for folder in ("data","candidates","rankings","metrics"):
        shutil.copytree(SOURCE/folder,destination/folder,ignore=shutil.ignore_patterns("corpus.jsonl"))
    shutil.copytree(aggregation,destination/"aggregation_audit",ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("replay_bridge.py","verify_actual.py"):
        shutil.copy2(Path(__file__).with_name(name),destination/name)
    (destination/"source_snapshots").mkdir()
    shutil.copytree(ROOT/"experiments/rene_ssqr_bridge_20261019",destination/"source_snapshots/rene_ssqr_bridge_20261019",ignore=shutil.ignore_patterns("__pycache__"))
    protocol=read(SOURCE/"PROTOCOL.json")
    for row in protocol["bound_files"]:
        if row["path"].endswith(".py"):
            target=destination/"source_snapshots/workspace"/row["path"]
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/row["path"],target)
    shutil.copytree(ROOT/"experiments/rene_ssqr_aggregation_audit_20261019",
                    destination/"source_snapshots/workspace/experiments/rene_ssqr_aggregation_audit_20261019",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT/"paper/rene2027/implementation_fidelity.md",destination/"IMPLEMENTATION_FIDELITY.md")
    shutil.copy2(ROOT/"paper/rene2027/replication_results.md",destination/"MANUSCRIPT_RESULTS.md")
    for name in ("READOUT.md","implementation_sensitivity.png"):
        shutil.copy2(SOURCE/name,destination/name)
    for name in ("REFERENCE_SOURCES.json","SSQR_PUBLICATION_METADATA.json"):
        shutil.copy2(OUT/name,destination/name)
    shutil.copy2(ROOT/"paper/rene2027/references_verified.bib",destination/"references_verified.bib")
    shutil.copy2(OUT/"RETRIEVAL_MODEL_AUDIT.json",destination/"RETRIEVAL_MODEL_AUDIT.json")
    model_manifest=read(OUT/"MODEL_BUNDLE_MANIFEST.json")
    model_manifest["archive"]=Path(model_manifest["archive"]).name
    (destination/"MODEL_BUNDLE_MANIFEST.json").write_text(json.dumps(model_manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    cases=[]
    for domain in ("source","cosqa"):
        rows=sorted(readl(SOURCE/f"candidates/{domain}/sample10.jsonl"),key=lambda r:(len(r["masked_query"]),r["query_id"],r["position"]))
        cases += [{"domain":domain,**r} for r in (rows[0],rows[len(rows)//2],rows[-1])]
    actual={"t5_sha256":"7acd4cb8379a1f5297bb978a748bc002b708ef5fd60a8b4737a6f8c7d2ebcaa5",
            "spiece_sha256":sha(ROOT/"models/t5-base/spiece.model"),"cases":cases}
    import numpy as np
    encoder_cases=[]
    for domain in ("source","cosqa"):
        cache=SOURCE/f"retrieval_cache/{domain}/CodeBERT"
        texts=read(cache/"new_texts.json")
        vectors=np.load(cache/"new_query_vectors.npy",mmap_mode="r")
        corpus_path=SOURCE/"data/source/corpus.jsonl" if domain=="source" else ROOT/"results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl"
        corpus=readl(corpus_path)
        code_path=cache/"code_vectors.npy" if domain=="source" else ROOT/"results/rene_cosqa_pilot_20261019/retrieval_cache/CodeBERT/code_vectors.npy"
        code_vectors=np.load(code_path,mmap_mode="r")
        for index in (0,len(texts)//3,2*len(texts)//3,len(texts)-1):
            encoder_cases.append({"domain":domain,"role":"query","text":texts[index],"vector":vectors[index].tolist()})
        for index in (0,len(corpus)//2,len(corpus)-1):
            encoder_cases.append({"domain":domain,"role":"code","text":corpus[index]["code_text"],"vector":code_vectors[index].tolist()})
    actual["codebert"]={"sha256":"0d533c04602916c22d1f13d282056f081b6bd04f9866ccc1ded46a0516e13e3d","cases":encoder_cases}
    (destination/"actual_cases.json").write_text(json.dumps(actual,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    packages=("torch","transformers","numpy","scipy","sentencepiece","threadpoolctl","safetensors","tokenizers")
    requirements=["--extra-index-url https://download.pytorch.org/whl/cu128"]
    requirements += [name+"=="+importlib.metadata.version(name) for name in packages]
    (destination/"requirements-inference.txt").write_text("\n".join(requirements)+"\n",encoding="utf-8")
    (destination/"README.md").write_text('''# SSQR source bridge: local review artifact

Evidence role: supplementary post-confirmation diagnostic. Source AdvTest and CoSQA development data are historically consumed. These tables do not replace the frozen CoSQA confirmation or establish query-only causation.

## Independent cached replay

From this directory, with Python 3.10 or newer and no third-party packages:

```
python replay_bridge.py --output replay_output
```

This verifies bundle hashes, independently reconstructs ENTR/PROB/20 RAND selections, computes every RR with exact rational arithmetic, and rebuilds all 56 result-table views plus 28 aggregation-diagnostic views. It uses only files in this bundle. It does not import workspace code or access model weights. The exported per-query metrics and original result tables are references for verification, not inputs to the recomputed RR; query identities and denominators come from the outcome-blind frozen plan.

## Real model inference smoke check

Install the exact versions listed in PROTOCOL.json. Extract the separate `RENE-frozen-inference-models.zip` beside these scripts, preserving the `models/` directory, then run:

```
python verify_actual.py --t5 models/t5-cqc-codenn-1m/best --spiece models/t5-base/spiece.model --codebert models/codebert-ssqr-pooler-2epoch-20261019 --output ACTUAL_AUDIT.json
```

The required checkpoint/vocabulary hashes are in actual_cases.json. This regenerates six sampled query-position cases with fixed RNG and independently checks FP64 full-vocabulary teacher-forcing entropy on a CUDA GPU. It also reloads the frozen CodeBERT and checks eight query and six code pooler vectors. Exact installed versions are provided in requirements-inference.txt; inference was validated with Python 3.10.11 and CUDA 12.8 PyTorch. This is distinct from cached replay and from retraining.

## Full experiment provenance and acquisition

The immutable PROTOCOL contains package versions, exact model/data/source hashes, all query identities, six finite profiles, RNG and ranking contracts, and the protected historical confirmation inputs. Source snapshots include the workspace runner and its bound helper sources in their original relative layout. Full generation and retrieval require the separately acquired data and historical supporting caches listed by hash; the small cached bundle is not a one-command complete-workspace reconstruction. It does not claim that arbitrary new model training will recover the same weights.

The primary source corpus is the 19,210 original CodeXGLUE AdvTest test functions, in author test.txt order; the CoSQA corpus contains 6,267 functions. Corpus texts and model weights are not included in this small cached review archive. Frozen inference weights and tokenizer files are provided in the separate local model archive, excluding optimizer state. Source lineage and query/target identities are provided for audit. The independent aggregation audit records a duplicate-URL overwrite in the public evaluator and compares three aggregation definitions without asserting which unpublished path produced the original article's tables.

Data sources: https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-Adv and https://github.com/Jun-jie-Huang/CoCLR . Reference method: https://github.com/RedSmallPanda/SSQR/tree/7fe5d780740eb4348500658ce6decf188a542d2b . CoSQA query data are governed by the source Computational Use of Data Agreement and source code retains its original repository licenses; six short code inference cases retain those same source terms. This is a local acceptance bundle; public distribution requires the existing project license review. No external publication is performed by packaging.
''',encoding="utf-8")
    files=[{"path":p.relative_to(destination).as_posix(),"sha256":sha(p)} for p in sorted(destination.rglob("*")) if p.is_file()]
    manifest={"schema_version":1,"role":protocol["role"],"files":files,
              "public_release":False,"cached_replay_is_standalone":True,"full_retraining_claimed":False}
    (destination/"BUNDLE_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    archive=shutil.make_archive(str(OUT/"RENE-SSQR-bridge-cached-review"),"zip",root_dir=destination)
    result={"bundle":str(destination),"zip":archive,"zip_sha256":sha(archive),"files":len(files)}
    (OUT/"PACKAGING.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(result,flush=True)


if __name__=="__main__":main()
