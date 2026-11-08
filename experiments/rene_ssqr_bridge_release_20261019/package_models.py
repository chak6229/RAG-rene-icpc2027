"""Local frozen inference-weight bundle; omit optimizer state and never move originals."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/rene_ssqr_bridge_release_20261019"


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/"MODEL_BUNDLE_MANIFEST.json"
    archive=OUT/"RENE-frozen-inference-models.zip"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha(archive)!=manifest["archive_sha256"]:raise ValueError("existing local model archive changed")
        print({"already_complete":True,"archive":str(archive)},flush=True)
        return
    files=[]
    for folder in (ROOT/"models/t5-cqc-codenn-1m/best",ROOT/"models/codebert-ssqr-pooler-2epoch-20261019"):
        files.extend(p for p in folder.iterdir() if p.is_file() and p.suffix!=".pt")
    files.append(ROOT/"models/t5-base/spiece.model")
    records=[{"path":p.relative_to(ROOT).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(files)]
    record_by_path={r["path"]:r for r in records}
    if record_by_path["models/t5-cqc-codenn-1m/best/model.safetensors"]["sha256"]!="7acd4cb8379a1f5297bb978a748bc002b708ef5fd60a8b4737a6f8c7d2ebcaa5":raise ValueError("wrong T5 weights")
    if record_by_path["models/codebert-ssqr-pooler-2epoch-20261019/model.safetensors"]["sha256"]!="0d533c04602916c22d1f13d282056f081b6bd04f9866ccc1ded46a0516e13e3d":raise ValueError("wrong CodeBERT weights")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_STORED,allowZip64=True) as bundle:
        for path in sorted(files):bundle.write(path,path.relative_to(ROOT).as_posix())
        bundle.writestr("INFERENCE_WEIGHTS.json",json.dumps({"files":records,"optimizer_state_included":False,
            "author_checkpoint":False,"role":"frozen independently trained reconstruction weights"},indent=2))
    result={"archive":str(archive),"archive_sha256":sha(archive),"archive_bytes":archive.stat().st_size,
            "files":records,"original_weights_unchanged":True,"public_release":False}
    manifest_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print({"archive":str(archive),"bytes":archive.stat().st_size,"files":len(records)},flush=True)


if __name__=="__main__":main()
