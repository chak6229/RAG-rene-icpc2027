"""Accept the actual ZIPs in an independent temporary directory, preserving evidence."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/rene_ssqr_bridge_release_20261019"
BRIDGE=ROOT/"results/rene_ssqr_bridge_20261019"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def safe_extract(archive,destination):
    destination=destination.resolve()
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as source:
        names=set()
        for info in source.infolist():
            target=(destination/info.filename).resolve()
            require(target.is_relative_to(destination),"archive member escapes destination")
            require(target not in names,"duplicate archive member")
            require((info.external_attr>>16)&0o170000!=0o120000,"archive contains a symlink")
            names.add(target)
        source.extractall(destination)


def execute(script,arguments,cwd,label):
    started=time.monotonic()
    command=[sys.executable,"-I","-X","utf8",str(script),*map(str,arguments)]
    environment=dict(os.environ)
    environment["HF_HUB_OFFLINE"]="1"
    result=subprocess.run(command,cwd=cwd,env=environment,capture_output=True,text=True,encoding="utf-8")
    (OUT/f"CLEAN_{label}_STDOUT.txt").write_text(result.stdout,encoding="utf-8")
    (OUT/f"CLEAN_{label}_STDERR.txt").write_text(result.stderr,encoding="utf-8")
    require(result.returncode==0,f"external {label} failed; preserved stdout and stderr")
    print({"step":label,"exit_code":result.returncode,"seconds":round(time.monotonic()-started,2)},flush=True)


def main():
    started=time.monotonic()
    acceptance_path=OUT/"RELEASE_ACCEPTANCE.json"
    require(not acceptance_path.exists(),"release already accepted; do not overwrite")
    package=read(OUT/"PACKAGING.json")
    models=read(OUT/"MODEL_BUNDLE_MANIFEST.json")
    require(sha(package["zip"])==package["zip_sha256"],"cached ZIP hash mismatch")
    require(sha(models["archive"])==models["archive_sha256"],"model ZIP hash mismatch")
    require(shutil.disk_usage(tempfile.gettempdir()).free>4_500_000_000,"insufficient free space for clean acceptance")
    directory=Path(tempfile.mkdtemp(prefix="rene-ssqr-bridge-release-"))
    require(not directory.resolve().is_relative_to(ROOT.resolve()),"acceptance directory is inside workspace")
    bundle=directory/"bundle"
    model_root=directory/"model_bundle"
    (OUT/"CLEAN_ACCEPTANCE_START.json").write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),
        "directory":str(directory),"cached_zip_sha256":package["zip_sha256"],"model_zip_sha256":models["archive_sha256"]},indent=2)+"\n",encoding="utf-8")
    safe_extract(package["zip"],bundle)
    execute(bundle/"replay_bridge.py",["--bundle",bundle,"--output",directory/"cached_replay"],directory,"CACHE")
    replay=read(directory/"cached_replay/REPLAY_AUDIT.json")
    require(replay["passed"] and replay["table_views"]==56 and replay["aggregation_views"]==28,
            "external cached replay has an unexpected scope")
    safe_extract(models["archive"],model_root)
    for row in models["files"]:
        path=(model_root/row["path"]).resolve()
        require(path.is_relative_to(model_root.resolve()),"model manifest path escapes directory")
        require(path.stat().st_size==row["bytes"] and sha(path)==row["sha256"],"extracted model identity differs")
    print({"step":"MODEL_EXTRACTION","file_hash_checks":len(models["files"])},flush=True)
    execute(bundle/"verify_actual.py",["--bundle",bundle,
        "--t5",model_root/"models/t5-cqc-codenn-1m/best",
        "--spiece",model_root/"models/t5-base/spiece.model",
        "--codebert",model_root/"models/codebert-ssqr-pooler-2epoch-20261019",
        "--output",directory/"ACTUAL_AUDIT.json"],directory,"ACTUAL")
    actual=read(directory/"ACTUAL_AUDIT.json")
    require(actual["passed"] and actual["actual_cases"]==6 and actual["codebert"]["actual_cases"]==14,
            "external real-model check has an unexpected scope")
    protocol=read(BRIDGE/"PROTOCOL.json")
    for row in protocol["bound_files"]+protocol["protected_files"]:
        require(sha(ROOT/row["path"])==row["sha256"],"frozen source, model, confirmation, or formal pointer changed")
    retrieval=read(OUT/"RETRIEVAL_MODEL_AUDIT.json")
    require(retrieval["passed"],"full-corpus retrieval spot audit failed")
    result={"passed":True,"created_utc":datetime.now(timezone.utc).isoformat(),"directory":str(directory),
            "independent_directory_outside_workspace":True,"python_isolated_mode":True,
            "cached_bundle_zip_sha256":package["zip_sha256"],"model_zip_sha256":models["archive_sha256"],
            "extracted_model_file_hashes":len(models["files"]),"cached_replay":replay,"actual_model_inference":actual,
            "full_corpus_retrieval_spot_audit":retrieval,
            "frozen_input_hash_checks":len(protocol["bound_files"]),"five_formal_pointers_unchanged":True,
            "old_confirmation_unchanged":True,"scientific_role":"post_confirmation_supplementary_diagnostic",
            "public_release":False,"new_confirmation_claimed":False,"full_retraining_claimed":False,
            "seconds":round(time.monotonic()-started,2)}
    acceptance_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print({"passed":True,"directory":str(directory),"seconds":result["seconds"]},flush=True)


if __name__=="__main__":
    main()
