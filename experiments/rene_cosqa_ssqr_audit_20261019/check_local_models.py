"""Record actual checkpoint identity and training metadata without inference."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from discover_sources import ROOT, OUT


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda:handle.read(2**23),b""):
            h.update(part)
    return h.hexdigest()


def main():
    import torch
    paths=[
        "models/t5-base/model.safetensors",
        "models/t5-cqc-codenn-1m/best/model.safetensors",
        "models/t5-cqc-codenn-1m/checkpoint-93750/model.safetensors",
        "models/codebert-base/model.safetensors",
        "models/codebert-codexglue-2epoch/checkpoint-3000/model.safetensors",
        "models/codebert-codexglue-full-2epoch/checkpoint-15740/model.safetensors",
    ]
    weights=[]
    for relative in paths:
        p=ROOT/relative
        if not p.exists():
            weights.append({"path":relative,"exists":False})
            continue
        record={"path":relative,"bytes":p.stat().st_size,"sha256":digest(p)}
        config=p.parent/"config.json"
        if config.exists():
            data=json.loads(config.read_text(encoding="utf-8"))
            record["architecture"]={k:data[k] for k in ["model_type","architectures","d_model","num_layers","hidden_size","num_hidden_layers","vocab_size","torch_dtype","dtype"] if k in data}
        weights.append(record)
        print(json.dumps(record),flush=True)
    states=[]
    for relative in ["models/codebert-codexglue-2epoch/checkpoint-3000/trainer_state.pt", "models/codebert-codexglue-full-2epoch/checkpoint-15740/trainer_state.pt"]:
        p=ROOT/relative
        obj=torch.load(p,map_location="cpu",weights_only=False,mmap=True)
        record={"path":relative,"metadata":{k:obj[k] for k in ["signature","global_step","step","epoch","best_step","best_mrr","batch_index"] if k in obj}}
        states.append(record)
        del obj
    metadata={}
    for relative in ["models/t5-cqc-codenn-1m/training_config.json","models/t5-cqc-codenn-1m/training_complete.json",
        "models/t5-cqc-codenn-1m/best/best_checkpoint.json","models/t5-cqc-codenn-1m/checkpoint-93750/checkpoint_complete.json",
        "models/codebert-codexglue-full-2epoch/training_config.json","models/codebert-codexglue-full-2epoch/training_complete.json",
        "models/t5-cqc-codenn-1m/best/generation_config.json"]:
        p=ROOT/relative
        metadata[relative]={"sha256":digest(p),"value":json.loads(p.read_text(encoding="utf-8"))}
    protected = []  # Unrelated author-workspace registries are not part of this release.
    protected_snapshot={relative:digest(ROOT/relative) for relative in protected}
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"weights":weights,"trainer_metadata":states,"configuration_files":metadata,
        "protected_snapshot":protected_snapshot,"scope":"checkpoint/header/metadata only; no model inference, training or model replacement"}
    (OUT/"LOCAL_CHECKPOINTS.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"trainer_metadata":states,"protected_files":len(protected_snapshot)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
