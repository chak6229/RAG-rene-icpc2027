"""Fetch inspectable files by immutable Git blob IDs, avoiding archive delays."""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json

from discover_sources import ROOT, OUT, SOURCES, capture


WANTED = {
    "ssqr": ["README.md", "Pretraining/README.md", "Pretraining/pretrain.py", "Pretraining/requirements.txt",
             "Query Reformulation/README.md", "Query Reformulation/code/generate.py", "Query Reformulation/code/run.py",
             "Query Reformulation/code/model.py", "Query Reformulation/dataset/preprocess.py", "Query Reformulation/evaluator/evaluator.py"],
    "coclr": ["README.md", "LICENSE", "data/search/split_code_for_retrieval.py", "code_search/utils.py",
              "code_search/run_siamese_test.py", "code_search/models.py", "data/search/cosqa-retrieval-dev-500.json",
              "data/search/cosqa-retrieval-test-500.json", "data/search/code_idx_map.txt"],
    "codesearchnet": ["LICENSE", "README.md"],
}
REPOS = {"ssqr": "RedSmallPanda/SSQR", "coclr": "Jun-jie-Huang/CoCLR", "codesearchnet": "github/CodeSearchNet"}


def retrieve(item):
    key, entry = item
    record = capture(f"{key}_blob_{entry['sha']}.json", f"https://api.github.com/repos/{REPOS[key]}/git/blobs/{entry['sha']}")
    record.update(repository=REPOS[key], upstream_path=entry["path"], git_blob_sha1=entry["sha"])
    if "error" not in record:
        payload = json.loads((SOURCES / record["name"]).read_text(encoding="utf-8"))
        if payload.get("encoding") != "base64":
            raise ValueError("unexpected blob encoding")
        data = base64.b64decode(payload["content"])
        git_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if git_sha != entry["sha"]:
            raise ValueError("blob content mismatch")
        path = OUT / "selected_source" / key / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        record.update(content_path=path.relative_to(ROOT).as_posix(), content_bytes=len(data), content_sha256=hashlib.sha256(data).hexdigest(), git_blob_verified=True)
    return record


def main():
    items = []
    for key, wanted in WANTED.items():
        tree = json.loads((SOURCES / f"{key}_tree.json").read_text(encoding="utf-8"))
        for path in wanted:
            found = [entry for entry in tree["tree"] if entry["path"] == path]
            if not found:
                print(json.dumps({"missing_requested_file": path, "repository": key}))
            items.extend((key, entry) for entry in found)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        records = list(executor.map(retrieve, items))
    (OUT / "OFFICIAL_SOURCE_FILES.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    for record in records:
        print(json.dumps({key: record[key] for key in ("repository", "upstream_path", "content_bytes", "content_sha256", "error") if key in record}, ensure_ascii=False))


if __name__ == "__main__":
    main()
