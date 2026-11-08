"""Archive the license text from Microsoft's official C-UDA repository."""
from __future__ import annotations

import base64
import hashlib
import json

from discover_sources import ROOT, OUT, SOURCES, capture


def main():
    repo = "microsoft/Computational-Use-of-Data-Agreement"
    metadata = json.loads((SOURCES / "cuda_github_repo.json").read_text(encoding="utf-8"))
    captures = []
    record = capture("cuda_commit.json", f"https://api.github.com/repos/{repo}/commits/{metadata['default_branch']}")
    captures.append(record)
    if "error" in record:
        raise RuntimeError(record["error"])
    commit = json.loads((ROOT / record["path"]).read_text(encoding="utf-8"))
    record = capture("cuda_tree.json", f"https://api.github.com/repos/{repo}/git/trees/{commit['commit']['tree']['sha']}?recursive=1")
    captures.append(record)
    if "error" in record:
        raise RuntimeError(record["error"])
    tree = json.loads((ROOT / record["path"]).read_text(encoding="utf-8"))
    selected = [entry for entry in tree["tree"] if entry["type"] == "blob"
                and entry["path"].lower().endswith((".md", ".txt"))
                and any(term in entry["path"].lower() for term in ("readme", "license", "agreement", "c-uda"))]
    files = []
    for entry in selected:
        record = capture(f"cuda_blob_{entry['sha']}.json", f"https://api.github.com/repos/{repo}/git/blobs/{entry['sha']}")
        captures.append(record)
        if "error" in record:
            raise RuntimeError(record["error"])
        obj = json.loads((ROOT / record["path"]).read_text(encoding="utf-8"))
        data = base64.b64decode(obj["content"])
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry["sha"]
        path = OUT / "selected_source/cuda" / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append({"upstream_path": entry["path"], "git_blob_sha1": entry["sha"],
                      "path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(data).hexdigest()})
    report = {"repository": repo, "commit": commit["sha"], "tree_truncated": tree.get("truncated"),
              "files": files, "captures": captures,
              "applicability": "official named license text; CoCLR declares CUDA without an explicit license version or hash"}
    (OUT / "CUDA_LICENSE_SOURCES.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"repository": repo, "commit": commit["sha"], "files": files}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
