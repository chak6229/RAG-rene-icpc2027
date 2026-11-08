"""Pin official repository commits and collect source snapshots for audit."""
from __future__ import annotations

import concurrent.futures
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from discover_sources import ROOT, OUT, SOURCES, capture


REPOS = {
    "ssqr": ("RedSmallPanda/SSQR", "ssqr_repo_0.json"),
    "coclr": ("Jun-jie-Huang/CoCLR", "coclr_repo.json"),
    "codexglue": ("microsoft/CodeXGLUE", "codexglue_repo.json"),
    "codesearchnet": ("github/CodeSearchNet", "ssqr_repo_2.json"),
}
DOWNLOAD_ARCHIVES = False


def cached_capture(name, url):
    path = SOURCES/name
    if path.exists():
        data = path.read_bytes()
        return {"name":name,"url":url,"status":"loaded_from_this_session_capture",
                "bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"path":path.relative_to(ROOT).as_posix()}
    return capture(name,url)


def git_blob_hash(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_repository(key: str, repo: str, meta_name: str) -> dict:
    meta = json.loads((SOURCES / meta_name).read_text(encoding="utf-8"))
    branch = meta["default_branch"]
    commit_record = cached_capture(f"{key}_commit.json", f"https://api.github.com/repos/{repo}/commits/{branch}")
    record = {"key": key, "repository": repo, "metadata": meta_name, "captures": [commit_record]}
    if "error" in commit_record:
        return record
    commit = json.loads((SOURCES / f"{key}_commit.json").read_text(encoding="utf-8"))
    sha = commit["sha"]
    record["commit"] = sha
    tree_record = cached_capture(f"{key}_tree.json", f"https://api.github.com/repos/{repo}/git/trees/{commit['commit']['tree']['sha']}?recursive=1")
    record["captures"].append(tree_record)
    if "error" not in tree_record:
        tree = json.loads((SOURCES / f"{key}_tree.json").read_text(encoding="utf-8"))
        record["tree_truncated"] = tree.get("truncated")
        record["files"] = [{k: item.get(k) for k in ("path", "type", "sha", "size")} for item in tree.get("tree", [])]
    if DOWNLOAD_ARCHIVES and key in {"ssqr", "coclr"}:
        archive_record = capture(f"{key}_source.zip", f"https://api.github.com/repos/{repo}/zipball/{sha}")
        record["captures"].append(archive_record)
        if "error" not in archive_record:
            target = OUT / "author_source" / key
            target.mkdir(parents=True, exist_ok=True)
            extracted = []
            with zipfile.ZipFile(SOURCES / f"{key}_source.zip") as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    relative = Path(*Path(member.filename).parts[1:])
                    destination = (target / relative).resolve()
                    if not destination.is_relative_to(target.resolve()):
                        raise RuntimeError("unsafe archive member")
                    if member.file_size > 60_000_000:
                        raise RuntimeError(f"unexpected large source member: {member.filename}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    data = archive.read(member)
                    destination.write_bytes(data)
                    extracted.append({"path": relative.as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            record["extracted"] = extracted
    return record


def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda item: fetch_repository(item[0], *item[1]), REPOS.items()))
    for record in records:
        summary = {key: value for key, value in record.items() if key not in {"files", "extracted"}}
        summary["file_count"] = len(record.get("files", []))
        summary["extracted_count"] = len(record.get("extracted", []))
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if record["key"] in {"ssqr", "coclr"}:
            print(json.dumps({"key": record["key"], "file_paths": [x["path"] for x in record.get("files", [])]}, ensure_ascii=False))
    (OUT / "REPOSITORY_SNAPSHOTS.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    codex = next(item for item in records if item["key"] == "codexglue")
    local_matches = []
    for item in codex.get("files", []):
        if item["type"] == "blob" and (item["path"].startswith("Text-Code/NL-code-search-WebQuery/") or item["path"] == "LICENSE"):
            path = ROOT / "imports/CodeXGLUE-main" / item["path"]
            if path.exists():
                actual = git_blob_hash(path)
                local_matches.append({"path": path.relative_to(ROOT).as_posix(), "upstream_commit": codex["commit"],
                                      "upstream_blob_sha1": item["sha"], "local_git_blob_sha1": actual,
                                      "exact_content_match": item["sha"] == actual,
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "LOCAL_UPSTREAM_MATCHES.json").write_text(json.dumps(local_matches, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"codexglue_local_compared": len(local_matches), "mismatches": [item["path"] for item in local_matches if not item["exact_content_match"]]}, ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--download-archives",action="store_true")
    DOWNLOAD_ARCHIVES=parser.parse_args().download_archives
    main()
