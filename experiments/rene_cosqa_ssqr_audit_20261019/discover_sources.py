"""Capture public provenance and local schemas; never run model inference."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rene_cosqa_ssqr_audit_20261019"
SOURCES = OUT / "sources"
SOURCES.mkdir(parents=True, exist_ok=True)


def capture(name: str, url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "research-provenance-audit/1.0", "Accept": "*/*"})
    record = {"name": name, "url": url, "requested_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            data = response.read()
            path = SOURCES / name
            path.write_bytes(data)
            record.update(status=response.status, final_url=response.url, bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest(), path=path.relative_to(ROOT).as_posix(),
                          content_type=response.headers.get("Content-Type"),
                          etag=response.headers.get("ETag"))
    except Exception as error:
        record.update(error=type(error).__name__ + ": " + str(error))
    return record


def main() -> None:
    article = (ROOT / "results/rene_thesis_review_20261019/SSQR.html").read_text(encoding="utf-8")
    links = sorted(set(re.findall(r'href=["\']([^"\']+)["\']', article)))
    relevant = [link for link in links if any(word in link.lower() for word in
                ("github.com", "zenodo", "figshare", "huggingface", "drive.google", "replication", "artifact"))]
    print(json.dumps({"ssqr_article_external_artifact_links": relevant}, ensure_ascii=False))
    (OUT / "SSQR_ARTIFACT_LINKS.json").write_text(json.dumps(relevant, indent=2), encoding="utf-8")
    urls = [
        ("codexglue_repo.json", "https://api.github.com/repos/microsoft/CodeXGLUE"),
        ("coclr_repo.json", "https://api.github.com/repos/Jun-jie-Huang/CoCLR"),
        ("cosqa_arxiv_abs.html", "https://arxiv.org/abs/2105.13239"),
        ("cosqa_arxiv_html.html", "https://arxiv.org/html/2105.13239"),
        ("cosqa_coclr_README.md", "https://raw.githubusercontent.com/Jun-jie-Huang/CoCLR/main/README.md"),
        ("cosqa_coclr_README_master.md", "https://raw.githubusercontent.com/Jun-jie-Huang/CoCLR/master/README.md"),
    ]
    repos = set()
    for link in relevant:
        match = re.match(r"https?://github\.com/([^/]+/[^/#?]+)", link)
        if match and match.group(1).lower() not in {"arxiv/html_feedback", "arxiv/arxiv-browse"}:
            repos.add(match.group(1).removesuffix(".git"))
    for number, repo in enumerate(sorted(repos)):
        urls.append((f"ssqr_repo_{number}.json", f"https://api.github.com/repos/{repo}"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        records = list(executor.map(lambda item: capture(*item), urls))
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
    (OUT / "SOURCE_DISCOVERY.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    paths = [
        "imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-all.json",
        "imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-train.json",
        "imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-dev.json",
        "imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/data/test_webquery.json",
        "data/codexglue/training/train.jsonl",
        "data/codexglue/advtest/valid.jsonl",
        "data/codexglue/advtest/test.jsonl",
        "imports/dataset/test_code.jsonl",
    ]
    schemas = []
    for relative in paths:
        path = ROOT / relative
        if not path.exists():
            schemas.append({"path": relative, "exists": False})
            continue
        if path.suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            rows = obj if isinstance(obj, list) else list(obj.values())
            example = rows[0]
            count = len(rows)
        else:
            with path.open(encoding="utf-8") as handle:
                example = json.loads(next(handle))
            count = None
        schema = {"path": relative, "bytes": path.stat().st_size, "count_if_array": count,
                  "keys": list(example) if isinstance(example, dict) else None,
                  "first_row_types": {key: type(value).__name__ for key, value in example.items()}}
        for key in ("idx", "index", "retrieval_idx", "label", "url", "repo", "language"):
            if key in example:
                schema[key] = example[key]
        schemas.append(schema)
    (OUT / "LOCAL_SCHEMAS.json").write_text(json.dumps(schemas, ensure_ascii=False, indent=2), encoding="utf-8")
    for schema in schemas:
        print(json.dumps(schema, ensure_ascii=False))


if __name__ == "__main__":
    main()
